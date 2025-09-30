import asyncio
import time
import os
import logging
from typing import Dict, List, Optional

import structlog
import torch
from opentelemetry import metrics
from pymongo import MongoClient

from tueri import input_scanners, output_scanners
from tueri.input_scanners.ban_competitors import MODEL_V1 as BAN_COMPETITORS_MODEL
from tueri.input_scanners.ban_topics import MODEL_DEBERTA_BASE_V2 as BAN_TOPICS_MODEL
from tueri.input_scanners.base import Scanner as InputScanner
from tueri.input_scanners.language import DEFAULT_MODEL as LANGUAGE_MODEL
from tueri.input_scanners.prompt_injection import V2_MODEL as PROMPT_INJECTION_MODEL
from tueri.model import Model
from tueri.output_scanners.base import Scanner as OutputScanner
from tueri.output_scanners.bias import DEFAULT_MODEL as BIAS_MODEL
from tueri.output_scanners.bad_url import DEFAULT_MODEL as BAD_URL_MODEL
from tueri.output_scanners.no_refusal import DEFAULT_MODEL as NO_REFUSAL_MODEL
from tueri.output_scanners.relevance import MODEL_EN_BGE_SMALL as RELEVANCE_MODEL
from tueri.vault import Vault

from .config import ScannerConfig
from .scanner_cache import get_scanner_cache_manager
from .util import get_resource_utilization

torch.set_num_threads(1)

LOGGER = structlog.getLogger(__name__)

MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://root:example@localhost:27017/")
MONGO_DB, MONGO_COLLECTION = os.getenv("MONGO_DB", "ChatApp"), os.getenv("MONGO_COLLECTION", "TueriScanners")

# Suppress MongoDB heartbeat logs
logging.getLogger("pymongo.topology").setLevel(logging.WARNING)
logging.getLogger("pymongo.serverSelection").setLevel(logging.WARNING)

try:
    mongo_client = MongoClient(MONGODB_URL, serverSelectionTimeoutMS=5000, heartbeatFrequencyMS=60000)
    db = mongo_client[MONGO_DB]
    scanners_collection = db[MONGO_COLLECTION]
    mongo_client.admin.command("ping")
except Exception as e:
    LOGGER.error("Error connecting to MongoDB", error=str(e))
    raise

meter = metrics.get_meter_provider().get_meter(__name__)
scanners_valid_counter = meter.create_counter(
    name="scanners.valid",
    unit="1",
    description="measures the number of valid scanners",
)

def _fetch_scanners_from_mongo(scanner_type: str) -> List[ScannerConfig]:
    coll = scanners_collection.find({"type": scanner_type})
    scanners: List[ScannerConfig] = []
    for scanner in coll:
        scanners.append(ScannerConfig(
            type=scanner.get("id"), 
            params=scanner.get("params", {})))
    return scanners

def get_input_scanners(scanners: List[ScannerConfig], vault: Vault, runtime_params: Optional[Dict[str, Dict]] = None) -> List[InputScanner]:
    """Load input scanners from MongoDB"""
    input_scanners_config = _fetch_scanners_from_mongo("input")
    loaded_input_scanners: List[InputScanner] = []

    # use caching when runtime parameters are provided
    if runtime_params:
        scanner_cache = get_scanner_cache_manager()
        for scanner in input_scanners_config:
            scanner_params = scanner.params.copy() if scanner.params else {}
            if scanner.type in runtime_params:
                scanner_params.update(runtime_params[scanner.type])
                LOGGER.debug("overriding parameters", scanner=scanner.type, overrides=runtime_params[scanner.type])

            def scanner_factory(scanner_type: str, params: dict) -> InputScanner:
                return _get_input_scanner(scanner_type, params, vault=vault)

            cached_scanner = scanner_cache.get_input_scanner(
                scanner.type,
                scanner_params,
                scanner_factory
            )
            loaded_input_scanners.append(cached_scanner)
            
    else:
        for scanner in input_scanners_config:
            scanner_params = scanner.params.copy() if scanner.params else {}
            loaded_input_scanners.append(
                _get_input_scanner(
                    scanner.type,
                    scanner_params,
                    vault=vault,
                )
            )

    return loaded_input_scanners


def get_output_scanners(scanners: List[ScannerConfig], vault: Vault, runtime_params: Optional[Dict[str, Dict]] = None) -> List[OutputScanner]:
    """Load output scanners from MongoDB"""
    output_scanners_config = _fetch_scanners_from_mongo("output")
    loaded_output_scanners: List[OutputScanner] = []

    # use caching when runtime parameters are provided
    if runtime_params:
        scanner_cache = get_scanner_cache_manager()

        for scanner in output_scanners_config:
            scanner_params = scanner.params.copy() if scanner.params else {}
            if scanner.type in runtime_params:
                scanner_params.update(runtime_params[scanner.type])
                LOGGER.debug("overriding parameters", scanner=scanner.type, overrides=runtime_params[scanner.type])

            def scanner_factory(scanner_type: str, params: dict) -> OutputScanner:
                return _get_output_scanner(scanner_type, params, vault=vault)

            cached_scanner = scanner_cache.get_output_scanner(
                scanner.type,
                scanner_params,
                scanner_factory
            )
            loaded_output_scanners.append(cached_scanner)
            
    else:
        for scanner in output_scanners_config:
            scanner_params = scanner.params.copy() if scanner.params else {}
            loaded_output_scanners.append(
                _get_output_scanner(
                    scanner.type,
                    scanner_params,
                    vault=vault,
                )
            )

    return loaded_output_scanners


def _configure_model(model: Model, scanner_config: Optional[Dict]): 
    if scanner_config is None:
        scanner_config = {}

    if "model_path" in scanner_config and scanner_config["model_path"] is not None:
        model.path = scanner_config["model_path"]
        model.onnx_path = scanner_config["model_path"]
        model.onnx_subfolder = ""
        model.kwargs = {"local_files_only": True}
        scanner_config.pop("model_path")

    if "model_batch_size" in scanner_config:
        model.pipeline_kwargs["batch_size"] = scanner_config["model_batch_size"]
        scanner_config.pop("model_batch_size")

    if "model_max_length" in scanner_config and scanner_config["model_max_length"] > 0:
        model.pipeline_kwargs["max_length"] = scanner_config["model_max_length"]
        scanner_config.pop("model_max_length")

    if (
        "model_onnx_file_name" in scanner_config
        and scanner_config["model_onnx_file_name"] is not None
    ):
        model.onnx_filename = scanner_config["model_onnx_file_name"]
        scanner_config.pop("model_onnx_file_name")


def _get_input_scanner(
    scanner_name: str,
    scanner_config: Optional[Dict],
    *,
    vault: Vault,
):
    if scanner_config is None:
        scanner_config = {}

    if scanner_name == "Anonymize":
        scanner_config["vault"] = vault

    if scanner_name in [
        "Anonymize",
        "BanTopics",
        "Language",
        "PromptInjection",
    ]:
        scanner_config["use_onnx"] = True

    if scanner_name == "Anonymize":
        # RoBERTa-based anonymizer doesn't need model configuration
        pass


    if scanner_name == "BanTopics":
        _configure_model(BAN_TOPICS_MODEL, scanner_config)
        scanner_config["model"] = BAN_TOPICS_MODEL

    if scanner_name == "BanCompetitors":
        _configure_model(BAN_COMPETITORS_MODEL, scanner_config)
        scanner_config["model"] = BAN_COMPETITORS_MODEL

    if scanner_name == "Language":
        _configure_model(LANGUAGE_MODEL, scanner_config)
        scanner_config["model"] = LANGUAGE_MODEL

    if scanner_name == "PromptInjection":
        _configure_model(PROMPT_INJECTION_MODEL, scanner_config)
        scanner_config["model"] = PROMPT_INJECTION_MODEL

    return input_scanners.get_scanner_by_name(scanner_name, scanner_config)


def _get_output_scanner(
    scanner_name: str,
    scanner_config: Optional[Dict],
    *,
    vault: Vault,
):
    if scanner_config is None:
        scanner_config = {}

    if scanner_name == "Deanonymize":
        scanner_config["vault"] = vault

    if scanner_name in [
        "BanTopics",
        "Bias",
        "Language",
        "LanguageSame",
        "BadURL",
        "NoRefusal",
        "FactualConsistency",
        "Relevance",
        "Sensitive",
    ]:
        scanner_config["use_onnx"] = True


    if scanner_name == "BanCompetitors":
        _configure_model(BAN_COMPETITORS_MODEL, scanner_config)
        scanner_config["model"] = BAN_COMPETITORS_MODEL

    if scanner_name == "BanTopics" or scanner_name == "FactualConsistency":
        _configure_model(BAN_TOPICS_MODEL, scanner_config)
        scanner_config["model"] = BAN_TOPICS_MODEL

    if scanner_name == "Bias":
        _configure_model(BIAS_MODEL, scanner_config)
        scanner_config["model"] = BIAS_MODEL

    if scanner_name == "Language":
        _configure_model(LANGUAGE_MODEL, scanner_config)
        scanner_config["model"] = LANGUAGE_MODEL

    if scanner_name == "LanguageSame":
        _configure_model(LANGUAGE_MODEL, scanner_config)
        scanner_config["model"] = LANGUAGE_MODEL

    if scanner_name == "BadURL":
        _configure_model(BAD_URL_MODEL, scanner_config)
        scanner_config["model"] = BAD_URL_MODEL

    if scanner_name == "NoRefusal":
        _configure_model(NO_REFUSAL_MODEL, scanner_config)
        scanner_config["model"] = NO_REFUSAL_MODEL

    if scanner_name == "Relevance":
        _configure_model(RELEVANCE_MODEL, scanner_config)
        scanner_config["model"] = RELEVANCE_MODEL

    if scanner_name == "Sensitive":
        # RoBERTa-based sensitive scanner doesn't need model configuration
        pass

    return output_scanners.get_scanner_by_name(scanner_name, scanner_config)


class InputIsInvalid(Exception):
    def __init__(self, scanner_name: str, input: str, risk_score: float):
        self.scanner_name = scanner_name
        self.input = input
        self.risk_score = risk_score

    def __str__(self):
        return f"Input is invalid based on {self.scanner_name}: {self.input} (risk score: {self.risk_score})"


class OutputIsInvalid(Exception):
    def __init__(self, scanner_name: str, output: str, risk_score: float):
        self.scanner_name = scanner_name
        self.output = output
        self.risk_score = risk_score

    def __str__(self):
        return f"Output is invalid based on {self.scanner_name}: {self.output} (risk score: {self.risk_score})"


def scan_prompt(scanner: InputScanner, prompt: str) -> (str, float):
    start_time_scanner = time.time()
    sanitized_prompt, is_valid, risk_score = scanner.scan(prompt)
    elapsed_time_scanner = time.time() - start_time_scanner

    scanner_name = type(scanner).__name__
    LOGGER.debug(
        "Input scanner completed",
        scanner=scanner_name,
        is_valid=is_valid,
        elapsed_time_seconds=round(elapsed_time_scanner, 2),
    )

    scanners_valid_counter.add(1, {"source": "input", "valid": is_valid, "scanner": scanner_name})

    if not is_valid:
        raise InputIsInvalid(scanner_name, prompt, risk_score)

    return type(scanner).__name__, risk_score


async def ascan_prompt(scanner: InputScanner, prompt: str) -> (str, float):
    return await asyncio.to_thread(scan_prompt, scanner, prompt)


def scan_output(scanner: OutputScanner, prompt: str, output: str) -> (str, float):
    start_time_scanner = time.time()
    sanitized_output, is_valid, risk_score = scanner.scan(prompt, output)
    elapsed_time_scanner = time.time() - start_time_scanner

    scanner_name = type(scanner).__name__
    LOGGER.debug(
        "Output scanner completed",
        scanner=scanner_name,
        is_valid=is_valid,
        elapsed_time_seconds=round(elapsed_time_scanner, 6),
    )

    scanners_valid_counter.add(1, {"source": "output", "valid": is_valid, "scanner": scanner_name})

    if not is_valid:
        raise OutputIsInvalid(scanner_name, output, risk_score)

    return type(scanner).__name__, risk_score


async def ascan_output(scanner: OutputScanner, prompt: str, output: str) -> (str, float):
    return await asyncio.to_thread(scan_output, scanner, prompt, output)
