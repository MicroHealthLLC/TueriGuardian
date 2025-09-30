from __future__ import annotations

import hashlib
import importlib
import json
from functools import lru_cache
from typing import Literal, get_args

from .exception import TueriValidationError
from .model import Model
from .util import device, get_logger, lazy_load_dep

LOGGER = get_logger()


def _create_model_cache_key(model: Model, use_onnx: bool = False) -> str:
    """
    Create a cache key for a model based on its configuration.

    Args:
        model (Model): The model to create a cache key for.
        use_onnx (bool): Whether ONNX is being used.

    Returns:
        str: A unique cache key for the model configuration.
    """
    # Convert any object to a JSON-serializable format
    def make_serializable(obj):
        if obj is None:
            return None
        elif isinstance(obj, (str, int, float, bool)):
            return obj
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        else:
            # Convert any complex object to string representation
            return str(obj)

    model_config = {
        "path": model.path,
        "subfolder": model.subfolder,
        "revision": model.revision,
        "onnx_path": model.onnx_path if use_onnx else None,
        "onnx_revision": model.onnx_revision if use_onnx else None,
        "onnx_subfolder": model.onnx_subfolder if use_onnx else None,
        "onnx_filename": model.onnx_filename if use_onnx else None,
        "kwargs": make_serializable(model.kwargs),
        "pipeline_kwargs": make_serializable(model.pipeline_kwargs),
        "tokenizer_kwargs": make_serializable(model.tokenizer_kwargs),
        "use_onnx": use_onnx,
        "device": str(device()),
    }

    # Create a deterministic hash of the configuration
    config_str = json.dumps(model_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


# Cache for tokenizers
_tokenizer_cache = {}

def get_tokenizer(model: Model):
    """
    This function loads a tokenizer given a model identifier and caches it.
    Subsequent calls with the same model_identifier will return the cached tokenizer.

    Args:
        model (Model): The model to load the tokenizer for.
    """
    cache_key = _create_model_cache_key(model, use_onnx=False)

    if cache_key in _tokenizer_cache:
        LOGGER.debug("Using cached tokenizer", model=model)
        return _tokenizer_cache[cache_key]

    transformers = lazy_load_dep("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model.path, revision=model.revision, **model.tokenizer_kwargs
    )

    # Cache the tokenizer (limit cache size to prevent memory issues)
    if len(_tokenizer_cache) >= 32:
        # Remove oldest entry (simple FIFO eviction)
        oldest_key = next(iter(_tokenizer_cache))
        del _tokenizer_cache[oldest_key]

    _tokenizer_cache[cache_key] = tokenizer
    LOGGER.debug("Cached new tokenizer", model=model)
    return tokenizer


@lru_cache(maxsize=None)  # Unbounded cache
def is_onnx_supported() -> bool:
    is_supported = importlib.util.find_spec("optimum.onnxruntime") is not None  # type: ignore
    if not is_supported:
        LOGGER.warning(
            "ONNX Runtime is not available. "
            "Please install optimum: "
            "`pip install llm-guard[onnxruntime]` for CPU or "
            "`pip install llm-guard[onnxruntime-gpu]` for GPU to enable ONNX Runtime optimizations."
        )

    return is_supported


# Cache for ONNX models
_onnx_model_cache = {}

def _ort_model_for_sequence_classification(
    model: Model,
):
    cache_key = _create_model_cache_key(model, use_onnx=True)

    if cache_key in _onnx_model_cache:
        LOGGER.debug("Using cached ONNX classification model", model=model)
        return _onnx_model_cache[cache_key]

    provider = "CPUExecutionProvider"
    package_name = "optimum[onnxruntime]"
    if device().type == "cuda":
        package_name = "optimum[onnxruntime-gpu]"
        provider = "CUDAExecutionProvider"

    onnxruntime = lazy_load_dep("optimum.onnxruntime", package_name)

    tf_model = onnxruntime.ORTModelForSequenceClassification.from_pretrained(
        model.onnx_path or model.path,
        export=model.onnx_path is None,
        file_name=model.onnx_filename,
        subfolder=model.onnx_subfolder,
        revision=model.onnx_revision,
        provider=provider,
        **model.kwargs,
    )

    # Cache the model (limit cache size to prevent memory issues)
    if len(_onnx_model_cache) >= 16:
        # Remove oldest entry (simple FIFO eviction)
        oldest_key = next(iter(_onnx_model_cache))
        del _onnx_model_cache[oldest_key]

    _onnx_model_cache[cache_key] = tf_model
    LOGGER.debug("Cached new ONNX classification model", model=model, device=device())

    return tf_model


# Cache for PyTorch classification models
_pytorch_classification_cache = {}

def get_tokenizer_and_model_for_classification(
    model: Model,
    use_onnx: bool = False,
):
    """
    This function loads a tokenizer and model given a model identifier and caches them.
    Subsequent calls with the same model_identifier will return the cached tokenizer.

    Args:
        model (str): The model identifier to load the tokenizer and model for.
        use_onnx (bool): Whether to use the ONNX version of the model. Defaults to False.
    """
    tf_tokenizer = get_tokenizer(model)

    if use_onnx and is_onnx_supported() is False:
        LOGGER.warning("ONNX is not supported on this machine. Using PyTorch instead of ONNX.")
        use_onnx = False

    if use_onnx is False:
        cache_key = _create_model_cache_key(model, use_onnx=False)

        if cache_key in _pytorch_classification_cache:
            LOGGER.debug("Using cached PyTorch classification model", model=model)
            tf_model = _pytorch_classification_cache[cache_key]
        else:
            transformers = lazy_load_dep("transformers")
            tf_model = transformers.AutoModelForSequenceClassification.from_pretrained(
                model.path,
                subfolder=model.subfolder,
                revision=model.revision,
                torch_dtype="auto",
                low_cpu_mem_usage=False,
                **model.kwargs,
            )
            # Handle meta device properly - use to_empty() when moving from meta device
            if hasattr(tf_model, 'device') and str(tf_model.device) == 'meta':
                tf_model = tf_model.to_empty(device=device())
            else:
                tf_model = tf_model.to(device())

            # Cache the model (limit cache size to prevent memory issues)
            if len(_pytorch_classification_cache) >= 16:
                # Remove oldest entry (simple FIFO eviction)
                oldest_key = next(iter(_pytorch_classification_cache))
                del _pytorch_classification_cache[oldest_key]

            _pytorch_classification_cache[cache_key] = tf_model
            LOGGER.debug("Cached new PyTorch classification model", model=model, device=device())

        return tf_tokenizer, tf_model

    tf_model = _ort_model_for_sequence_classification(model)

    return tf_tokenizer, tf_model


# Cache for PyTorch NER models
_pytorch_ner_cache = {}
# Cache for ONNX NER models
_onnx_ner_cache = {}

def get_tokenizer_and_model_for_ner(
    model: Model,
    use_onnx: bool = False,
):
    """
    This function loads a tokenizer and model given a model identifier and caches them.
    Subsequent calls with the same model_identifier will return the cached tokenizer.

    Args:
        model (str): The model identifier to load the tokenizer and model for.
        use_onnx (bool): Whether to use the ONNX version of the model. Defaults to False.
    """
    tf_tokenizer = get_tokenizer(model)

    if use_onnx and is_onnx_supported() is False:
        LOGGER.warning("ONNX is not supported on this machine. Using PyTorch instead of ONNX.")
        use_onnx = False

    if use_onnx is False:
        cache_key = _create_model_cache_key(model, use_onnx=False)

        if cache_key in _pytorch_ner_cache:
            LOGGER.debug("Using cached PyTorch NER model", model=model)
            tf_model = _pytorch_ner_cache[cache_key]
        else:
            transformers = lazy_load_dep("transformers")
            tf_model = transformers.AutoModelForTokenClassification.from_pretrained(
                model.path,
                subfolder=model.subfolder,
                revision=model.revision,
                torch_dtype="auto",
                low_cpu_mem_usage=False,
                **model.kwargs,
            )
            # Handle meta device properly - use to_empty() when moving from meta device
            if hasattr(tf_model, 'device') and str(tf_model.device) == 'meta':
                tf_model = tf_model.to_empty(device=device())
            else:
                tf_model = tf_model.to(device())

            # Cache the model (limit cache size to prevent memory issues)
            if len(_pytorch_ner_cache) >= 16:
                # Remove oldest entry (simple FIFO eviction)
                oldest_key = next(iter(_pytorch_ner_cache))
                del _pytorch_ner_cache[oldest_key]

            _pytorch_ner_cache[cache_key] = tf_model
            LOGGER.debug("Cached new PyTorch NER model", model=model, device=device())

        return tf_tokenizer, tf_model

    cache_key = _create_model_cache_key(model, use_onnx=True)

    if cache_key in _onnx_ner_cache:
        LOGGER.debug("Using cached ONNX NER model", model=model)
        tf_model = _onnx_ner_cache[cache_key]
    else:
        optimum_onnxruntime = lazy_load_dep(
            "optimum.onnxruntime",
            ("optimum[onnxruntime]" if device().type != "cuda" else "optimum[onnxruntime-gpu]"),
        )

        tf_model = optimum_onnxruntime.ORTModelForTokenClassification.from_pretrained(
            model.onnx_path,
            export=False,
            subfolder=model.onnx_subfolder,
            provider=("CUDAExecutionProvider" if device().type == "cuda" else "CPUExecutionProvider"),
            revision=model.onnx_revision,
            file_name=model.onnx_filename,
            **model.kwargs,
        )

        # Cache the model (limit cache size to prevent memory issues)
        if len(_onnx_ner_cache) >= 16:
            # Remove oldest entry (simple FIFO eviction)
            oldest_key = next(iter(_onnx_ner_cache))
            del _onnx_ner_cache[oldest_key]

        _onnx_ner_cache[cache_key] = tf_model
        LOGGER.debug("Cached new ONNX NER model", model=model, device=device())

    return tf_tokenizer, tf_model


ClassificationTask = Literal["text-classification", "zero-shot-classification"]


def pipeline(
    task: str,
    model,
    tokenizer,
    **kwargs,
):
    if task not in get_args(ClassificationTask):
        raise TueriValidationError(f"Invalid task. Must be one of {ClassificationTask}")

    if kwargs.get("max_length", None) is None:
        kwargs["max_length"] = tokenizer.model_max_length

    transformers = lazy_load_dep("transformers")
    return transformers.pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        **kwargs,
    )
