from __future__ import annotations

import re
import requests
from tueri.model import Model
from tueri.transformers_helpers import get_tokenizer_and_model_for_classification, pipeline
from tueri.util import calculate_risk_score, extract_urls, get_logger

from .base import Scanner

LOGGER = get_logger()

DEFAULT_MODEL = Model(
    path="DunnBC22/codebert-base-Malicious_URLs",
    revision="1221284b2495a4182cdb521be9d755de56e66899",
    onnx_path="ProtectAI/codebert-base-Malicious_URLs-onnx",
    onnx_revision="7bc4fa926eeae5e752d0790cc42faa24eb32fa64",
    pipeline_kwargs={
        "top_k": None,
        "return_token_type_ids": False,
        "max_length": 128,
        "truncation": True,
    },
)

_malicious_labels = [
    "defacement",
    "phishing",
    "malware",
]


class BadURL(Scanner):
    """
    This scanner combines URL reachability checking and maliciousness detection.
    
    It first checks if URLs are reachable, and only runs maliciousness detection
    on URLs that are actually reachable. This optimizes performance by avoiding
    expensive ML inference on unreachable URLs.
    """

    def __init__(
        self,
        *,
        model: Model | None = None,
        threshold: float = 0.5,
        use_onnx: bool = False,
        success_status_codes: list[int] | None = None,
        timeout: int = 5,
    ) -> None:
        """
        Parameters:
            model: The model to use for malicious URL detection.
            threshold: The threshold used to determine if the URL is malicious.
            use_onnx: Whether to use the ONNX version of the model.
            success_status_codes: A list of status codes that are considered as successful.
            timeout: The timeout in seconds for the HTTP requests.
        """
        self._malicious_threshold = threshold
        self._timeout = timeout

        if success_status_codes is None:
            success_status_codes = [
                requests.codes.ok,
                requests.codes.created,
                requests.codes.accepted,
            ]
        self._success_status_codes = success_status_codes

        if model is None:
            model = DEFAULT_MODEL

        tf_tokenizer, tf_model = get_tokenizer_and_model_for_classification(
            model=model,
            use_onnx=use_onnx,
        )

        self._classifier = pipeline(
            task="text-classification",
            model=tf_model,
            tokenizer=tf_tokenizer,
            **model.pipeline_kwargs,
        )

    def is_reachable(self, url: str) -> bool:
        """
        Check if the URL is reachable.
        """
        try:
            response = requests.get(url, timeout=self._timeout)
            return response.status_code in self._success_status_codes
        except requests.RequestException:
            return False

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        if output.strip() == "":
            return output, True, -1.0

        urls = extract_urls(output)
        if len(urls) == 0:
            return output, True, -1.0

        LOGGER.debug("Found URLs in the output", len=len(urls))

        # Check reachability for all URLs in one pass
        reachability_results = [(url, self.is_reachable(url)) for url in urls]
        unreachable_urls = [url for url, is_reachable in reachability_results if not is_reachable]
        reachable_urls = [url for url, is_reachable in reachability_results if is_reachable]

        # If there are unreachable URLs, mask them and return failure
        if unreachable_urls:
            LOGGER.warning("Unreachable URLs detected", urls=unreachable_urls)
            # Use regex for efficient bulk replacement
            masked_output = output
            for url in unreachable_urls:
                masked_output = re.sub(re.escape(url), "[UNREACHABLE_URL]", masked_output)
            return masked_output, False, 1.0

        # Only check maliciousness for reachable URLs
        if not reachable_urls:
            LOGGER.debug("No reachable URLs to check for maliciousness")
            return output, True, -1.0

        LOGGER.debug("Checking reachable URLs for maliciousness", count=len(reachable_urls))

        # Check for malicious URLs among reachable ones - batch processing
        results = self._classifier(reachable_urls)
        
        # Process all results at once using list comprehensions
        malicious_data = [
            (url, max([item["score"] for item in result if item["label"] in _malicious_labels]))
            for url, result in zip(reachable_urls, results)
            if any(item["label"] in _malicious_labels for item in result)
            and max([item["score"] for item in result if item["label"] in _malicious_labels]) > self._malicious_threshold
        ]
        
        highest_malicious_score = max([score for _, score in malicious_data], default=0.0)

        # If there are malicious URLs, mask them and return failure
        if malicious_data:
            # Log all at once
            for url, score in malicious_data:
                LOGGER.warning("Detected malicious URL", url=url, score=score)
            
            # Use regex for efficient bulk replacement
            masked_output = output
            for url, _ in malicious_data:
                masked_output = re.sub(re.escape(url), "[MALICIOUS_URL]", masked_output)
            return masked_output, False, calculate_risk_score(highest_malicious_score, self._malicious_threshold)

        LOGGER.debug("All URLs are safe and reachable")
        return (
            output,
            True,
            calculate_risk_score(highest_malicious_score, self._malicious_threshold),
        )