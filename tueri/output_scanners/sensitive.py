from __future__ import annotations

import os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

from tueri.exception import TueriValidationError
from tueri.input_scanners.anonymize import ALL_SUPPORTED_LANGUAGES
from tueri.util import calculate_risk_score, get_logger

from .base import Scanner

LOGGER = get_logger()


class Sensitive(Scanner):
    """
    A class used to detect sensitive (PII) data in the output of a language model using RoBERTa.
    """

    def __init__(
        self,
        *,
        redact: bool = False,
        threshold: float = 0.5,
        language: str = "en",
        use_onnx: bool = False,
    ) -> None:
        """
        Initializes an instance of the Sensitive class.

        Parameters:
           redact (bool): Redact found sensitive entities. Default to False.
           threshold (float): Acceptance threshold for PII detection. Default is 0.5.
           language (str): Language of the text. Default is "en".
           use_onnx (bool): Kept for API compatibility but not used. Default is False.
        """
        if language not in ALL_SUPPORTED_LANGUAGES:
            raise TueriValidationError(f"Language must be in the list of allowed: {ALL_SUPPORTED_LANGUAGES}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables huggingface/tokenizers warning

        self._redact = redact
        self._threshold = threshold
        self._language = language
        self._model = None
        
        # Initialize the model
        self._load_model()

    def _load_model(self):
        """Load the RoBERTa de-identification model and tokenizer."""
        model_name = "obi/deid_roberta_i2b2"
        
        LOGGER.info("Loading RoBERTa de-identification model...")
        
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForTokenClassification.from_pretrained(model_name)
            
            self._model = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
                device=-1  # Force CPU usage
            )
            
            LOGGER.info("RoBERTa model loaded successfully")
            
        except Exception as e:
            LOGGER.error(f"Failed to load RoBERTa model: {str(e)}")
            raise TueriValidationError(f"Failed to load de-identification model: {str(e)}")

    def _process_text(self, text: str) -> list[dict]:
        """Process text through the RoBERTa de-identification model."""
        if not self._model:
            raise TueriValidationError("Model not initialized")
            
        try:
            results = self._model(text)
            return results
        except Exception as e:
            LOGGER.error(f"Error processing text with RoBERTa model: {str(e)}")
            return []

    def _create_redacted_text(self, text: str, results: list[dict]) -> str:
        """
        Create a version of the text with PII redacted if redaction is enabled.
        """
        if not results or not self._redact:
            return text

        filtered_results = [r for r in results if r['score'] >= self._threshold]
        if not filtered_results:
            return text

        sorted_entities = sorted(filtered_results, key=lambda x: x['start'], reverse=True)  # Sort by start position in reverse order to avoid index shifting
        redacted_text = text
        
        for entity in sorted_entities:
            start, end = entity['start'], entity['end']
            entity_type = entity['entity_group'].upper()
            
            original_text = entity['word']
            if original_text.startswith('##'):
                continue
            placeholder = f"[REDACTED_{entity_type}]"
            redacted_text = redacted_text[:start] + placeholder + redacted_text[end:]
        
        return redacted_text

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        """
        Scan the output for sensitive information using RoBERTa model.
        
        Parameters:
            prompt: The original input prompt (not used in output scanning but required by interface).
            output: The LLM output to scan for PII.
            
        Returns:
            tuple: (processed_output, is_valid, risk_score)
        """
        if output.strip() == "":
            return output, True, -1.0

        results = self._process_text(output)
        if not results:
            LOGGER.debug("No PII detected in output")
            return output, True, -1.0

        # Filter by threshold and calculate risk score
        filtered_results = [r for r in results if r['score'] >= self._threshold]
        if not filtered_results:
            LOGGER.debug("No PII detected above threshold in output", threshold=self._threshold)
            return output, True, -1.0

        # Calculate risk score as the maximum confidence score
        risk_score = round(max(result['score'] for result in filtered_results), 2)
        
        processed_output = self._create_redacted_text(output, results)
        LOGGER.warning(
            "Found sensitive data in the output",
            detected_entities=len(filtered_results),
            risk_score=risk_score,
            redacted=self._redact
        )
        return processed_output, False, calculate_risk_score(risk_score, self._threshold)