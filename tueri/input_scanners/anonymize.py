from __future__ import annotations

import os
from typing import Final
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.models.auto.modeling_auto import AutoModelForTokenClassification
from transformers.pipelines import pipeline

from ..exception import TueriValidationError
from ..util import calculate_risk_score, get_logger
from ..vault import Vault
from .base import Scanner

LOGGER = get_logger()

ALL_SUPPORTED_LANGUAGES: Final[list[str]] = ["en"]


class Anonymize(Scanner):
    """
    Anonymize sensitive data in the text using RoBERTa-based de-identification model.

    Anonymizes detected entities with placeholders like [REDACTED_PERSON_1] and stores the real values in a Vault.
    Uses the obi/deid_roberta_i2b2 model for medical text de-identification.
    """

    def __init__(
        self,
        vault: Vault,
        *,
        preamble: str = "",
        threshold: float = 0.5,
        language: str = "en",
        use_onnx: bool = False,
    ) -> None:
        """
        Initialize an instance of the Anonymize class.

        Parameters:
            vault: A vault instance to store the anonymized data.
            preamble: Text to prepend to sanitized prompt. If not provided, defaults to an empty string.
            threshold: Acceptance threshold for PII detection. Default is 0.5.
            language: Language of the anonymize detect. Default is "en".
            use_onnx: Kept for API compatibility but not used. Default is False.
        """
        
        if language not in ALL_SUPPORTED_LANGUAGES:
            raise TueriValidationError(f"Language must be in the list of allowed: {ALL_SUPPORTED_LANGUAGES}")

        os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disables huggingface/tokenizers warning

        self._vault = vault
        self._preamble = preamble
        self._threshold = threshold
        self._language = language
        self._model = None
        
        self._load_model()

    def _load_model(self):
        """Load RoBERTa de-identification model and tokenizer."""
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
            
            LOGGER.info("RoBERTa model loaded successfully on CPU")
            
        except Exception as e:
            LOGGER.error(f"Failed to load RoBERTa model: {str(e)}")
            raise TueriValidationError(f"Failed to load de-identification model: {str(e)}")

    def _process_text(self, text: str) -> list[dict]:
        """Process text through RoBERTa de-identification model."""
        if not self._model:
            raise TueriValidationError("Model not initialized")
            
        try:
            results = self._model(text)
            return results
        except Exception as e:
            LOGGER.error(f"Error processing text with RoBERTa model: {str(e)}")
            return []

    def _create_output_and_vault_entries(self, text: str, results: list[dict]) -> tuple[str, list[tuple[str, str]]]:
        """
        Create output with PII removed and generate vault entries.
        
        Returns tuple: (output, vault_entries)
        """
        if not results:
            return text, []
        
        filtered_results = [r for r in results if r['score'] >= self._threshold]
        if not filtered_results:
            return text, []
        
        sorted_entities = sorted(filtered_results, key=lambda x: x['start'], reverse=True) # Sort by start position in reverse order
        output = text
        entries = []
        entity_counters = {}
        
        for entity in sorted_entities:
            start, end = entity['start'], entity['end']
            entity_type = entity['entity_group'].upper()
            
            original_text = entity['word']
            if original_text.startswith('##'):
                continue
            original_text = text[start:end]
            
            if entity_type not in entity_counters:
                entity_counters[entity_type] = {}
            
            if original_text not in entity_counters[entity_type]:
                # Check if this entity value already exists in the vault
                existing_placeholders = [placeholder for placeholder, value in self._vault.get() if value == original_text and entity_type in placeholder]
                
                if existing_placeholders:
                    placeholder = existing_placeholders[0]
                else:
                    existing_count = len([placeholder for placeholder, _ in self._vault.get() if entity_type in placeholder])
                    new_count = len([k for k in entity_counters[entity_type].keys()])
                    index = existing_count + new_count + 1
                    placeholder = f"[{entity_type}_{index}]"
                entity_counters[entity_type][original_text] = placeholder
            else:
                placeholder = entity_counters[entity_type][original_text]
            
            # Replace in text
            output = output[:start] + placeholder + output[end:]
            
            # Add to vault entries if not already exists
            if not any(entry[0] == placeholder for entry in entries):
                entries.append((placeholder, original_text))
        
        return output, entries

    # @staticmethod
    # def remove_single_quotes(text: str) -> str:
    #     """Remove single quotes from text (used by other scanners for compatibility)."""
    #     return text.replace("'", " ")

    def scan(self, prompt: str) -> tuple[str, bool, float]:
        """
        Scan and anonymize the prompt using RoBERTa de-identification model.
        
        Returns tuple: (sanitized_prompt, is_valid, risk_score)
        """
        risk_score = -1.0
        
        if prompt.strip() == "":
            return prompt, True, risk_score
        
        results = self._process_text(prompt)
        if not results:
            LOGGER.debug("No PII detected in prompt", risk_score=0.0)
            return prompt, True, -1.0

        filtered_results = [r for r in results if r['score'] >= self._threshold]
        if not filtered_results:
            LOGGER.debug("No PII detected above threshold", threshold=self._threshold)
            return prompt, True, -1.0

        # Calculate risk score as the maximum confidence score
        risk_score = round(max(result['score'] for result in filtered_results), 2)
        
        sanitized_prompt, vault_entries = self._create_output_and_vault_entries(prompt, results)
        if prompt != sanitized_prompt:
            LOGGER.warning(
                "Replaced sensitive data in prompt",
                detected_entities=len(filtered_results),
                risk_score=risk_score,
            )
            for placeholder, original_value in vault_entries:
                if not self._vault.placeholder_exists(placeholder):
                    self._vault.append((placeholder, original_value))
            return (
                self._preamble + sanitized_prompt,
                True,
                calculate_risk_score(risk_score, self._threshold),
            )

        LOGGER.debug("No sensitive data to replace", risk_score=risk_score)
        return prompt, True, -1.0