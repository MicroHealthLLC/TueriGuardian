# Tueri - The Security Toolkit for LLM Interactions

Tueri is a comprehensive tool designed to fortify the security of Large Language Models (LLMs).

## What is Tueri?

By offering sanitization, detection of harmful language, prevention of data leakage, and resistance against prompt injection attacks, Tueri ensures that your interactions with LLMs remain safe and secure.

## Getting Started

**Important Notes**:

- Tueri is designed for easy integration and deployment in production environments. While it's ready to use out-of-the-box, please be informed that we're constantly improving and updating the repository.
- Base functionality requires a limited number of libraries. As you explore more advanced features, necessary libraries will be automatically installed.
- Ensure you're using Python version 3.9 or higher. Confirm with: `python --version`.
- Library installation issues? Consider upgrading pip: `python -m pip install --upgrade pip`.

**Examples**:

- Deploy Tueri as [API](./docs/api/overview.md)

## Supported scanners

### Prompt scanners

- [Anonymize](./docs/input_scanners/anonymize.md)
- [BanCompetitors](./docs/input_scanners/ban_competitors.md)
- [BanSubstrings](./docs/input_scanners/ban_substrings.md)
- [BanTopics](./docs/input_scanners/ban_topics.md)
- [InvisibleText](./docs/input_scanners/invisible_text.md)
- [Language](./docs/input_scanners/language.md)
- [MaskCode](./docs/input_scanners/mask_code.md)
- [PromptInjection](./docs/input_scanners/prompt_injection.md)
- [Regex](./docs/input_scanners/regex.md)
- [Secrets](./docs/input_scanners/secrets.md)
- [Sentiment](./docs/input_scanners/sentiment.md)
- [TokenLimit](./docs/input_scanners/token_limit.md)

### Output scanners

- [BadURL](./docs/output_scanners/bad_url.md)
- [BanCompetitors](./docs/output_scanners/ban_competitors.md)
- [BanSubstrings](./docs/output_scanners/ban_substrings.md)
- [BanTopics](./docs/output_scanners/ban_topics.md)
- [Bias](./docs/output_scanners/bias.md)
- [Deanonymize](./docs/output_scanners/deanonymize.md)
- [FactualConsistency](./docs/output_scanners/factual_consistency.md)
- [JSON](./docs/output_scanners/json.md)
- [Language](./docs/output_scanners/language.md)
- [LanguageSame](./docs/output_scanners/language_same.md)
- [MaskCode](./docs/input_scanners/mask_code.md)
- [NoRefusal](./docs/output_scanners/no_refusal.md)
- [Regex](./docs/output_scanners/regex.md)
- [Relevance](./docs/output_scanners/relevance.md)
- [Sensitive](./docs/output_scanners/sensitive.md)
- [Sentiment](./docs/output_scanners/sentiment.md)
