from typing import Dict, Optional

from .bad_url import BadURL
from .ban_competitors import BanCompetitors
from .ban_substrings import BanSubstrings
from .ban_topics import BanTopics
from .base import Scanner
from .bias import Bias
from .deanonymize import Deanonymize
from .factual_consistency import FactualConsistency
from .json import JSON
from .language import Language
from .language_same import LanguageSame
from .mask_code import MaskCode
from .no_refusal import NoRefusal, NoRefusalLight
from .regex import Regex
from .relevance import Relevance
from .sensitive import Sensitive
from .sentiment import Sentiment


def get_scanner_by_name(scanner_name: str, scanner_config: Optional[Dict] = None) -> Scanner:
    """
    Get scanner by name.

    Parameters:
        scanner_name (str): Name of scanner.
        scanner_config (Optional[Dict], optional): Scanner configuration. Defaults to None.

    Raises:
        ValueError: If scanner name is unknown.
    """
    if scanner_config is None:
        scanner_config = {}

    if scanner_name == "BadURL":
        return BadURL(**scanner_config)
    
    if scanner_name == "MaskCode":
        return MaskCode(**scanner_config)

    if scanner_name == "BanCompetitors":
        return BanCompetitors(**scanner_config)

    if scanner_name == "BanSubstrings":
        return BanSubstrings(**scanner_config)

    if scanner_name == "BanTopics":
        return BanTopics(**scanner_config)

    if scanner_name == "Bias":
        return Bias(**scanner_config)

    if scanner_name == "Deanonymize":
        return Deanonymize(**scanner_config)

    if scanner_name == "FactualConsistency":
        return FactualConsistency(**scanner_config)

    if scanner_name == "JSON":
        return JSON(**scanner_config)

    if scanner_name == "Language":
        return Language(**scanner_config)

    if scanner_name == "LanguageSame":
        return LanguageSame(**scanner_config)

    if scanner_name == "NoRefusal":
        return NoRefusal(**scanner_config)

    if scanner_name == "NoRefusalLight":
        return NoRefusalLight()

    if scanner_name == "Regex":
        return Regex(**scanner_config)

    if scanner_name == "Relevance":
        return Relevance(**scanner_config)

    if scanner_name == "Sensitive":
        return Sensitive(**scanner_config)

    if scanner_name == "Sentiment":
        return Sentiment(**scanner_config)
    
    raise ValueError(f"Unknown scanner name: {scanner_name}!")
