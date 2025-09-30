from __future__ import annotations

from tueri.input_scanners.mask_code import MaskCode as InputMaskCode
from tueri.model import Model
from typing import Union

from .base import Scanner


class MaskCode(Scanner):
    """
    A scanner that detects code snippets in the model output and masks them.
    """

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        mask_token: str = "[CODE]",
        language_model: Union[Model, None] = None,
        use_onnx: bool = False,
        language_threshold: float = 0.5,
        **kwargs
    ) -> None:
        """
        Initialize a new MaskCode scanner.

        Parameters:
            threshold: The probability threshold for code detection. Default is 0.5.
            mask_token: Token to replace detected code with. Default is "[CODE]".
            language_model: The model to use for language identification.
            use_onnx: Whether to use ONNX for language identification inference. Default is False.
            language_threshold: The threshold for language identification confidence. Default is 0.5.
            **kwargs: Additional parameters passed to the underlying scanner.
        """

        self._scanner = InputMaskCode(
            threshold=threshold,
            mask_token=mask_token,
            language_model=language_model,
            use_onnx=use_onnx,
            language_threshold=language_threshold,
            **kwargs
        )

    def scan(self, prompt: str, output: str) -> tuple[str, bool, float]:
        return self._scanner.scan(output)