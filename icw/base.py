from abc import ABC, abstractmethod


class BaseICW(ABC):
    def __init__(self, algorithm_config: dict) -> None:
        pass

    @abstractmethod
    def generate_watermarked_text(self, prompt: str) -> str:
        pass

    @abstractmethod
    def detect_watermark(self, text: str) -> float:
        pass
