"""
渐进披露控制器（独立模块）

本模块从 intent_classifier.py 独立出来，方便单独引用。
完整的 IntentClassifier + ProgressiveDisclosure 逻辑请参见 intent_classifier.py

本文件主要作为便捷重导出。
"""

from su_memory._sys.intent_classifier import (
    IntentClassifier,
    IntentConfig,
    ProgressiveDisclosure,
    DisclosureStage,
    DISCLOSURE_STAGES,
    DEFAULT_INTENTS,
)

__all__ = [
    "IntentClassifier",
    "IntentConfig",
    "ProgressiveDisclosure",
    "DisclosureStage",
    "DISCLOSURE_STAGES",
    "DEFAULT_INTENTS",
]
