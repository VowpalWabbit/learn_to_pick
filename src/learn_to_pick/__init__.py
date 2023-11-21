import logging

from learn_to_pick.base import (
    AutoSelectionScorer,
    BasedOn,
    Embed,
    Featurizer,
    Policy,
    SelectionScorer,
    ToSelectFrom,
    embed,
)
from learn_to_pick.pick_best import (
    PickBest,
    PickBestEvent,
    PickBestFeaturizer,
    PickBestRandomPolicy,
    PickBestSelected,
)


from learn_to_pick.vw.policy import VwPolicy
from learn_to_pick.vw.model_repository import ModelRepository
from learn_to_pick.vw.logger import VwLogger

from learn_to_pick.pytorch.policy import PyTorchPolicy
from learn_to_pick.pytorch.pytorch_feature_embedder import PyTorchFeatureEmbedder


def configure_logger() -> None:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)


configure_logger()

__all__ = [
    "PickBest",
    "PickBestEvent",
    "PickBestSelected",
    "PickBestFeaturizer",
    "PickBestRandomPolicy",
    "Embed",
    "BasedOn",
    "ToSelectFrom",
    "SelectionScorer",
    "AutoSelectionScorer",
    "Featurizer",
    "Policy",
    "PyTorchPolicy",
    "PyTorchFeatureEmbedder",
    "embed",
    "ModelRepository",
    "VwPolicy",
    "VwLogger"
]
