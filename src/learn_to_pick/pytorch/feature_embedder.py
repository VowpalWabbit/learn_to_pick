from sentence_transformers import SentenceTransformer
import torch
from torch import Tensor

from learn_to_pick import PickBestFeaturizer
from learn_to_pick.base import Event
from learn_to_pick.features import SparseFeatures
from typing import Any, Tuple, TypeVar, Union

TEvent = TypeVar("TEvent", bound=Event)


class PyTorchFeatureEmbedder:
    def __init__(self, model: Any = None):
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        self.model = model
        self.featurizer = PickBestFeaturizer(auto_embed=False)

    def encode(self, to_encode: str) -> Tensor:
        embeddings = self.model.encode(to_encode, convert_to_tensor=True)
        normalized = torch.nn.functional.normalize(embeddings)
        return normalized

    def convert_features_to_text(self, sparse_features: SparseFeatures) -> str:
        results = []
        for ns, obj in sparse_features.items():
            value = obj.get("default_ft", "")
            results.append(f"{ns}={value}")
        return " ".join(results)

    def format(
        self, event: TEvent
    ) -> Union[Tuple[Tensor, Tensor, Tensor], Tuple[Tensor, Tensor]]:
        context_featurized, actions_featurized, selected = self.featurizer.featurize(
            event
        )

        if len(context_featurized.dense) > 0:
            raise NotImplementedError(
                "pytorch policy doesn't support context with dense features"
            )

        for action_featurized in actions_featurized:
            if len(action_featurized.dense) > 0:
                raise NotImplementedError(
                    "pytorch policy doesn't support action with dense features"
                )

        context_sparse = self.encode(
            [self.convert_features_to_text(context_featurized.sparse)]
        )

        actions_sparse = []
        for action_featurized in actions_featurized:
            actions_sparse.append(
                self.convert_features_to_text(action_featurized.sparse)
            )
        actions_sparse = self.encode(actions_sparse).unsqueeze(0)

        if selected.score is not None:
            return (
                torch.Tensor([[selected.score]]),
                context_sparse,
                actions_sparse[:, selected.index, :].unsqueeze(1),
            )
        else:
            return context_sparse, actions_sparse
