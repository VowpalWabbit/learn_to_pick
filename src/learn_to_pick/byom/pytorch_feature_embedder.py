from sentence_transformers import SentenceTransformer
import torch
from learn_to_pick import PickBestFeaturizer


class PyTorchFeatureEmbedder:
    def __init__(self, auto_embed=False, model=None, *args, **kwargs):
        if model is None:
            model = SentenceTransformer("all-MiniLM-L6-v2")

        self.model = model
        self.featurizer = PickBestFeaturizer(auto_embed=auto_embed)

    def encode(self, stuff):
        embeddings = self.model.encode(stuff, convert_to_tensor=True)
        normalized = torch.nn.functional.normalize(embeddings)
        return normalized

    def convert_features_to_text(self, features):
        def process_feature(feature):
            if isinstance(feature, dict):
                return " ".join(
                    [f"{k}_{process_feature(v)}" for k, v in feature.items()]
                )
            elif isinstance(feature, list):
                return " ".join([process_feature(elem) for elem in feature])
            else:
                return str(feature)

        return process_feature(features)

    def format(self, event):
        # TODO: handle dense
        context_featurized, actions_featurized, selected = self.featurizer.featurize(
            event
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
