from learn_to_pick import base, PickBestEvent
from learn_to_pick.pytorch.logistic_regression import ResidualLogisticRegressor
from learn_to_pick.pytorch.igw import SamplingIGW
from learn_to_pick.pytorch.feature_embedder import PyTorchFeatureEmbedder
import torch
import os
from typing import Any, Optional, PathLike, TypeVar, Union

TEvent = TypeVar("TEvent", bound=base.Event)


class PyTorchPolicy(base.Policy[PickBestEvent]):
    def __init__(
        self,
        feature_embedder=PyTorchFeatureEmbedder(),
        depth: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        *args: Any,
        **kwargs: Any,
    ):
        print(f"Device: {device}")
        super().__init__(*args, **kwargs)
        self.workspace = ResidualLogisticRegressor(
            feature_embedder.model.get_sentence_embedding_dimension() * 2, depth, device
        ).to(device)
        self.feature_embedder = feature_embedder
        self.device = device
        self.index = 0
        self.loss = None

    def predict(self, event: TEvent) -> list:
        X, A = self.feature_embedder.format(event)
        # TODO IGW sampling then create the distro so that the one
        # that was sampled here is the one that will def be sampled by
        # the base sampler, and in the future replace the sampler so that it
        # is something that can be plugged in
        p = self.workspace.predict(X, A)
        import math

        explore = SamplingIGW(A, p, math.sqrt(self.index))
        self.index += 1
        r = []
        for index in range(p.shape[1]):
            if index == explore[0]:
                r.append((index, 1))
            else:
                r.append((index, 0))
        return r

    def learn(self, event: TEvent) -> None:
        R, X, A = self.feature_embedder.format(event)
        R, X, A = R.to(self.device), X.to(self.device), A.to(self.device)
        self.loss = self.workspace.bandit_learn(X, A, R)

    def log(self, event):
        pass

    def save(self, path: Optional[Union[str, PathLike]]) -> None:
        state = {
            "workspace_state_dict": self.workspace.state_dict(),
            "optimizer_state_dict": self.workspace.optim.state_dict(),
            "device": self.device,
            "index": self.index,
            "loss": self.loss,
        }
        print(f"Saving model to {path}")
        dir, _ = os.path.split(path)
        if dir and not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        torch.save(state, path)

    def load(self, path: Optional[Union[str, PathLike]]) -> None:
        import parameterfree

        if os.path.exists(path):
            print(f"Loading model from {path}")
            checkpoint = torch.load(path, map_location=self.device)

            self.workspace.load_state_dict(checkpoint["workspace_state_dict"])
            self.workspace.optim = parameterfree.COCOB(self.workspace.parameters())
            self.workspace.optim.load_state_dict(checkpoint["optimizer_state_dict"])
            self.device = checkpoint["device"]
            self.workspace.to(self.device)
            self.index = checkpoint["index"]
            self.loss = checkpoint["loss"]
