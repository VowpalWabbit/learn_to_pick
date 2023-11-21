from learn_to_pick import base, PickBestEvent
from learn_to_pick.pytorch.logistic_regression import ResidualLogisticRegressor
from learn_to_pick.pytorch.igw import SamplingIGW
import torch
import os


class PyTorchPolicy(base.Policy[PickBestEvent]):
    def __init__(
        self,
        feature_embedder,
        depth: int = 2,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        *args,
        **kwargs,
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

    def predict(self, event):
        X, A = self.feature_embedder.format(event)
        # print(f"X shape: {X.shape}")
        # print(f"A shape: {A.shape}")
        # TODO IGW sampling then create the distro so that the one
        # that was sampled here is the one that will def be sampled by
        # the base sampler, and in the future replace the sampler so that it
        # is something that can be plugged in
        p = self.workspace.predict(X, A)
        # print(f"p: {p}")
        import math

        explore = SamplingIGW(A, p, math.sqrt(self.index))
        self.index += 1
        # print(f"explore: {explore}")
        r = []
        for index in range(p.shape[1]):
            if index == explore[0]:
                r.append((index, 1))
            else:
                r.append((index, 0))
        # print(f"returning: {r}")
        return r

    def learn(self, event):
        R, X, A = self.feature_embedder.format(event)
        # print(f"R: {R}")
        R, X, A = R.to(self.device), X.to(self.device), A.to(self.device)
        self.loss = self.workspace.bandit_learn(X, A, R)

    def log(self, event):
        pass

    def save(self, path) -> None:
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

    def load(self, path) -> None:
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
