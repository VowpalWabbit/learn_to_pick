from learn_to_pick import base, PickBestEvent
from learn_to_pick.byom.logistic_regression import ResidualLogisticRegressor
from learn_to_pick.byom.igw import SamplingIGW

class PyTorchPolicy(base.Policy[PickBestEvent]):
    def __init__(
        self,
        feature_embedder,
        depth: int = 2,
        device: str = 'cuda',
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.workspace = ResidualLogisticRegressor(feature_embedder.model.get_sentence_embedding_dimension() * 2, depth).to(device)
        self.feature_embedder = feature_embedder
        self.device = device
        self.index = 0

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
        return [(index, val) for index, val in enumerate(p[0].tolist())]

    def learn(self, event):
        R, X, A = self.feature_embedder.format(event)
        # print(f"R: {R}")
        R, X, A = R.to(self.device), X.to(self.device), A.to(self.device)
        self.workspace.bandit_learn(X, A, R)

    def log(self, event):
        pass

    def save(self) -> None:
        pass