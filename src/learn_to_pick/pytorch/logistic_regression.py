import parameterfree
import torch
from torch import Tensor
import torch.nn.functional as F


class MLP(torch.nn.Module):
    @staticmethod
    def new_gelu(x: Tensor) -> Tensor:
        import math

        return (
            0.5
            * x
            * (
                1.0
                + torch.tanh(
                    math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))
                )
            )
        )

    def __init__(self, dim: int):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, 4 * dim)
        self.c_proj = torch.nn.Linear(4 * dim, dim)
        self.dropout = torch.nn.Dropout(0.5)

    def forward(self, x: Tensor) -> Tensor:
        x = self.c_fc(x)
        x = self.new_gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(torch.nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer = MLP(dim)

    def forward(self, x: Tensor):
        return x + self.layer(x)


class ResidualLogisticRegressor(torch.nn.Module):
    def __init__(self, in_features: int, depth: int, device: str):
        super().__init__()
        self._in_features = in_features
        self._depth = depth
        self.blocks = torch.nn.Sequential(*[Block(in_features) for _ in range(depth)])
        self.linear = torch.nn.Linear(in_features=in_features, out_features=1)
        self.optim = parameterfree.COCOB(self.parameters())
        self._device = device

    def clone(self) -> "ResidualLogisticRegressor":
        other = ResidualLogisticRegressor(self._in_features, self._depth, self._device)
        other.load_state_dict(self.state_dict())
        other.optim = parameterfree.COCOB(other.parameters())
        other.optim.load_state_dict(self.optim.state_dict())
        return other

    def forward(self, X: Tensor, A: Tensor) -> Tensor:
        return self.logits(X, A)

    def logits(self, X: Tensor, A: Tensor) -> Tensor:
        # X = batch x features
        # A = batch x actionbatch x actionfeatures

        Xreshap = X.unsqueeze(1).expand(
            -1, A.shape[1], -1
        )  # batch x actionbatch x features
        XA = (
            torch.cat((Xreshap, A), dim=-1)
            .reshape(X.shape[0], A.shape[1], -1)
            .to(self._device)
        )  # batch x actionbatch x (features + actionfeatures)
        return self.linear(self.blocks(XA)).squeeze(2)  # batch x actionbatch

    def predict(self, X: Tensor, A: Tensor) -> Tensor:
        self.eval()
        return torch.special.expit(self.logits(X, A))

    def bandit_learn(self, X: Tensor, A: Tensor, R: Tensor) -> float:
        self.train()
        self.optim.zero_grad()
        output = self(X, A)
        loss = F.binary_cross_entropy_with_logits(output, R)
        loss.backward()
        self.optim.step()
        return loss.item()
