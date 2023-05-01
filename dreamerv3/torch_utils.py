import random
import torch
import torch.nn.functional as F
import numpy as np
import os

def random_seeding(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)


def symlog(x):
    return x.sign() * x.abs().add(1.0).log()

def symexp(x):
    return x.sign() * x.abs().exp().add(-1.0)

class MSEDist:
    def __init__(self, mode, dims, agg="sum"):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._agg = agg
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return self._mode

    def mean(self):
        return self._mode

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        distance = (self._mode - value) ** 2
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss

class SymlogDist:
    def __init__(self, mode, dims, dist="mse", agg="sum", tol=1e-8):
        self._mode = mode
        self._dims = tuple([-x for x in range(1, dims + 1)])
        self._dist = dist
        self._agg = agg
        self._tol = tol
        self.batch_shape = mode.shape[: len(mode.shape) - dims]
        self.event_shape = mode.shape[len(mode.shape) - dims :]

    def mode(self):
        return symexp(self._mode)

    def mean(self):
        return symexp(self._mode)

    def log_prob(self, value):
        assert self._mode.shape == value.shape, (self._mode.shape, value.shape)
        if self._dist == "mse":
            distance = (self._mode - symlog(value)) ** 2
            distance = torch.where(distance < self._tol, 0, distance)
        elif self._dist == "abs":
            distance = torch.abs(self._mode - symlog(value))
            distance = torch.where(distance < self._tol, 0, distance)
        else:
            raise NotImplementedError(self._dist)
        if self._agg == "mean":
            loss = distance.mean(self._dims)
        elif self._agg == "sum":
            loss = distance.sum(self._dims)
        else:
            raise NotImplementedError(self._agg)
        return -loss


class DiscDist:
    def __init__(self, logits, dims=0, low=-20, high=20, transfwd=symlog, transbwd=symexp):
        self.logits = logits
        self.probs = F.softmax(logits)
        self.dims = tuple([-x for x in range(1, dims + 1)])
        self.bins = torch.linspace(low, high, logits.shape[-1])
        self.low = low
        self.high = high
        self.transfwd = transfwd
        self.transbwd = transbwd
        self.batch_shape = logits.shape[: len(logits.shape) - dims - 1]
        self.event_shape = logits.shape[len(logits.shape) - dims : -1]

    def mean(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def mode(self):
        return self.transbwd((self.probs * self.bins).sum(-1))

    def log_prob(self, x):
        x = self.transfwd(x)
        below = (self.bins <= x[..., None]).int().sum(-1) - 1
        above = len(self.bins) - (self.bins > x[..., None]).int().sum(-1)
        below = torch.clip(below, 0, len(self.bins) - 1)
        above = torch.clip(above, 0, len(self.bins) - 1)
        equal = below == above
        dist_to_below = torch.where(equal, 1, torch.abs(self.bins[below] - x))
        dist_to_above = torch.where(equal, 1, torch.abs(self.bins[above] - x))
        total = dist_to_below + dist_to_above
        weight_below = dist_to_above / total
        weight_above = dist_to_below / total
        target = (
            F.one_hot(below, len(self.bins)) * weight_below[..., None]
            + F.one_hot(above, len(self.bins)) * weight_above[..., None]
        )
        log_pred = self.logits - torch.logsumexp(self.logits, -1, keepdims=True)
        return (target * log_pred).sum(-1).sum(self.dims)