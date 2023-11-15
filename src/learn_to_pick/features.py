from typing import Union, Optional, Dict, List
import numpy as np


class SparseFeatures(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DenseFeatures(list):
    def __init__(self, *args, **kwargs):
        super().__init__(np.array(*args, **kwargs))


class Featurized:
    def __init__(
        self,
        sparse: Optional[Dict[str, SparseFeatures]] = None,
        dense: Optional[Dict[str, DenseFeatures]] = None,
    ):
        self.sparse = sparse or {}
        self.dense = dense or {}

    def __setitem__(self, key, value):
        if isinstance(value, Dict):
            self.sparse[key] = SparseFeatures(value)
        elif isinstance(value, List) or isinstance(value, np.ndarray):
            self.dense[key] = DenseFeatures(value)
        else:
            raise ValueError(
                f"Cannot convert {type(value)} to either DenseFeatures or SparseFeatures"
            )

    def merge(self, other):
        self.sparse.update(other.sparse)
        self.dense.update(other.dense)
