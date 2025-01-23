"""Define custom types."""

from collections.abc import Sequence
from typing import NewType

import numpoly
import numpy as np
import numpy.typing as npt
import torch

ArrayLike1DFloat = Sequence[np.float32] | npt.NDArray[np.float32] | torch.FloatTensor
ArrayLike1DInt = Sequence[np.int32] | npt.NDArray[np.int32] | torch.IntTensor

PolyExpansion = NewType("PolyExpansion", numpoly.baseclass.ndpoly)  # type: ignore [valid-newtype]
