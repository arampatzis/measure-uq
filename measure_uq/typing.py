"""Define custom types."""
from collections.abc import Sequence

import numpy as np
import torch

ArrayLike1D = Sequence[float] | np.ndarray | torch.Tensor
