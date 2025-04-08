"""
Define type aliases and classes for the measure-uq package.

This module provides utility functions and classes for use in the measure-uq package.

The module includes:
- Type aliases for array-like objects
- Functions for tensor operations like cartesian_product_of_rows
- DynamicArray class for efficient resizable arrays
- SparseDynamicArray class for memory-efficient sparse data storage
- Utility functions for working with polynomial expansions

Classes
-------
DynamicArray
    A dynamic array implementation that automatically resizes as elements are added.
SparseDynamicArray
    A sparse dynamic array implementation that efficiently stores non-zero values.

Functions
---------
cartesian_product_of_rows
    Compute the Cartesian product of the rows of multiple 2D tensors.
torch_numpoly_call
    Evaluate a polynomial expansion using PyTorch tensors.
"""

from collections.abc import Sequence
from copy import deepcopy
from typing import Any, NewType

import numpoly
import numpy as np
import numpy.typing as npt
import torch

INT_INF = (1 << 32) - 1

ArrayLike1DFloat = (
    Sequence[float] | Sequence[np.float32] | npt.NDArray[np.float32] | torch.FloatTensor
)
ArrayLike1DInt = (
    Sequence[int] | Sequence[np.int32] | npt.NDArray[np.int32] | torch.IntTensor
)

PolyExpansion = NewType("PolyExpansion", numpoly.baseclass.ndpoly)  # type: ignore [valid-newtype]


def cartesian_product_of_rows(*tensors: torch.Tensor) -> torch.Tensor:
    """
    Compute the Cartesian product of the rows of multiple 2D tensors.

    Parameters
    ----------
    *tensors : torch.Tensor
        A variable number of 2D tensors.

    Returns
    -------
    torch.Tensor
        A tensor containing all combinations of rows from the input tensors.
        Each row is a concatenation of rows from the input tensors.
    """
    assert len(tensors) > 0

    # Start with the first tensor
    result = tensors[0]

    for tensor in tensors[1:]:
        # Get the number of rows in the current result and the new tensor
        rows_result = result.shape[0]
        rows_tensor = tensor.shape[0]

        # Repeat the rows of the current result
        repeated_result = result.repeat_interleave(rows_tensor, dim=0)

        # Repeat the rows of the new tensor
        repeated_tensor = tensor.repeat(rows_result, 1)

        # Concatenate along the last dimension
        result = torch.cat([repeated_result, repeated_tensor], dim=1)

    return result


def extend_vector_tensor(
    x: torch.Tensor,
    n: int,
    default_value: float = 0,
) -> torch.Tensor:
    """
    Extend a tensor to a specified size, filling with a default value if necessary.

    Parameters
    ----------
    x : Tensor
        The tensor to extend.
    n : int
        The desired length of the output tensor.
    default_value : int or float, optional
        The default value to fill the tensor with when extending its length.
        Defaults to 0.

    Returns
    -------
    Tensor
        The extended tensor, with length `N`.
    """
    nx = x.shape[0]

    if n <= 0:
        raise ValueError("`n` must be positive.")

    if nx == 0:
        return torch.full((n,), default_value)

    if nx == n:
        return x

    if nx < n:
        z = torch.empty(n)
        z[:nx] = x
        z[nx:] = x[-1]
        return z

    raise ValueError("The size of `x` cannot be greater than the size of `y`.")


def torch_numpoly_call(
    exponents: torch.Tensor,
    coefficients: torch.Tensor,
    x: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the polynomial evaluation given exponents and coefficients.

    This function evaluates a polynomial function at given input points `x`,
    using specified `exponents` and `coefficients`.

    The function replicates the behavior of numpoly `call`. It uses only pytorch
    tensors in order to be executed on GPUs.

    Parameters
    ----------
    exponents : torch.Tensor
        A tensor of shape (K, D) representing the exponents of the polynomial terms.
    coefficients : torch.Tensor
        A tensor of shape (K, K) representing the coefficients of the polynomial terms.
    x : torch.Tensor
        A tensor of shape (N, D) representing the input points where the polynomial
        is to be evaluated.

    Returns
    -------
    torch.Tensor
        A tensor of shape (K, N) representing the evaluated polynomial at each input
        point.

    Examples
    --------
    >>> import chaospy
    >>> import numpy as np
    >>> from measure_uq.utilities import torch_numpoly_call

    >>> joint = chaospy.J(
    ...     chaospy.Uniform(1, 3),
    ...     chaospy.Uniform(-2, 1),
    ... )

    >>> expansion = chaospy.generate_expansion(
    ...     5,
    ...     joint,
    ...     normed=True,
    ... )

    >>> x = np.random.uniform(1, 3, (2, 10))

    >>> e1 = expansion(*x)

    >>> e2 = torch_numpoly_call(
    ...     torch.tensor(expansion.exponents, dtype=torch.float64),
    ...     torch.tensor(np.array(expansion.coefficients), dtype=torch.float64),
    ...     torch.tensor(x.T, dtype=torch.float64),
    ... ).T

    >>> np.allclose(e1, e2.detach().numpy(), atol=1e-12)
    True
    """
    X_exp = x.unsqueeze(1)  # (N, 1, D)
    E = exponents.unsqueeze(0)  # (1, K, D)
    X_pow = X_exp**E  # (N, K, D)

    monomials = torch.prod(X_pow, dim=2)  # (N, K)

    return monomials @ coefficients  # (N, K)


number_type = (int, float)
array_type = (np.ndarray, list, tuple)
value_type = array_type + number_type
value_alias = int | float | list[Any] | tuple[Any] | np.ndarray
index_alias = int | slice | tuple[int]


# This class is adapted from numpy_dynamic_array by Dylan Walsh
# Licensed under the BSD 3-Clause License. See NOTICE.txt for details.
class DynamicArray:
    """
    A dynamic array implementation that automatically resizes as elements are added.

    This class provides a numpy array-like interface with dynamic resizing capabilities.
    It efficiently manages memory by growing the underlying array as needed when new
    elements are added, while maintaining access to only the valid portion of data.

    Parameters
    ----------
    shape : int or tuple or list, optional
        The initial shape of the array. Default is 100.
    dtype : data-type, optional
        The data type of the array. Default is None, which corresponds to numpy's
        default data type.
    index_expansion : bool, optional
        If True, the array will automatically grow to accommodate indices beyond
        the current size. Default is False.

    Attributes
    ----------
    _data : ndarray
        The underlying numpy array that stores the data.
    capacity : int
        The total capacity of the underlying array.
    size : int
        The number of elements currently in use.
    index_expansion : bool
        Whether to automatically expand the array when accessing indices beyond size.

    Notes
    -----
    The array grows by either doubling its size or expanding to exactly fit the
    incoming data, whichever is larger. This amortizes the cost of resizing
    operations over time.

    Examples
    --------
    >>> a = DynamicArray((100, 2))
    >>> a.append(np.ones((20, 2)))
    >>> a.append(np.ones((120, 2)))
    >>> a.append(np.ones((10020, 2)))
    >>> print(a.data)
    >>> print(a.data.shape)
    """

    def __init__(
        self,
        shape: int | tuple[int, ...] | list[int] = 100,
        dtype: np.dtype[Any] | type[Any] | None = None,
        index_expansion: bool = False,
    ) -> None:
        self._data = np.zeros(shape, dtype) if dtype is not None else np.zeros(shape)
        self.capacity = self._data.shape[0]
        self.size = 0
        self.index_expansion = index_expansion

    def __str__(self) -> str:
        """Return a string representation of the DynamicArray."""
        return self.data.__str__()

    def __repr__(self) -> str:
        """Return a string representation of the DynamicArray."""
        return self.data.__repr__().replace(
            "array",
            f"DynamicArray(size={self.size}, capacity={self.capacity})",
        )

    def __getitem__(self, index: index_alias) -> value_alias:
        """Get item from the DynamicArray."""
        return self.data[index]

    def __setitem__(self, index: index_alias, value: value_alias) -> None:
        """
        Set value at specified index in the dynamic array.

        Parameters
        ----------
        index : int, slice, or tuple
            Index or indices where value should be set.
        value : array_like
            Value to set at the specified index.

        Raises
        ------
        IndexError
            If attempting to access an index outside the current size and
            index_expansion is False.

        Notes
        -----
        If index_expansion is True, the array will automatically grow to
        accommodate indices beyond the current size, filling intermediate
        values with zeros.
        """
        max_index = self._get_max_index(index)
        if not self.index_expansion and max_index > self.size:
            raise IndexError(
                "Attempting to reach index outside of data array. "
                f"Size: {self.size}, attempt index: {max_index}\n"
                "If you want the array to grow with indexing, "
                "set index_expansion to True.",
            )

        self._capacity_check_index(max_index)

        # add data
        if (
            isinstance(index, int)
            and index < 0
            or isinstance(index, slice)
            and (index.start < 0 or index.stop < 0)
            or isinstance(index, tuple)
            and any(i < 0 for i in index)
        ):
            # handle negative indexing
            self.data[index] = value
        else:
            # handling positive indexing
            self._data[index] = value

        # update capacity and size (if it was outside current size)
        if max_index > self.size:
            capacity_change = max_index - self.size
            self.capacity -= capacity_change
            self.size += capacity_change

    @staticmethod
    def _get_max_index(index: index_alias) -> int:
        """Get the maximum index of the DynamicArray."""
        if isinstance(index, slice):
            return int(index.stop)
        if isinstance(index, tuple):
            return int(index[0]) + 1

        # must be an int
        return index + 1

    def __getattr__(self, name: str) -> Any:
        """Get an attribute of the DynamicArray."""
        return getattr(self.data, name)

    def __add__(self, a: value_alias) -> value_alias:
        """Add two DynamicArrays."""
        return self.data + a

    def __eq__(self, a: object) -> bool:
        """Compare two DynamicArrays."""
        if isinstance(a, DynamicArray):
            return bool(np.equal(self.data, a.data).all())
        if isinstance(a, np.ndarray):
            return bool(np.equal(self.data, a).all())
        raise TypeError("Invalid item to compare.")

    def __floordiv__(self, a: value_alias) -> value_alias:
        """Floor divide two DynamicArrays."""
        return self.data // a

    def __mod__(self, a: value_alias) -> value_alias:
        """Modulo two DynamicArrays."""
        return self.data % a

    def __mul__(self, a: value_alias) -> value_alias:
        """Multiply two DynamicArrays."""
        return self.data * a

    def __neg__(self) -> value_alias:
        """Negate a DynamicArray."""
        return -self.data

    def __pow__(self, a: value_alias) -> value_alias:
        """Raise a DynamicArray to a power."""
        return self.data**a

    def __truediv__(self, a: value_alias) -> value_alias:
        """Divide two DynamicArrays."""
        return self.data / a

    def __sub__(self, a: value_alias) -> value_alias:
        """Subtract two DynamicArrays."""
        return self.data - a

    def __len__(self) -> int:
        """Get the length of the DynamicArray."""
        return self.size

    def append(self, x: value_alias) -> None:
        """Add data to array."""
        add_size = self._capacity_check(x)

        # Add new data to array
        self._data[self.size : self.size + add_size] = x
        self.size += add_size
        self.capacity -= add_size

    def _capacity_check_index(self, index: int = 0) -> None:
        """Check if the index is within the capacity of the DynamicArray."""
        if index > len(self._data):
            add_size = (index - len(self._data)) + self.capacity
            self._grow_capacity(add_size)

    def _capacity_check(self, x: value_alias) -> int:
        """Check if there is room for the new data."""
        if isinstance(x, number_type):
            add_size = 1
        elif isinstance(x, array_type):
            add_size = len(x)
        else:
            raise TypeError("Invalid item to add.")

        if add_size > self.capacity:
            self._grow_capacity(add_size)

        return add_size

    def _grow_capacity(self, add_size: int) -> None:
        """
        Grow the capacity of the dynamic array to accommodate new data.

        This method increases the capacity of the array by either doubling its size
        or growing it to exactly fit the incoming data, whichever is larger.

        Parameters
        ----------
        add_size : int
            The number of additional elements needed in the array.

        Notes
        -----
        The method calculates how much additional capacity is needed beyond the
        current capacity. If doubling the array size is sufficient, it does that;
        otherwise, it grows the array to exactly fit the incoming data.
        """
        # calculate what change is needed.
        change_need = add_size - self.capacity
        # make new larger data array
        shape_ = list(self._data.shape)
        if shape_[0] + self.capacity > add_size:
            # double in size
            self.capacity += shape_[0]
            shape_[0] = shape_[0] * 2
        else:
            # if doubling is not enough, grow to fit incoming data exactly.
            self.capacity += change_need
            shape_[0] = shape_[0] + change_need
        newdata = np.zeros(shape_, dtype=self._data.dtype)

        # copy data into new array and replace old one
        newdata[: self._data.shape[0]] = self._data
        self._data = newdata

    @property
    def data(self) -> np.ndarray:
        """Returns data without extra spaces."""
        return self._data[: self.size]

    def __reduce__(
        self,
    ) -> tuple[
        type["DynamicArray"],
        tuple[tuple[int, ...], np.dtype[Any], bool],
        dict[str, object],
    ]:
        """
        Prepare the DynamicArray for pickling.

        This method defines how the DynamicArray should be pickled by returning
        a tuple containing:
        1. The class itself
        2. Arguments to pass to the constructor
        3. State dictionary to restore the object's attributes

        Returns
        -------
        tuple
            A tuple containing the class, constructor arguments, and state dictionary.
        """
        return (
            self.__class__,
            (self._data.shape, self._data.dtype, self.index_expansion),
            {"_data": self._data, "size": self.size, "capacity": self.capacity},
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the DynamicArray from its pickled state."""
        self._data = state["_data"]
        self.size = state["size"]
        self.capacity = state["capacity"]

    def __deepcopy__(self, memo: dict[Any, Any]) -> "DynamicArray":
        """
        Create a deep copy of the DynamicArray instance.

        This method creates a new DynamicArray with the same shape, dtype, and
        index_expansion settings as the original, then copies all the data and state.

        Parameters
        ----------
        memo : dict
            Dictionary of objects already copied during the current copying pass.

        Returns
        -------
        DynamicArray
            A new DynamicArray instance with copied data and identical state.
        """
        new_obj = DynamicArray(
            shape=self._data.shape,
            dtype=self._data.dtype,
            index_expansion=self.index_expansion,
        )
        new_obj._data = self._data.copy()
        new_obj.capacity = self.capacity
        new_obj.size = self.size
        return new_obj


class SparseDynamicArray:
    """
    A sparse dynamic array implementation that efficiently stores non-zero values.

    This class provides a sparse array representation that only stores non-zero values
    and their corresponding indices, allowing for memory-efficient storage of sparse
    data. The array dynamically resizes as elements are added.

    Parameters
    ----------
    shape : int, optional
        The initial size of the array. Default is 100.
    dtype : data-type, optional
        The data type of the array. Default is None, which corresponds to numpy's
        default data type.

    Attributes
    ----------
    _values : DynamicArray
        Stores the non-zero values in the sparse array.
    _indices : DynamicArray
        Stores the indices corresponding to the non-zero values.
    _counter : int
        Tracks the number of non-zero elements stored.

    Notes
    -----
    This implementation is useful for scenarios where most elements in an array are zero
    or when dealing with large arrays with relatively few non-zero elements.
    """

    def __init__(
        self,
        shape: int | tuple[int] | list[int] = 100,
        dtype: np.dtype[Any] | type[Any] | None = None,
    ) -> None:
        """
        Initialize the SparseDynamicArray.

        Parameters
        ----------
        shape : int, optional
            The initial size of the array. By default, it is set to 100.
        dtype : data-type, optional
            The data type of the array. By default, it is set to None, which
            corresponds to numpy's default data type.
        """
        if isinstance(shape, tuple | list) and len(shape) != 1:
            raise ValueError("Shape tuple or list must be one-dimensional.")

        self._values = DynamicArray(shape, dtype, index_expansion=True)
        self._indices = DynamicArray(shape, int, index_expansion=True)
        self._counter = 0

    def __getitem__(self, index: int) -> value_alias:
        """
        Retrieve the value at the specified index.

        The index should take values from 0 to len(self) - 1.

        Parameters
        ----------
        index : int
            The index for which the value is to be retrieved. This index refers to the
            position in the internal storage array, not the sparse index.

        Returns
        -------
        value_alias
            The value at the specified index. The return type matches the dtype
            specified during initialization.
        """
        return self._values[index]

    def __setitem__(self, index: int, value: float) -> None:
        """
        Set the value at the specified index.

        Parameters
        ----------
        index : int
            The index for which the value is to be set.
        value : float
            The value to be set at the specified index.
        """
        self._values[self._counter] = value
        self._indices[self._counter] = index
        self._counter += 1

    def __call__(self, index: int) -> value_alias:
        """
        Retrieve the value at the specified index.

        This index correspond to the true sparse index in the array.

        Parameters
        ----------
        index : int
            The index for which the value is to be retrieved.

        Returns
        -------
        value_alias
            The value at the specified index. The return type matches the dtype
            specified during initialization.

        Raises
        ------
        IndexError
            If the specified index is not in the array.
        """
        i = int(np.searchsorted(self._indices, index))

        if i < self._counter and self._indices[i] == index:
            return self._values[i]

        if i > 0 and self._indices[i - 1] == index:
            return self._values[i - 1]

        raise IndexError(f"{index} is not an index in array")

    def __len__(self) -> int:
        """
        Return the length of the array, i.e. the number of elements it contains.

        Returns
        -------
        int
            The length of the array.
        """
        return self._counter

    def __str__(self) -> str:
        """Return a string representation of the SparseDynamicArray."""
        return (
            f"Indices: {self._indices.__str__()} \n"
            f" Values: {self._values.__str__()} \n "
        )

    @property
    def v(self) -> DynamicArray:
        """
        The values of the array.

        Returns
        -------
        array
            The values of the array.
        """
        return self._values

    @property
    def i(self) -> DynamicArray:
        """
        The indices of the array.

        Returns
        -------
        array
            The indices of the array.
        """
        return self._indices

    def __deepcopy__(self, memo: dict[Any, Any]) -> "SparseDynamicArray":
        """
        Create a deep copy of the SparseDynamicArray instance.

        This method creates a new SparseDynamicArray with the same shape as the original
        then copies all the data and state including values, indices, and counter.

        Parameters
        ----------
        memo : dict
            Dictionary of objects already copied during the current copying pass.

        Returns
        -------
        Self
            A new SparseDynamicArray instance with copied data and identical state.
        """
        new_obj = SparseDynamicArray(shape=self._values._data.shape[0])
        new_obj._values = deepcopy(self._values, memo)
        new_obj._indices = deepcopy(self._indices, memo)
        new_obj._counter = self._counter
        return new_obj

    def __reduce__(
        self,
    ) -> tuple[type["SparseDynamicArray"], tuple[int], dict[str, Any]]:
        """
        Prepare the SparseDynamicArray for pickling.

        This method defines how the SparseDynamicArray should be pickled by returning
        a tuple containing:
        1. The class itself
        2. Arguments to pass to the constructor
        3. State dictionary to restore the object's attributes

        Returns
        -------
        tuple
            A tuple containing the class, constructor arguments, and state dictionary.
        """
        return (
            self.__class__,
            (self._values._data.shape[0],),
            {
                "_values": self._values,
                "_indices": self._indices,
                "_counter": self._counter,
            },
        )

    def __setstate__(self, state: dict[str, Any]) -> None:
        """
        Restore the SparseDynamicArray from its pickled state.

        This method restores the state of a SparseDynamicArray instance from
        the dictionary of attributes saved during pickling.

        Parameters
        ----------
        state : dict
            Dictionary containing the saved attributes of the SparseDynamicArray.
        """
        self._values = state["_values"]
        self._indices = state["_indices"]
        self._counter = state["_counter"]
