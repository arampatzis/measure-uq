"""
Define type aliases and classes for the measure-uq package.

This module provides utility functions and classes for use in the measure-uq package.

This module provides:
    - Type aliases for array-like objects (ArrayLike1DFloat, ArrayLike1DInt)
    - Type alias for polynomial expansions (PolyExpansion)
    - DynamicArray class for resizable arrays
    - SparseDynamicArray class for efficient sparse array storage
    - Buffer class for managing collections of tensors
    - KeyControl class for keyboard control of program execution
    - Utility functions:
        - to_numpy: Convert tensors to numpy arrays
        - cartesian_product_of_rows: Compute the Cartesian product of the rows of
        multiple 2D tensors
        - extend_vector_tensor: Extend a tensor to a specified size
        - torch_numpoly_call: Evaluate a polynomial expansion using PyTorch tensors
"""

import atexit
import contextlib
import signal
import sys
import termios
import threading
import tty
from collections.abc import Sequence
from copy import deepcopy
from types import FrameType
from typing import Any, NewType

import matplotlib.pyplot as plt
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

number_type = (int, float)
array_type = (np.ndarray, list, tuple)
value_type = array_type + number_type
value_alias = int | float | list[Any] | tuple[Any] | np.ndarray
index_alias = int | slice | tuple[int]

numpy_array_like = np.ndarray | Sequence | float | int


def to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    """
    Convert an input to a NumPy array.

    This function takes an input, which can be a PyTorch tensor or any array-like
    object, and converts it to a NumPy array. If the input is a PyTorch tensor, it is
    first detached from the computation graph and moved to the CPU before being
    converted to a NumPy array.

    Parameters
    ----------
    x : torch.Tensor | np.ndarray
        The input to be converted to a NumPy array. It can be a PyTorch tensor or
        a numpy array.

    Returns
    -------
    numpy.ndarray
        The converted NumPy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.array(x)


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
        """
        Initialize a DynamicArray instance.

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

        Returns
        -------
        None
            This method does not return anything.
        """
        self._data = np.zeros(shape, dtype) if dtype is not None else np.zeros(shape)
        self.capacity = self._data.shape[0]
        self.size = 0
        self.index_expansion = index_expansion

    def __str__(self) -> str:
        """
        Return a string representation of the DynamicArray.

        Returns
        -------
        str
            A string representation of the DynamicArray.
        """
        return self.data.__str__()

    def __repr__(self) -> str:
        """
        Return a string representation of the DynamicArray.

        Returns
        -------
        str
            A string representation of the DynamicArray.
        """
        return self.data.__repr__().replace(
            "array",
            f"DynamicArray(size={self.size}, capacity={self.capacity})",
        )

    def __getitem__(self, index: index_alias) -> value_alias:
        """
        Get item from the DynamicArray.

        Parameters
        ----------
        index : int, slice, or tuple
            Index or indices to retrieve from the array.

        Returns
        -------
        value_alias
            The value at the specified index.
        """
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
        """
        Get the maximum index of the DynamicArray.

        Parameters
        ----------
        index : int, slice, or tuple
            Index or indices to evaluate.

        Returns
        -------
        int
            The maximum index value.
        """
        if isinstance(index, slice):
            return int(index.stop)
        if isinstance(index, tuple):
            return int(index[0]) + 1

        # must be an int
        return index + 1

    def __getattr__(self, name: str) -> Any:
        """
        Get an attribute of the DynamicArray.

        Parameters
        ----------
        name : str
            The name of the attribute to retrieve.

        Returns
        -------
        Any
            The attribute value.
        """
        return getattr(self.data, name)

    def __add__(self, a: value_alias) -> value_alias:
        """
        Add two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to add to the DynamicArray.

        Returns
        -------
        value_alias
            The result of the addition.
        """
        return self.data + a

    def __eq__(self, a: object) -> bool:
        """
        Compare two DynamicArrays.

        Parameters
        ----------
        a : object
            The object to compare with the DynamicArray.

        Returns
        -------
        bool
            True if the objects are equal, False otherwise.
        """
        if isinstance(a, DynamicArray):
            return bool(np.equal(self.data, a.data).all())
        if isinstance(a, np.ndarray):
            return bool(np.equal(self.data, a).all())
        raise TypeError("Invalid item to compare.")

    def __floordiv__(self, a: value_alias) -> value_alias:
        """
        Floor divide two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to floor divide the DynamicArray by.

        Returns
        -------
        value_alias
            The result of the floor division.
        """
        return self.data // a

    def __mod__(self, a: value_alias) -> value_alias:
        """
        Modulo two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to modulo the DynamicArray by.

        Returns
        -------
        value_alias
            The result of the modulo operation.
        """
        return self.data % a

    def __mul__(self, a: value_alias) -> value_alias:
        """
        Multiply two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to multiply the DynamicArray by.

        Returns
        -------
        value_alias
            The result of the multiplication.
        """
        return self.data * a

    def __neg__(self) -> value_alias:
        """
        Negate a DynamicArray.

        Returns
        -------
        value_alias
            The negated DynamicArray.
        """
        return -self.data

    def __pow__(self, a: value_alias) -> value_alias:
        """
        Raise a DynamicArray to a power.

        Parameters
        ----------
        a : value_alias
            The exponent to raise the DynamicArray to.

        Returns
        -------
        value_alias
            The result of the power operation.
        """
        return self.data**a

    def __truediv__(self, a: value_alias) -> value_alias:
        """
        Divide two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to divide the DynamicArray by.

        Returns
        -------
        value_alias
            The result of the division.
        """
        return self.data / a

    def __sub__(self, a: value_alias) -> value_alias:
        """
        Subtract two DynamicArrays.

        Parameters
        ----------
        a : value_alias
            The value to subtract from the DynamicArray.

        Returns
        -------
        value_alias
            The result of the subtraction.
        """
        return self.data - a

    def __len__(self) -> int:
        """
        Get the length of the DynamicArray.

        Returns
        -------
        int
            The number of elements in the DynamicArray.
        """
        return self.size

    def append(self, x: value_alias) -> None:
        """
        Add data to the DynamicArray.

        Parameters
        ----------
        x : value_alias
            The data to append to the DynamicArray.

        Returns
        -------
        None
            This method does not return anything.
        """
        add_size = self._capacity_check(x)

        # Add new data to array
        self._data[self.size : self.size + add_size] = x
        self.size += add_size
        self.capacity -= add_size

    def _capacity_check_index(self, index: int = 0) -> None:
        """
        Check if the index is within the capacity of the DynamicArray.

        Parameters
        ----------
        index : int, optional
            The index to check. Default is 0.

        Returns
        -------
        None
            This method does not return anything.
        """
        if index > len(self._data):
            add_size = (index - len(self._data)) + self.capacity
            self._grow_capacity(add_size)

    def _capacity_check(self, x: value_alias) -> int:
        """
        Check if there is room for the new data.

        Parameters
        ----------
        x : value_alias
            The data to check capacity for.

        Returns
        -------
        int
            The size of the data to be added.
        """
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
        """
        Return data without extra spaces.

        Returns
        -------
        np.ndarray
            The data of the DynamicArray without extra spaces.
        """
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
        """
        Restore the DynamicArray from its pickled state.

        Parameters
        ----------
        state : dict[str, Any]
            The state dictionary to restore the DynamicArray from.

        Returns
        -------
        None
            This method does not return anything.
        """
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
        """
        Return a string representation of the SparseDynamicArray.

        Returns
        -------
        str
            A string representation of the SparseDynamicArray.
        """
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


class Buffer:
    """
    A class to manage a collection of tensors.

    This class provides methods to register tensors, move them to a specified device,
    and save/load their state.

    Attributes
    ----------
    _buffers : dict
        A dictionary to store the registered tensors.

    Methods
    -------
    register(name, tensor)
        Register a tensor with a given name.
    to(device)
        Move all registered tensors to the specified device.
    state_dict()
        Return a dictionary containing the state of all registered tensors.
    load_state_dict(state_dict)
        Load the state of the registered tensors from a dictionary.
    """

    def __init__(self) -> None:
        """
        Initialize the Buffer instance.

        This method initializes the Buffer instance by setting up the internal
        dictionary to store registered tensors.
        """
        self._buffers: dict[str, torch.Tensor] = {}

    def register(self, name: str, tensor: torch.Tensor) -> None:
        """
        Register a tensor with a given name.

        This method registers a tensor under a specified name in the internal
        dictionary of the Buffer instance. The tensor can later be accessed,
        moved to a different device, or saved/loaded using the other methods
        of the Buffer class.

        Parameters
        ----------
        name : str
            The name to register the tensor under.
        tensor : torch.Tensor
            The tensor to be registered.

        Returns
        -------
        None
            This method does not return anything.

        Raises
        ------
        TypeError
            If the provided tensor is not an instance of torch.Tensor.
        """
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"Expected torch.Tensor, got {type(tensor)}")
        self._buffers[name] = tensor

    def __getitem__(self, name: str) -> torch.Tensor:
        """
        Retrieve a registered tensor by name.

        This method allows access to a tensor that has been registered in the
        Buffer instance using its name.

        Parameters
        ----------
        name : str
            The name of the registered tensor to retrieve.

        Returns
        -------
        torch.Tensor
            The tensor registered under the specified name.

        Raises
        ------
        KeyError
            If no tensor is registered under the specified name.
        """
        return self._buffers[name]

    def to(self, device: torch.device | str) -> None:
        """
        Move all registered tensors to the specified device.

        This method iterates over all tensors registered in the Buffer instance
        and moves them to the specified device (e.g., CPU, GPU).

        Parameters
        ----------
        device : torch.device
            The device to move the tensors to.

        Returns
        -------
        None
            This method does not return anything.
        """
        for name, tensor in self._buffers.items():
            self._buffers[name] = tensor.to(device)

    def state_dict(self) -> dict[str, torch.Tensor]:
        """
        Return a dictionary containing all registered tensors.

        This method creates a dictionary where the keys are the names of the
        registered tensors and the values are the tensors themselves, moved to
        the CPU.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary with tensor names as keys and tensors as values,
            all moved to the CPU.
        """
        return {name: tensor.cpu() for name, tensor in self._buffers.items()}

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        """
        Load tensors from a state dictionary.

        This method takes a state dictionary, where the keys are tensor names
        and the values are the corresponding tensors, and loads the tensors
        into the Buffer instance.

        Parameters
        ----------
        state_dict : dict[str, torch.Tensor]
            A dictionary containing tensor names as keys and tensors as values.

        Returns
        -------
        None
            This method does not return anything.
        """
        for name, tensor in state_dict.items():
            self._buffers[name] = tensor


class KeyController:
    """
    Standalone key controller for pause/resume/stop control in any loop.

    This class provides a mechanism to control the execution of loops using
    keyboard inputs. It supports pausing, resuming, and stopping the loop
    based on user-defined key bindings. The class can be configured to use
    separate keys for pause and resume or a single toggle key for both actions.

    Parameters
    ----------
    key_bindings : dict, optional
        A dictionary specifying the keys for controlling the loop. The default
        is {'pause': 'p', 'resume': 'r', 'stop': 'q'} for separate pause and
        resume keys, or {'toggle': ' ', 'stop': 'q'} if `use_toggle` is True.
    use_toggle : bool, optional
        If True, a single key is used to toggle between pause and resume states.
        The default is False.

    Attributes
    ----------
    stop_requested : bool
        Indicates whether a stop has been requested.
    pause_requested : bool
        Indicates whether a pause has been requested.
    use_toggle : bool
        Determines if a single key is used for toggling pause and resume.

    Methods
    -------
    check_pause()
        Checks if a pause has been requested and waits until resumed.
    close()
        Restores the terminal to its original settings and stops the key listener.
    restore_terminal()
        Restores the terminal to its original settings.
    """

    stop_requested: bool
    pause_requested: bool
    use_toggle: bool
    _closed: bool
    fd: int | None
    old_settings: list | None

    def __init__(
        self,
        key_bindings: dict | None = None,
        use_toggle: bool = False,
    ):
        """
        Standalone key controller for pause/resume/stop control in any loop.

        Parameters
        ----------
        key_bindings : dict, optional
            {'pause': 'p', 'resume': 'r', 'stop': 'q'}
            or {'toggle': ' ', 'stop': 'q'} if use_toggle is True.
        use_toggle : bool, optional
            Use a single key to toggle pause/resume.
        """
        self.stop_requested = False
        self.pause_requested = False
        self.use_toggle = use_toggle
        self._closed = False

        # Setup keys
        default_keys = (
            {"pause": "p", "resume": "r", "stop": "q"}
            if not use_toggle
            else {"toggle": " ", "stop": "q"}
        )
        self.keys = {**default_keys, **(key_bindings or {})}

        # Terminal config
        try:
            self.fd = sys.stdin.fileno()
            self.old_settings = termios.tcgetattr(self.fd)
        except (termios.error, OSError):
            print(
                "[Warning] Failed to access terminal settings — key control disabled.",
            )
            self.fd = None
            self.old_settings = None
            return

        # Register cleanup
        atexit.register(self.close)

        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_sigint)

        self._print_controls()
        threading.Thread(target=self._key_listener, daemon=True).start()

    def _print_controls(self) -> None:
        """
        Print the key controls for the user.

        This method displays the key bindings to the user, indicating which
        keys are used for pausing, resuming, and stopping the loop.
        """
        print("Key controls:")
        for action, key in self.keys.items():
            print(f"Press '{key}' to {action}.")

    def _handle_sigint(self, _signum: int, _frame: FrameType | None) -> None:
        """
        Handle the SIGINT signal to stop the loop.

        This method is called when a SIGINT signal is received (e.g., when
        the user presses Ctrl+C). It sets the stop_requested flag to True.

        Parameters
        ----------
        _signum : int
            The signal number.
        _frame : FrameType
            The current stack frame.
        """
        self.stop_requested = True

    def _key_listener(self) -> None:
        """Listen for key presses to control the loop."""
        if self.fd is None:
            return
        try:
            tty.setcbreak(self.fd)
            while not self.stop_requested:
                try:
                    key = sys.stdin.read(1)
                except KeyboardInterrupt:
                    self.stop_requested = True
                    break

                if self.use_toggle:
                    self._handle_toggle_mode(key)
                else:
                    self._handle_separate_mode(key)
        finally:
            self.close()

    def _handle_toggle_mode(self, key: str) -> None:
        """
        Handle the toggle mode.

        This method handles the toggle mode, where a single key is used to toggle
        between pause and resume states.

        Parameters
        ----------
        key : str
            The key pressed by the user.
        """
        if key == self.keys["toggle"]:
            self.pause_requested = not self.pause_requested
            print("[Paused]" if self.pause_requested else "[Resumed]")
        elif key == self.keys["stop"]:
            print("[Stop Requested]")
            self.stop_requested = True

    def _handle_separate_mode(self, key: str) -> None:
        """
        Handle the separate mode.

        This method handles the separate mode, where separate keys are used for
        pause and resume.

        Parameters
        ----------
        key : str
            The key pressed by the user.
        """
        if key == self.keys["pause"]:
            if not self.pause_requested:
                print(f"[Paused] Press '{self.keys['resume']}' to resume.")
            self.pause_requested = True
        elif key == self.keys["resume"]:
            if self.pause_requested:
                print("[Resumed]")
            self.pause_requested = False
        elif key == self.keys["stop"]:
            print("[Stop Requested]")
            self.stop_requested = True

    def check_pause(self) -> None:
        """Call this from inside your loop to respect pause/stop flags."""
        if self.pause_requested:
            print("[Paused] Waiting to resume...")
        while self.pause_requested and not self.stop_requested:
            plt.pause(0.1)

    def close(self) -> None:
        """Restore terminal and signal state safely (idempotent)."""
        if self._closed:
            return
        self._closed = True
        if self.fd is not None and self.old_settings is not None:
            with contextlib.suppress(Exception):
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)
        with contextlib.suppress(Exception):
            signal.signal(signal.SIGINT, self.original_sigint_handler)
