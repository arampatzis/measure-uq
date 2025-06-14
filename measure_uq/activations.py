"""Implementations of custom activation functions."""

from typing import Any

import torch
from torch import addcdiv, sin, square
from torch.autograd import Function
from torch.nn import Module, Parameter

# The snake function implementations is taken from https://github.com/EdwardDixon/snake
# There is no license.


class SnakeFunction(Function):
    r"""
    Snake activation function.

    Autograd function implementing the serpentine-like sine-based periodic activation
    function.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \\sin^2(ax)

    This function computes the forward and backward pass for the Snake activation,
    which helps in better extrapolating to unseen data, particularly when dealing with
    periodic functions.

    Attributes
    ----------
        ctx (torch.autograd.function._ContextMethodMixin): Context object used for
        saving and retrieving tensors.
    """

    @staticmethod
    def forward(
        ctx: Any,
        x: torch.Tensor,
        a: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass for the Snake activation function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object used for saving and retrieving tensors.
        x : torch.Tensor
            Input tensor.
        a : torch.Tensor
            Trainable parameter controlling the frequency of the sine function.

        Returns
        -------
        torch.Tensor
            Result of applying the Snake activation function to the input tensor.
        """
        ctx.save_for_backward(x, a)

        # Handle case where `a` is zero to avoid division by zero errors.
        return torch.where(a == 0, x, addcdiv(x, square(sin(a * x)), a))

    @staticmethod
    def backward(  # type: ignore[no-untyped-def]
        ctx,
        grad_output,
    ):
        """
        Backward pass for the Snake activation function.

        Parameters
        ----------
        ctx : torch.autograd.function._ContextMethodMixin
            Context object used for saving and retrieving tensors.
        grad_output : torch.Tensor
            The gradient of the loss with respect to the output.

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            The gradients of the loss with respect to `x` and `a`.
        """
        x, a = ctx.saved_tensors

        # Calculate the gradient of the input `x`
        sin2ax = sin(2 * a * x) if any(ctx.needs_input_grad) else None
        grad_x = grad_output * (1 + sin2ax) if ctx.needs_input_grad[0] else None

        # Calculate the gradient of the parameter `a`
        grad_a = (
            grad_output
            * torch.where(a == 0, square(x), sin2ax * x / a - square(sin(a * x) / a))
            if ctx.needs_input_grad[1]
            else None
        )

        return grad_x, grad_a


class Snake(Module):
    r"""
    Implementation of the Snake activation function as a torch module.

    .. math::
         \text{Snake}_a := x + \frac{1}{a} \\sin^2(ax) =
         x - \frac{1}{2a}\\cos(2ax) + \frac{1}{2a}

    This activation function is designed to better extrapolate unseen data,
    particularly periodic functions.

    Parameters
    ----------
    in_features : int | list[int]
        The shape or number of input features.
    a : float, optional
        Initial value of the trainable parameter `a`, controlling the sine frequency.
        Defaults to None.
    trainable : bool, optional
        If `True`, the parameter `a` will be trainable. Defaults to True.

    References
    ----------
        This activation function is from this paper by Liu et.al.:
        https://arxiv.org/abs/2006.08195

    Examples
    --------
        >>> snake_layer = Snake(256)
        >>> x = torch.randn(256)
        >>> x = snake_layer(x)
    """

    def __init__(
        self,
        in_features: int | list[int],
        a: float = 1.0,
        trainable: bool = True,
    ):
        """
        Initialize the Snake activation layer.

        Parameters
        ----------
            in_features : int | list[int]
                Shape of the input, either a single integer or a list of integers
                indicating feature dimensions.
            a : float
                Initial value for the parameter `a`, which controls the sine frequency.
            trainable : bool
                If `True`, the parameter `a` will be trained during backpropagation.
        """
        super().__init__()
        self.in_features = (
            in_features if isinstance(in_features, list) else [in_features]
        )

        # Ensure initial_a is a floating point tensor
        if isinstance(in_features, int):
            initial_a = torch.full((in_features,), a, dtype=torch.float32)
            # Explicitly set dtype to float32
        else:
            initial_a = torch.full(in_features, a, dtype=torch.float32)
            # Assuming in_features is a list/tuple of dimensions

        if trainable:
            self.a = Parameter(initial_a)
        else:
            self.register_buffer("a", initial_a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Snake activation layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor to the layer.

        Returns
        -------
        torch.Tensor
            Result of applying the Snake activation function.
        """
        return SnakeFunction.apply(x, self.a)  # type: ignore[no-untyped-call]
