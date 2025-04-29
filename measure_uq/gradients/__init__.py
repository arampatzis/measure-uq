"""
Provide functions for computing gradients, Jacobians and Hessians.

Provides functions for computing gradients, Jacobians and Hessians
using reverse-mode automatic differentiation.

The functions in this module are used for computing gradients of the loss
function of a model with respect to the parameters of the model.

The functions in this module are a modification of the `gradients` module,
copied from `deepxde`. Licensed under the GNU Lesser General Public License
(LGPL) 2.1. See the LICENSE.LGPL file in the root directory for details.
Original source: https://github.com/lululxvi/deepxde
"""

from measure_uq.gradients.gradients import clear, hessian, jacobian

__all__ = ["clear", "hessian", "jacobian"]
