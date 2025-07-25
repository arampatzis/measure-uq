# measure-uq

This repository contains the code for the paper
[A generative modeling / Physics-Informed Neural Network approach to random differential equations](https://arxiv.org/abs/2507.01687)

The code implements a Physics-Informed Neural Network (PINN) and
Polynomial Chaos Expansion (PCE) to solve random partial differential equations.

## 📦 Installation

To install the dependencies and the project, run:
```bash
poetry install
```

Either use the `shell` pluginn to open a shell with the project's environment activated:
```bash
poetry self add poetry-plugin-shell
poetry shell
```

or run the following command to activate the environment:
```bash
eval $(poetry env activate)
```


## 🔩 Usage

You can run the notebooks in the `examples` directory to see the code in action.


## Create a new example

### General form of a random partial differential equation

Consider the the following general form of a random partial differential equation:

$$
\mathcal{L}_\xi u(z, \xi) = f(z, \xi), \quad z \in C, \quad \xi \in \Xi \,.
$$

where $z$ is a vector of spatial and temporal variables,
and $\xi$ is a random vector with support on $\Xi$ that follows the distribution
$\gamma$.

The initial and boundary conditions are given by
$$
\mathcal{C}_{\xi, i} u(z, \xi) = g_i(z, \xi), \quad z \in C_i \subset C, \quad \xi \in \Xi \,,
$$

for $i = 1, \ldots, m$.

We call $\mathcal{L}$ the PDE operator and
$\mathcal{C}_i$ the initial or the boundary condition operator.


### The heat equation in 1D

The heat equation on the domain $[0,L]$ is given by

$$
u_t - k \, u_{xx} = 0, \quad t \in [0,T],\quad x\in [0,L]  \,,
$$

where $k\sim\mathcal{U}(1,2)$.
The initial and boundary conditions are given by

$$
u(0,x) = u_0(x), \quad x \in [0,L] \,,
$$

$$
u(t,0) = u_1(t), \quad t \in [0,T] \,,
$$

$$
u(t,L) = u_2(t), \quad t \in [0,T] \,,
$$


### The heat equation written in the general form

This PDE can be written in the form of a general form of a partial differential equation
as

$$
\mathcal{L}_\xi u(z, \xi) = u_{z_1} - \xi \, u_{z_2z_2} = 0
\,, \quad z \in C \,, \quad \xi \in \Xi \,,
$$

with $z\in C = [0,T] \times [0,L]$.
Notice that in this notation, the vector $z$ encodes the spatial and temporal variables,
i.e. $z = (t, x)$.

The $m=3$ conditions, 1 initial and 2 boundary conditions, can be written as

$$
\mathcal{C}_{\xi, i} = I\,, \quad i=1,2,3 \,,
$$

where $I$ is the identity operator,

$$
g_i(z, \xi) = u_i(z, \xi)\,, \quad z \in C_i \,, \quad \xi \in \Xi \,,
$$

and

$$
C_1 = \{ (0,z_2) \in C \mid z_2 \in [0,L] \} \,,
$$

$$
C_2 = \{ (z_1,0) \in C \mid z_1 \in [0,T] \} \,,
$$

$$
C_3 = \{ (z_1,L) \in C \mid z_1 \in [0,T] \} \,.
$$


### Implementation of the heat equation in 1D

To implement the heat equation in 1D with random parameters, you need to define:

1. **A class for sampling the random parameters** (inheriting from `Parameters`).
2. **Classes for the PDE residual, initial, and boundary conditions** (each inheriting from `Condition`).

These classes allow you to generate samples of the random parameters $\xi$ and points $z$ in the domain or on the boundaries, which are then used to train the PINN.

#### 1. Random Parameters

You must implement a class that inherits from `Parameters` and defines how to sample the random parameters. For example:

```python
@dataclass(kw_only=True)
class RandomParameters(Parameters):
    """
    Random parameters for the PDE.
    """
    N: int  # Number of samples

    def sample_values(self) -> None:
        """Re-sample the parameters of the PDE."""
        self.values = tensor(np.random.uniform(1, 2, (self.N, 1))).float()
```

This samples the thermal diffusivity $k$ from a uniform distribution $\mathcal{U}(1, 2)$.

#### 2. PDE Residual Condition

The residual of the heat equation is implemented as:

```python
@dataclass(kw_only=True)
class Residual(Condition):
    Nt: int
    Nx: int
    T: float
    L: float

    def sample_points(self) -> None:
        """Sample points for the PDE."""
        self.points = cartesian_product_of_rows(
            tensor(np.random.uniform(0, self.T, (self.Nt, 1))),
            tensor(np.random.uniform(0, self.L, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Evaluate the residual of the PDE.
        """
        dy_dt = jacobian(y, x, j=0)
        dy_dx = jacobian(y, x, j=1)
        dy_d2 = jacobian(dy_dx, x, j=1)
        k = x[:, 2][:, None]
        return dy_dt - k * dy_d2
```

- `sample_points` generates random time and space points in the domain.
- `eval` computes the residual $u_t - k u_{xx}$ at the sampled points.

#### 3. Initial and Boundary Conditions

Similarly, you define classes for the initial and boundary conditions. For example, the initial condition:

```python
@dataclass(kw_only=True)
class InitialCondition(Condition):
    Nx: int
    L: float

    def sample_points(self) -> None:
        self.points = cartesian_product_of_rows(
            torch.tensor([[0.0]]),
            tensor(np.random.uniform(0, self.L, (self.Nx, 1))),
        ).float()

    def eval(self, x: Tensor, y: Tensor) -> Tensor:
        # Example: u(0, x) = sin(pi * x)
        return y - torch.sin(np.pi * x[:, 1:2])
```

Boundary conditions are implemented in a similar way, sampling points on the boundaries and defining the corresponding `eval` method.

#### 4. Usage

- The `sample_points` methods generate the training data for the PINN.
- The `eval` methods compute the loss terms for the PDE and the conditions.

**Note:** The input tensor `x` is constructed automatically by concatenating the
sampled $z$ (time and space) and the random parameter $k$.
Thus, the values of the time variable are store in the first column of `x`,
the values of the space variable are store in the second column of `x`,
and the values of the parameter $k$ are store in the third column of `x`.

Each row of the tensor `x` corresponds to a single sample that is used to train the PINN.




For a complete, working example, see [`examples/equations/heat_1d/pde.py`](examples/equations/heat_1d/pde.py).





## 🫱🏽‍🫲🏻 Acknowledgments
This project makes use of code from `deepxde`, which is available under the BSD 3-Clause License.

Original project URL: https://github.com/lululxvi/deepxde

See the `NOTICE.txt` file for the full license text.
