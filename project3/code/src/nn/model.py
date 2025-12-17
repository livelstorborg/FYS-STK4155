import jax
import flax.nnx as nnx
import jax.numpy as jnp


# class MLP(nnx.Module):
#     def __init__(self, layers, activations, key):
#         assert len(activations) == len(layers) - 2, (
#             "Length of activations must be number of layers minus 2 "
#             "(one activation per hidden layer)."
#         )

#         self.rngs = nnx.Rngs(params=key)
#         self.layers = nnx.List(
#             [
#                 nnx.Linear(layers[i], layers[i + 1], rngs=self.rngs)
#                 for i in range(len(layers) - 1)
#             ]
#         )

#         self.activations = activations

#     def __call__(self, x):
#         for linear, act in zip(self.layers[:-1], self.activations):
#             x = act(linear(x))
#         return self.layers[-1](x)


class MLP_HardBC(nnx.Module):
    """
    MLP with hard boundary conditions.

    Trial solution: u(x,t) = (1-t)*sin(πx) + t*x*(1-x)*N(x,t)

    This automatically satisfies:
    - BC: u(0,t) = 0, u(1,t) = 0
    - IC: u(x,0) = sin(πx)
    """

    def __init__(self, layers, activations, key):
        self.network = MLP(layers, activations, key)

    def __call__(self, xt):
        """
        xt: (N, 2) array where xt[:, 0] = x, xt[:, 1] = t
        """
        x = xt[:, 0:1]  # shape (N, 1)
        t = xt[:, 1:2]  # shape (N, 1)

        # Get unconstrained network output
        N = self.network(xt)

        # Apply hard BC: u = (1-t)*sin(πx) + t*x*(1-x)*N
        u = (1.0 - t) * jnp.sin(jnp.pi * x) + t * x * (1.0 - x) * N

        return u
