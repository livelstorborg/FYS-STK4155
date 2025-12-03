import jax
import flax.nnx as nnx
import jax.numpy as jnp


class MLP(nnx.Module):
    def __init__(self, layers, activations, key):
        assert len(activations) == len(layers) - 2, (
            "Length of activations must be number of layers minus 2 "
            "(one activation per hidden layer)."
        )

        self.rngs = nnx.Rngs(params=key)
        self.layers = nnx.List(
            [
                nnx.Linear(layers[i], layers[i + 1], rngs=self.rngs)
                for i in range(len(layers) - 1)
            ]
        )

        self.activations = activations

    def __call__(self, x):
        for linear, act in zip(self.layers[:-1], self.activations):
            x = act(linear(x))
        return self.layers[-1](x)
