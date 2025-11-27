import jax
import flax.nnx as nnx
import jax.numpy as jnp


class MLP(nnx.Module):
    def __init__(self, layers, key):
        self.rngs = nnx.Rngs(params=key)
        L = [
            nnx.Linear(layers[i], layers[i + 1], rngs=self.rngs)
            for i in range(len(layers) - 1)
        ]
        self.layers = nnx.List(L)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.tanh(layer(x))
        return self.layers[-1](x)
