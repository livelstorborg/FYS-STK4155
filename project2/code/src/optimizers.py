import numpy as np


class Optimizer:
    """Base class for optimizers."""

    def __init__(self, update_batch_norm_params: bool = False):
        self.update_batch_norm_params = update_batch_norm_params

    def update(self, nn, layer_grads, bn_grads=None):
        """
        Update network parameters using gradients.

        Args:
            nn: NeuralNetwork instance
            layer_grads: List of (dW, db) tuples from compute_gradient
            bn_grads: Optional list of (dgamma, dbeta) tuples for BatchNorm layers
        """
        raise NotImplementedError

    def reset(self):
        """Reset optimizer state between training runs."""
        pass


class GD(Optimizer):
    """Plain Gradient Descent (batch or mini-batch)."""

    def __init__(self, eta=0.01, update_batch_norm_params: bool = False):
        super().__init__(update_batch_norm_params=update_batch_norm_params)
        self.eta = eta

    def update(self, nn, layer_grads, bn_grads=None):
        """Update using plain GD."""
        for i, ((W, b), (W_g, b_g)) in enumerate(zip(nn.layers, layer_grads)):
            W -= self.eta * W_g
            b -= self.eta * b_g
            nn.layers[i] = (W, b)


        # Update BatchNorm parameters if requested
        if (
            self.update_batch_norm_params
            and bn_grads is not None
            and nn.use_batch_norm
        ):
            for bn_layer, grads in zip(nn.batch_norm_layers, bn_grads):
                if not grads:
                    continue
                dgamma, dbeta = grads
                bn_layer.gamma -= self.eta * dgamma
                bn_layer.beta -= self.eta * dbeta


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        eta=0.001,
        decay_rate=0.9,
        epsilon=1e-8,
        update_batch_norm_params: bool = False,
    ):
        super().__init__(update_batch_norm_params=update_batch_norm_params)
        self.eta = eta
        self.decay_rate = decay_rate
        self.epsilon = epsilon
        self.cache = None
        self.bn_cache = None

    def update(self, nn, layer_grads, bn_grads=None):
        """Update using RMSprop."""
        if self.cache is None:
            self.cache = [
                (np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers
            ]

        for i, ((W, b), (W_g, b_g), (cache_W, cache_b)) in enumerate(
            zip(nn.layers, layer_grads, self.cache)
        ):

            # Update cache
            cache_W = self.decay_rate * cache_W + (1 - self.decay_rate) * W_g**2
            cache_b = self.decay_rate * cache_b + (1 - self.decay_rate) * b_g**2

            # Update parameters
            W -= self.eta * W_g / (np.sqrt(cache_W) + self.epsilon)
            b -= self.eta * b_g / (np.sqrt(cache_b) + self.epsilon)

            self.cache[i] = (cache_W, cache_b)
            nn.layers[i] = (W, b)

        if (
            self.update_batch_norm_params
            and bn_grads is not None
            and nn.use_batch_norm
        ):
            if self.bn_cache is None:
                self.bn_cache = [
                    (np.zeros_like(bn.gamma), np.zeros_like(bn.beta))
                    for bn in nn.batch_norm_layers
                ]
            for idx, (bn_layer, grads, (cache_gamma, cache_beta)) in enumerate(
                zip(nn.batch_norm_layers, bn_grads, self.bn_cache)
            ):
                if not grads:
                    continue
                dgamma, dbeta = grads
                cache_gamma = (
                    self.decay_rate * cache_gamma + (1 - self.decay_rate) * dgamma**2
                )
                cache_beta = (
                    self.decay_rate * cache_beta + (1 - self.decay_rate) * dbeta**2
                )
                bn_layer.gamma -= self.eta * dgamma / (
                    np.sqrt(cache_gamma) + self.epsilon
                )
                bn_layer.beta -= self.eta * dbeta / (
                    np.sqrt(cache_beta) + self.epsilon
                )
                self.bn_cache[idx] = (cache_gamma, cache_beta)

    def reset(self):
        """Reset cache for new training run."""
        self.cache = None
        self.bn_cache = None


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        eta=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        update_batch_norm_params: bool = False,
    ):
        super().__init__(update_batch_norm_params=update_batch_norm_params)
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.bn_m = None
        self.bn_v = None
        self.t = 0

    def update(self, nn, layer_grads, bn_grads=None):
        """Update using Adam."""
        if self.m is None:
            self.m = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
            self.v = [(np.zeros_like(W), np.zeros_like(b)) for W, b in nn.layers]
        if (
            self.update_batch_norm_params
            and self.bn_m is None
            and nn.use_batch_norm
        ):
            self.bn_m = [
                (np.zeros_like(bn.gamma), np.zeros_like(bn.beta))
                for bn in nn.batch_norm_layers
            ]
            self.bn_v = [
                (np.zeros_like(bn.gamma), np.zeros_like(bn.beta))
                for bn in nn.batch_norm_layers
            ]

        self.t += 1

        for i, ((W, b), (W_g, b_g), (m_W, m_b), (v_W, v_b)) in enumerate(
            zip(nn.layers, layer_grads, self.m, self.v)
        ):

            # Update moments
            m_W = self.beta1 * m_W + (1 - self.beta1) * W_g
            m_b = self.beta1 * m_b + (1 - self.beta1) * b_g
            v_W = self.beta2 * v_W + (1 - self.beta2) * W_g**2
            v_b = self.beta2 * v_b + (1 - self.beta2) * b_g**2

            # Bias correction
            m_W_hat = m_W / (1 - self.beta1**self.t)
            m_b_hat = m_b / (1 - self.beta1**self.t)
            v_W_hat = v_W / (1 - self.beta2**self.t)
            v_b_hat = v_b / (1 - self.beta2**self.t)

            # Update parameters
            W -= self.eta * m_W_hat / (np.sqrt(v_W_hat) + self.epsilon)
            b -= self.eta * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)

            self.m[i] = (m_W, m_b)
            self.v[i] = (v_W, v_b)
            nn.layers[i] = (W, b)

        if (
            self.update_batch_norm_params
            and bn_grads is not None
            and nn.use_batch_norm
        ):
            for idx, (
                bn_layer,
                grads,
                (m_gamma, m_beta),
                (v_gamma, v_beta),
            ) in enumerate(zip(nn.batch_norm_layers, bn_grads, self.bn_m, self.bn_v)):
                if not grads:
                    continue
                dgamma, dbeta = grads
                m_gamma = self.beta1 * m_gamma + (1 - self.beta1) * dgamma
                m_beta = self.beta1 * m_beta + (1 - self.beta1) * dbeta
                v_gamma = self.beta2 * v_gamma + (1 - self.beta2) * dgamma**2
                v_beta = self.beta2 * v_beta + (1 - self.beta2) * dbeta**2

                m_gamma_hat = m_gamma / (1 - self.beta1**self.t)
                m_beta_hat = m_beta / (1 - self.beta1**self.t)
                v_gamma_hat = v_gamma / (1 - self.beta2**self.t)
                v_beta_hat = v_beta / (1 - self.beta2**self.t)

                bn_layer.gamma -= self.eta * m_gamma_hat / (
                    np.sqrt(v_gamma_hat) + self.epsilon
                )
                bn_layer.beta -= self.eta * m_beta_hat / (
                    np.sqrt(v_beta_hat) + self.epsilon
                )

                self.bn_m[idx] = (m_gamma, m_beta)
                self.bn_v[idx] = (v_gamma, v_beta)

    def reset(self):
        """Reset moments and timestep for new training run."""
        self.m = None
        self.v = None
        self.bn_m = None
        self.bn_v = None
        self.t = 0
