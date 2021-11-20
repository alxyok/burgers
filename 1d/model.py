# MIT License

# Copyright (c) 2021 alxyok

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf


def _u0(X: tf.Tensor, 
        mu: float = 0.5, 
        sigma2: float = 1/200, 
        K: float = np.sqrt(np.pi)/20) -> tf.Tensor:
    """Computes the IC, which is a Gaussian. mu is the mean, sigma2, the deviation and K is derived from the curve's height peak. Returns the Gaussian at X."""

    return K * tf.exp(-.5 * tf.square(X[:, 0:1] - mu) / sigma2) / np.sqrt(2 * np.pi * sigma2)


class PINN(tf.keras.Model):
    
    def __init__(self, 
                 n_in: int, 
                 n_out: int, 
                 depth: int, 
                 units_per_layer: int, 
                 activation: str, 
                 inviscid: bool = False,
                 nu: float = .01/np.pi,
                 name: str = 'pinn',
                 dtype: tf.dtypes.DType = tf.float64, 
                 **kwargs):
        """Physics-Informed Neural Network. Once trained, returns the solution for the 1D formulation of Burger's equation. Implements the training logic for an easy use with Keras' API, including tf.keras.Model.fit(). This implementation is a simple straightforward MLP, because it's sufficient for the use-case.
        
        Arguments:
            n_int (int): number of input features.
            n_out (int): number of output features.
            depth (int): number of layers in the MLP.
            units_per_layer (int): number of neurons per layer.
            activation (str): activation function to use at every layer.
            inviscid (bool): If True, solves the inviscid, first-order 1D Burger. Defaults to False.
            nu (float): mean to use for IC's Gaussian.
            name (str): name of the neural model.
            dtype (tf.dtypes.DType): tensorflow type of tensor to use throughout this implementation."""
        
        super().__init__(name=name, **kwargs)
        self._depth = depth
        self._inviscid = inviscid
        self._nu = nu
        
        self._input_layer = tf.keras.layers.InputLayer(input_shape=(n_in,), dtype=dtype)
        
        self._hidden_layers = [
            tf.keras.layers.Dense(
                units=units_per_layer, 
                activation=activation,
                dtype=dtype
            ) for _ in range(self._depth)]
        
        self._output_layer = tf.keras.layers.Dense(
            units=n_out, 
            activation=activation,
            dtype=dtype
        )
        
        
    def call(self, inputs: tf.Tensor) -> tf.Tensor:
        """PINN's forward pass."""
        
        x = self._input_layer(inputs)
        
        for idx in range(self._depth):
            x = self._hidden_layers[idx](x)
            
        x = self._output_layer(x)
        
        return x

    
    def train_step(self, data: tf.Tensor) -> dict:
        """PINN's training logic. Computes the residuals for IC, BC, and in the domain, and optimizes for the sum of individual losses."""
        
        X = data

        with tf.GradientTape() as tape:
            # BC residuals ==========
            Xlim_condition = tf.concat((tf.logical_or(
                X[:, 0:1] == 1., 
                X[:, 0:1] == 0.),) * 2, axis=1)
            Xlim = tf.reshape(tf.boolean_mask(X, Xlim_condition), (-1, 2))
            ylim = self(Xlim)
            ulim_residuals = self.compiled_loss(ylim, tf.zeros_like(ylim))
            # =======================
            
            # IC residuals ==========
            X0_condition = tf.concat((X[:, 1:2] == 0.,) * 2, axis=1)
            X0 = tf.reshape(tf.boolean_mask(X, X0_condition), (-1, 2))
            u0_residuals = self.compiled_loss(self(X0), _u0(X0))
            # =======================
            
            # domain residuals ======
            Xd_condition = tf.logical_not(tf.logical_or(Xlim_condition, X0_condition))
            Xd = tf.reshape(tf.boolean_mask(X, Xd_condition), (-1, 2))
            
            with tf.GradientTape() as second_order:
                second_order.watch(Xd)
                with tf.GradientTape() as first_order:
                    first_order.watch(Xd)
                    u = self(Xd)
                u_X = first_order.gradient(u, Xd)
            u_XX = second_order.gradient(u_X, Xd)

            u_t = u_X[:, 1:2]
            u_x = u_X[:, 0:1]
            u_xx = u_XX[:, 0:1]

            if self._inviscid:
                self._nu = 0

            yd = u_t + u * u_x - self._nu * u_xx
            pde_residuals = self.compiled_loss(yd, tf.zeros_like(yd))
            # =======================
            
            loss = tf.reduce_sum(tf.stack((ulim_residuals, u0_residuals, pde_residuals)))
        gradients = tape.gradient(loss, self.trainable_variables)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(tf.zeros_like(loss), loss)

        return {m.name: m.result() for m in self.metrics}