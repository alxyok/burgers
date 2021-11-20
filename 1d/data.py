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

import logging
import tensorflow as tf

logger = logging.getLogger(__name__)

def sample_uniform(n_domain: int,
                   n_initial: int,
                   n_limit: int,
                   x_min: float = 0.,
                   x_max: float = 1.,
                   t_min: float = 0.,
                   t_max: float = 1.,
                   seed: int = 0) -> tf.Tensor:
    """Sample data from the Uniform distribution."""
    
    tf.random.set_seed(seed)

    # sample uniformly between 'x_min' and 'x_max' the x Random Variable and between 't_min' and 't_max' the t RV, and stack the vectors
    x = tf.random.uniform((n_domain, 1), dtype=tf.float64) * (x_max - x_min) + x_min
    t = tf.random.uniform((n_domain, 1), dtype=tf.float64) * (t_max - t_min) + t_min
    domain = tf.squeeze(tf.stack((x, t), axis=1))

    # do the same for the initial, at the exception that we need a 0-vector for t
    x = tf.random.uniform((n_initial, 1), dtype=tf.float64) * (x_max - x_min) + x_min
    t = tf.zeros_like(x, dtype=tf.float64)
    initial = tf.squeeze(tf.stack((x, t), axis=1))
    
    # do the same for the boundary with x = 0 and x = 1
    t = tf.random.uniform((n_limit, 1), dtype=tf.float64) * (t_max - t_min) + t_min
    x_1 = tf.ones_like(t, dtype=tf.float64) * (-1.)
    limit_1 = tf.squeeze(tf.stack((x_1, t), axis=1))
    t = tf.random.uniform((n_limit, 1), dtype=tf.float64) * (t_max - t_min) + t_min
    x1 = tf.ones_like(t, dtype=tf.float64)
    limit1 = tf.squeeze(tf.stack((x1, t), axis=1))

    # return a stack of domain and boundary
    return tf.concat((domain, initial, limit_1, limit1), axis=0)


def dataset(X: tf.Tensor, batch_size: int = 128, reshuffle: bool = True) -> tf.data.Dataset:
    """Prepare a dataset for training from initial tensor input."""

    logger.info(f'building a training dataset of batch-size: {batch_size}, reshuffle: {reshuffle}')
    ds = tf.data.Dataset.from_tensor_slices(X)
    ds = ds.shuffle(buffer_size=X.shape[0], reshuffle_each_iteration=reshuffle)
    ds = ds.batch(batch_size, drop_remainder=True)

    return ds