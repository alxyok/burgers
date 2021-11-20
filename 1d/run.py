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

import argparse
import config
import json
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

import data
import model


def main():

    parser = argparse.ArgumentParser(description='1D-Burger MLP PINN training script')
    parser.add_argument('--name', type=str, default=config.name, help='Name of the experiment')
    
    parser.add_argument('--depth', type=int, default=2, help='MLP depth')
    parser.add_argument('--units-per-layer', type=int, default=128, help='Number of units per layer')
    parser.add_argument('--activation', type=str, default='softplus', help='Activation function at each layer')
    
    parser.add_argument('--n-domain', type=int, default=2048, help='Number of samples in the domain')
    parser.add_argument('--n-ic', type=int, default=4096, help='Number of samples for the Initial Condition')
    parser.add_argument('--n-bc', type=int, default=4096, help='Number of samples for the Boundary Condition')
    
    parser.add_argument('--batch-size', type=int, default=128, help='Training input batch size')
    parser.add_argument('--max-epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=.001, help='learning rate')
    
    parser.add_argument('--seed', type=int, default=42, help='seed for random number generator')
    args = parser.parse_args()
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(logging.DEBUG)
    stdout_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(os.path.join(config.logs_path, f'run.logs'))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)


    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
    
    logger.info(f'current root directory: {config.root_path}')
    
    tf.keras.backend.set_floatx('float64')
    gpus = tf.config.list_physical_devices('GPU')
    tf.config.set_visible_devices(gpus, 'GPU')
    logger.info(f'{len(gpus)} gpu(s) currently available. Making them visible for TensorFlow.')

    
    logging.info(f'sampling X from uniform with: {args.n_domain} (domain), {args.n_ic} (ic), {args.n_bc} (bc).')
    X = data.sample_uniform(
        args.n_domain, 
        args.n_ic, 
        args.n_bc, 
        seed=args.seed)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        min_delta=1e-4,
        patience=100,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )

    # loss is well-known mean squared error
    loss = tf.keras.losses.MeanSquaredError()
    pinn = model.PINN(n_in=2, 
                      n_out=1, 
                      depth=args.depth, 
                      units_per_layer=args.units_per_layer, 
                      activation=args.activation)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr)
    pinn.compile(optimizer, loss)
    
    logger.info(f'start fitting over {args.max_epochs} epochs')
    history = pinn.fit(X,
                       epochs=args.max_epochs, 
                       callbacks=[early_stopping], )
    
    logger.info(f'saving converged model artifact.')
    pinn.save(os.path.join(config.artifacts_path, 'model.tf'))
    
    # logger.info(f'saving model architecture.')
    # with open(os.path.join(config.artifacts_path, 'model.json'), 'w') as file:
    #     json.dump(pinn.to_json(), file)
    
    logger.info(f'training done. plotting losses.')
    losses = history.history['loss']
    x = range(len(losses))
    plt.figure()
    plt.plot(x, losses)
    plt.savefig(fname=os.path.join(config.plots_path, f'plot-losses.png'), format='png')
    
    logger.info(f'training done. plotting 2D contour on entire domain.')
    n_x, n_t = 100, 100
    x, t = np.linspace(0., 1., n_x), np.linspace(0., 1., n_t)

    x_, t_ = np.meshgrid(x, t)
    X_test = np.squeeze(np.dstack((x_.flatten(), t_.flatten())))
    u = np.reshape(pinn(X_test).numpy(), x_.shape)

    plt.figure()
    plt.contourf(x_, t_, u)
    plt.colorbar()
    plt.title('u')
    plt.xlabel('x')
    plt.ylabel('t')
    plt.savefig(fname=os.path.join(config.plots_path, f'plot-2d-contour.png'), format='png')

    logger.info("u. min: {}, max: {}".format(np.amin(u), np.amax(u)))
    

if __name__ == '__main__':
    
    main()