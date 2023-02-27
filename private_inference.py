import aluminum_shark.core as shark
import tensorflow as tf
from mlsurgery import *
import numpy as np
import argparse

import sys
import time

custom_objects = {
    'DynamicPolyReLU_D2': DynamicPolyReLU_D2,
    'DynamicPolyReLU_D3': DynamicPolyReLU_D3,
    'DynamicPolyReLU_D4': DynamicPolyReLU_D4,
    'Square': Square,
    'CustomModel': CustomModel
}

# crypto configs for the different models
crypto_configs = {
    'mnist': {  # depth esititmate: 8
        'poly_modulus_degree': 16384,
        'coeff_modulus': [30, 23, 23, 23, 23, 23, 23, 23, 23, 30],
        'scale': 23.0,
        'multiplicative_depth': 8
    },
    'cifar10': {  # depth esititmate: 25
        'poly_modulus_degree': 32768,
        'coeff_modulus': [
            40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
            30, 30, 30, 30, 30, 30, 30, 30, 30, 40
        ],
        'scale': 30.0,
        'multiplicative_depth': 25
    },
    'electric_grid_stability': {  # depth esititmate: 10
        'poly_modulus_degree': 16384,
        'coeff_modulus': [40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 10
    }
}

parser = argparse.ArgumentParser(
    prog='private_inference.py',
    description='Run private inference on a vareity of models and datasets')

parser.add_argument('dataset',
                    help='Name of dataset and corresponding model',
                    choices=crypto_configs.keys())
parser.add_argument('-v',
                    '--verbose',
                    action='store_true',
                    help='prints runtime information. impacts performance')
parser.add_argument(
    '-vv',
    '--extra_verbose',
    action='store_true',
    help='prints debug runtime information. greatly impacts performance')
parser.add_argument('-i',
                    '--model_info',
                    action='store_true',
                    help='only display model info and architecture')
parser.add_argument('-O',
                    '--original',
                    action='store_true',
                    help='run the orginal unpruned model instead')

experimental = parser.add_argument_group('experimental features')
experimental.add_argument(
    '-c',
    '--clear_memory',
    action='store_true',
    help=
    'can help reduce memory consompution espeacily for larger models. might impact execution time'
)

args = parser.parse_args()
data_set = args.dataset
verbose = args.verbose or args.extra_verbose

_, _, x_test, y_test = fun_data(name=data_set, calibrate=True)


# load model
def create_model():
  if (args.original):
    model = tf.keras.models.load_model(f'model_original_{data_set}.h5',
                                       custom_objects=custom_objects)
  else:
    model = tf.keras.models.load_model(f'model_culled_{data_set}.h5',
                                       custom_objects=custom_objects)
  if verbose or args.model_info:
    model.summary()
  return model


# display model summary and exit
if args.model_info:
  create_model()
  exit(0)

# enable logging
if verbose:
  shark.enable_logging(True)
  if args.extra_verbose:
    shark.set_log_level(shark.DEBUG)
  else:
    shark.set_log_level(shark.INFO)

# load SEAL backend
backend = shark.HEBackend()
# enable backend logging
if verbose:
  if args.extra_verbose:
    backend.set_log_level(shark.DEBUG)
  else:
    backend.set_log_level(shark.INFO)

# create context and keys
start = time.time()
sys.stdout.write('Creating context...')
context = backend.createContext(scheme='ckks', **crypto_configs[data_set])
print(' done. {:.2f}seconds'.format(time.time() - start))

start = time.time()
sys.stdout.write('Generating keys...')
context.create_keys()
print(' done. {:.2f}seconds'.format(time.time() - start))

# extract and encrypt a batch of data
n_slots = context.n_slots
print('running with a batchsize of:', n_slots)
x_in = x_test[:n_slots]
start = time.time()
sys.stdout.write('Encrypting data...')
ctxt = context.encrypt(x_in, name='x', dtype=float, layout='batch')
print(' done. {:.2f}seconds'.format(time.time() - start))

# run encrypted computation
enc_model = shark.EncryptedExecution(model_fn=create_model,
                                     context=context,
                                     clear_memory=args.clear_memory)
start = time.time()
sys.stdout.write('Running private inference... ')
result_ctxt = enc_model(ctxt)
end = time.time()
print('private inference took: ', end - start, 'seconds')
# decrypt data
start = time.time()
sys.stdout.write('Decrypting data...')
y_pi = context.decrypt_double(result_ctxt[0])
print(' done. {:.2f}seconds'.format(time.time() - start))
print(y_pi.shape, y_test.shape)

# bring the output data into the the correct form. we assume that this
# calssifaciotn and y contains labels
if len(y_test.shape) > 1:
  if y_test.shape[1] > 1:
    y_test = np.argmax(y_test, axis=1)
  else:
    y_test = y_test.reshape(-1)

# run on plain data for comparison
model = create_model()
start = time.time()
sys.stdout.write('Running plain model...')
y_plain = model(x_in)
print(' done. {:.2f}seconds'.format(time.time() - start))

# compute plain and encyrpted accuracy
acc_plain = np.sum(
    np.argmax(y_plain, axis=1) == y_test[:n_slots]) / len(y_plain)
acc_pi = np.sum(np.argmax(y_pi, axis=1) == y_test[:n_slots]) / len(y_pi)

print(f'encrypted accuracy {acc_pi} plain accuracy {acc_plain}')
error = np.sum(
    np.argmax(y_plain, axis=1) != np.argmax(y_pi, axis=1)) / len(y_pi)
print(f'error introduced by encryption: {error}')
