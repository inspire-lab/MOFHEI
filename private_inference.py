import os
import argparse
import datetime
import json
from sklearn.metrics import mean_squared_error

model_configs = {
    'electrical-stability-fcnet': {
        'dataset': 'electrical-stability',
        'task': 'classification'
    },
    'mnist-lenet': {
        'dataset': 'mnist',
        'task': 'classification',
        'model_type': 'lenet'
    },
    'cifar10-alexnet': {
        'dataset': 'cifar10',
        'task': 'classification'
    },
    'cifar10-vgg16': {
        'dataset': 'cifar10',
        'task': 'classification'
    },
    'x-ray-vgg16': {
        'dataset': 'x-ray',
        'task': 'classification'
    },
    'mnist-hepex-ae1': {
        'dataset': 'mnist',
        'task': 'regression',
        'model_type': 'ae'
    },
    'mnist-hepex-ae2': {
        'dataset': 'mnist',
        'task': 'regression',
        'model_type': 'ae'
    },
    'mnist-hepex-ae3': {
        'dataset': 'mnist',
        'task': 'regression',
        'model_type': 'ae'
    },
    'cifar10-hepex-ae1': {
        'dataset': 'cifar10',
        'task': 'regression',
        'model_type': 'ae'
    },
    'cifar10-hepex-ae2': {
        'dataset': 'cifar10',
        'task': 'regression',
        'model_type': 'ae'
    },
    'cifar10-hepex-ae3': {
        'dataset': 'cifar10',
        'task': 'regression',
        'model_type': 'ae'
    },
    'cifar10-modified-lenet': {
        'dataset': 'cifar10',
        'task': 'regression',
        'model_type': 'classification'
    }
}

# crypto configs for the different models
crypto_configs = {
    'mnist-lenet': {  # depth esititmate: 19
        'poly_modulus_degree': 32768,
        'coeff_modulus': [
            40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
            30, 30, 30, 40
        ],
        'scale': 30.0,
        'multiplicative_depth': 19
    },
    'mnist-hepex-ae1': {  # depth esititmate: 4
        'poly_modulus_degree': 8192,
        'coeff_modulus': [40, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 4
    },
    'mnist-hepex-ae2': {  # depth esititmate: 4
        'poly_modulus_degree': 8192,
        'coeff_modulus': [40, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 4
    },
    'mnist-hepex-ae3': {  # depth esititmate: 10
        'poly_modulus_degree': 16384,
        'coeff_modulus': [40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 10
    },
    'cifar10-hepex-ae1': {  # depth esititmate: 4
        'poly_modulus_degree': 8192,
        'coeff_modulus': [40, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 4
    },
    'cifar10-hepex-ae2': {  # depth esititmate: 4
        'poly_modulus_degree': 8192,
        'coeff_modulus': [40, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 4
    },
    'electrical-stability-fcnet': {  # depth esititmate: 7
        'poly_modulus_degree': 16384,
        'coeff_modulus': [40, 30, 30, 30, 30, 30, 30, 30, 40],
        'scale': 30.0,
        'multiplicative_depth': 7
    },
    'cifar10-modified-lenet': {  # depth esititmate: 19
      'poly_modulus_degree': 32768,
      'coeff_modulus': [
          40, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30, 30,
          30, 30, 30, 40
      ],
      'scale': 30.0,
      'multiplicative_depth': 19
   }
}

parser = argparse.ArgumentParser(
    prog='private_inference.py',
    description='Run private inference on a vareity of models and datasets.'
    'Runs the pruned model by default. use -O to run the he-friendly'
    'pre-pruning model')

group = parser.add_argument_group('model and dataset selection')
group.add_argument('model', help='Name of model', choices=model_configs.keys())
group.add_argument(
    '-s',
    '--sparsity',
    type=int,
    default=50,
    help='Sparsity percentage. Defaults to 50. Ignored when -O is set')
group.add_argument('-O',
                   '--original',
                   action='store_true',
                   help='run he-friendly unpruned model instead')

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

parser.add_argument('-q',
                    '--quiet',
                    action='store_true',
                    help='quite tensorflow. also quites warnings and errors')
parser.add_argument('-x',
                    '--progress',
                    action='store_true',
                    help='shows progress of the computation')

experimental = parser.add_argument_group(
    'experimental features. most of these impact performance')
experimental.add_argument(
    '-c',
    '--clear_memory',
    action='store_true',
    help='can help reduce memory consompution espeacily for larger models. '
    'might impact execution time')
experimental.add_argument(
    '-p',
    '--parallel_encryption',
    action='store_true',
    help=
    'runs input encryption on multiple threads. massivley speeds up encryption'
    ' time, BUT might impact inference performance')
experimental.add_argument(
    '-l',
    '--log_memory',
    action='store_true',
    help='logs the maximum memory requirement of the private inference.'
    ' could impact performance')
experimental.add_argument(
    '-lh',
    '--log_memory_history',
    action='store_true',
    help=
    'tracks the memory consumption of the private inference, recording values '
    ' over time. implies -l. could impact performance. ')
experimental.add_argument(
    '-co',
    '--count_operations',
    action='store_true',
    help='counts the number of cipertext operations. impacts performance')

args = parser.parse_args()

model_name = args.model
data_set = model_configs[model_name]['dataset']

# create an object that will hold all our results and configs
result_dict = {}
result_dict['model'] = model_name
result_dict['dataset'] = data_set
result_dict['config'] = vars(args)

if args.quiet:
  # disable a bunch of logging
  os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if args.parallel_encryption:
  # enable parallel encryption
  os.environ['ALUMINUM_SHARK_PARALLEL_ENCRYPTION'] = '1'

import aluminum_shark.core as shark
import tensorflow as tf

if args.quiet:
  # disable a bunch of logging
  tf.compat.v1.logging.set_verbosity(50)

from main_mlsurgery import *
from mlsurgery import DynamicPolyReLU_D2, DynamicPolyReLU_D3, DynamicPolyReLU_D4
import numpy as np

import sys
import time

custom_objects = {
    'DynamicPolyReLU_D2': DynamicPolyReLU_D2,
    'DynamicPolyReLU_D3': DynamicPolyReLU_D3,
    'DynamicPolyReLU_D4': DynamicPolyReLU_D4,
    'DynamicPolyActn_D2': DynamicPolyActn_D2,
    'DynamicPolyActn_D3': DynamicPolyActn_D3,
    'DynamicPolyActn_D4': DynamicPolyActn_D4,
    'Square': Square,
    'CustomModel': CustomModel
}

verbose = args.verbose or args.extra_verbose

print('creating memory callback')
from aluminum_shark.tools.memory_logger import MemoryLogger

if args.log_memory or args.log_memory_history:
  logger = MemoryLogger(log_history=args.log_memory_history)
  logger.start()

base_dir = 'experiment_' + model_name
if args.original:
  model_file = os.path.join(base_dir, 'hefriendly', 'model.h5')
  result_dict['sparsity'] = 0
else:
  sparsity = args.sparsity
  model_file = os.path.join(base_dir, 'pruned', f'model_{sparsity}.h5')
  result_dict['sparsity'] = sparsity
result_dict['config']['model_file'] = model_file


# load model
def create_model():
  model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
  if verbose or args.model_info:
    model.summary()
  return model


result_dict['pruned'] = not args.original

# display model summary and exit
if args.model_info:
  create_model()
  exit(0)

# load data
data_file = os.path.join(base_dir, 'data', 'data.npy')

# let's have some fun with data loading
# first we need to create an `opt` object to pass to the loading functions.
# we do this by reading the relevant .sh script, take the command line arguments
# and throw them in the argparser from MLsurgery

# 1. read .sh file
with open('run_' + model_name + '.sh') as f:
  for line in f.readlines():
    if line.startswith('python'):
      break
# 2. extract arguments
mls_args = line.split()[2:]

# 3. parse the mls arguments
mls_args = fun_get_arg_parser().parse_args(mls_args)
mls_args.model_available = 'True'

# 4. create opt object
mls_opt = fun_initiate(mls_args)

if data_set == 'mnist':
  data, _ = fun_loader_mnist(
      mls_opt, experiment_model=model_configs[model_name].get('model_type'))
elif data_set == 'cifar10':
  data, _ = fun_loader_cifar10(
      mls_opt, experiment_model=model_configs[model_name].get('model_type'))
elif data_set == 'electrical-stability':
  data, _ = fun_loader_electrical_stability(mls_opt)
else:
  raise RuntimeError('unkown dataset ' + data_set)

result_dict['config']['data_file'] = data_file
x_test = data['datain_te']
if model_configs[model_name]['task'] == 'classification':
  y_test = data['dataou_te']
else:
  y_test = data['datain_te']

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

if args.count_operations:
  backend.enable_ressource_monitor(True)

# create context and keys
start = time.time()
sys.stdout.write('Creating context...')
result_dict['crypt_config'] = crypto_configs[model_name]
context = backend.createContext(scheme='ckks', **crypto_configs[model_name])
end = time.time()
result_dict['context_creation'] = end - start
print(' done. {:.2f}seconds'.format(end - start))

start = time.time()
sys.stdout.write('Generating keys...')
sys.stdout.flush()
context.create_keys()
end = time.time()
result_dict['key_creation'] = end - start
print(' done. {:.2f}seconds'.format(end - start))

# extract and encrypt a batch of data
n_slots = context.n_slots
print('running with a batchsize of:', n_slots)
result_dict['batch_size'] = end - start
x_in = x_test[:n_slots]
start = time.time()
sys.stdout.write('Encrypting data...')
sys.stdout.flush()
ctxt = context.encrypt(x_in, name='x', dtype=float, layout='batch')
end = time.time()
result_dict['encryption'] = end - start
print(' done. {:.2f}seconds'.format(end - start))

# run encrypted computation
enc_model = shark.EncryptedExecution(model_fn=create_model,
                                     context=context,
                                     clear_memory=args.clear_memory,
                                     show_progress=args.progress)
start = time.time()
sys.stdout.write('Running private inference... ')
result_ctxt = enc_model(ctxt)
end = time.time()
result_dict['private_inference'] = end - start
print('private inference took: ', end - start, 'seconds')
result_dict['max_memory'] = -1
if args.log_memory or args.log_memory_history:
  mem_log = logger.stop_and_read(unit='gb')
  result_dict['max_memory'] = mem_log['rss']
  print("Memmory requirements: ", mem_log)
  # plot memroy history if availabe
  if args.log_memory_history:
    import matplotlib.pyplot as plt
    vms_history = mem_log['vms_history']
    rms_history = mem_log['rss_history']

    t = list(range(len(vms_history)))
    plt.plot(t, vms_history, label='vms')
    plt.plot(t, rms_history, label='rms')

    plt.xlabel('Time in s')
    plt.ylabel('Memory in GB')
    plt.legend()

    plt.savefig('memory_log_' +
                datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S') + '.png')

# decrypt data
start = time.time()
sys.stdout.write('Decrypting data...')
y_pi = context.decrypt_double(result_ctxt[0])
end = time.time()
result_dict['decryption'] = end - start
print(' done. {:.2f}seconds'.format(end - start))

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
end = time.time()
result_dict['plain_inference'] = end - start
print(' done. {:.2f}seconds'.format(end - start))

if model_configs[model_name]['task'] == 'classification':
  # compute plain and encyrpted accuracy
  acc_plain = np.sum(
      np.argmax(y_plain, axis=1) == y_test[:n_slots]) / len(y_plain)
  acc_pi = np.sum(np.argmax(y_pi, axis=1) == y_test[:n_slots]) / len(y_pi)
  result_dict['plain_performance'] = acc_plain
  result_dict['encrypted_performance'] = acc_pi
  result_dict['metric'] = 'accuracy'

  print(f'encrypted accuracy {acc_pi} plain accuracy {acc_plain}')
  error = np.sum(
      np.argmax(y_plain, axis=1) != np.argmax(y_pi, axis=1)) / len(y_pi)
  print(f'error introduced by encryption: {error}')
  result_dict['encyrption_error'] = error
else:
  y_batch = x_test[:n_slots]
  mse_plain = mean_squared_error(y_batch, y_plain)
  mse_pi = mean_squared_error(y_batch, y_pi)
  result_dict['plain_performance'] = mse_plain
  result_dict['encrypted_performance'] = mse_pi
  result_dict['metric'] = 'mse'

  print(f'encrypted mse {mse_pi} plain mse {mse_plain}')

  error = mean_squared_error(y_plain, y_pi)
  print(f'error introduced by encryption: {error}')
  result_dict['encyrption_error'] = error

monitor = enc_model.monitor
history = monitor.compile_history(clear_no_ciphertext_ops=True)
result_dict.update(history)

# format it nicely and write it to file:
now = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

model_type = 'hefriendly_' if args.original else 'pruned_' + str(args.sparsity)
dict_string = json.dumps(result_dict, indent=2)

file_name = os.path.join(base_dir, 'results', model_type + '_' + now + '.json')
with open(file_name, 'w') as f:
  f.write(dict_string)
