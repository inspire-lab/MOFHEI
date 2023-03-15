import os
import sys
import subprocess
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(50)
from main_mlsurgery import *
import pandas
import json

N_RUNS = 5

SPARSITIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95]

# need those to load the model files
custom_objects = {
    'DynamicPolyActn_D2': DynamicPolyActn_D2,
    'DynamicPolyActn_D3': DynamicPolyActn_D3,
    'DynamicPolyActn_D4': DynamicPolyActn_D4,
    'Square': Square,
    'CustomModel': CustomModel
}


def parse_model(model_file):
  model = tf.keras.models.load_model(model_file, custom_objects=custom_objects)
  ret = []
  for layer in model.layers[:-1]:
    d = {'name': layer.name}
    # dense layer
    if hasattr(layer, 'units'):
      d['units'] = layer.units
      d['value'] = layer.units
    if hasattr(layer, 'filters'):
      d['filters'] = layer.filters
      d['value'] = layer.filters
    if len(d) > 1:
      ret.append(d)
  return ret


def parse_result_file(result_file):
  models = ['Pruned Model', 'HE-Friendly Model', 'Original Model']
  ret = {}
  with open(result_file, 'r') as f:
    for i, line in enumerate(f.readlines()):
      line = line.strip()
      if len(line) == 0:
        continue

      if line[:-1] in models:
        model = line[:-1]
        models.remove(model)
        ret[model] = {}
        continue

      # # code to handle legacy files with line breaks after `Final Sparsity:`
      # if 'Final Sparsity' in line:
      #   continue
      # if len(line.split(':')) == 1:
      #   key = float(line)
      #   ret['Final Sparsity'] = key
      #   continue

      key, value = line.split(':')
      key = key.strip()
      value = value.strip()
      # split of unit
      value = value.split()[0]
      # if this is false we are done with parsing acc/mse
      if len(ret[model]) < 3:
        ret[model][key] = float(value)
        continue
      ret[key] = float(value)

  # fake missing values until we have the correct file
  if 'Time HE-Friendly Model' not in ret:
    ret['Time HE-Friendly Model'] = -1
  if 'Time Pruned Model' not in ret:
    ret['Time Pruned Model'] = -1

  return ret


def parse_pi_results(file_base, dir):
  files = os.listdir(dir)
  files = [f for f in files if f.startswith(file_base)]
  if len(files) == 0:
    return False
  res = None
  for file in files:
    with open(os.path.join(dir, file), 'r') as f:
      d = json.loads(f.read())
    if res is None:
      res = d
      continue
    res['max_memory'] += d['max_memory']
    res['private_inference'] += d['private_inference']
  res['max_memory'] /= len(files)
  res['private_inference'] /= len(files)

  # extract layer info
  hlos = res['hlos']
  layers = []
  layer = {}
  for hlo in hlos:
    # start of a new layer
    if 'dot' in hlo['op'] or 'conv' in hlo['op']:
      # currently in a layer
      if len(layer) != 0:
        # close the current layer and start a new one
        layers.append(layer)
        layer = {}
      layer['type'] = 'dense' if 'dot' in hlo['op'] else 'conv'
      # merge layers
      layer = {**layer, **hlo}
      # drop before and after
      layer.pop('before')
      layer.pop('after')

      continue

    # now looking for the end of a layer. basically anything but an add concludes
    # the layer
    if 'add' not in hlo['op']:
      layers.append(layer)
      layer = {}
      continue

    # add the values from the current
    layer['time'] += hlo['time']
    layer['ctxt_ctxt_mulitplication'] += hlo['ctxt_ctxt_mulitplication']
    layer['ctxt_ptxt_mulitplication'] += hlo['ctxt_ptxt_mulitplication']
    layer['ctxt_ctxt_addition'] += hlo['ctxt_ctxt_addition']
    layer['ctxt_ptxt_addition'] += hlo['ctxt_ptxt_addition']

  # sum up total ops
  for layer in layers:
    layer['total_ctxt_ops'] = layer['ctxt_ctxt_mulitplication'] + \
                              layer['ctxt_ptxt_mulitplication'] + \
                              layer['ctxt_ctxt_addition'] + \
                              layer['ctxt_ptxt_addition']

  res['layers'] = layers
  return res


# get all available experiments
current_dir = os.listdir()

experiments = [
    f.replace('experiment_', '')
    for f in current_dir
    # if os.path.isdir(f) and f == 'experiment_mnist-hepex-ae1'
    if os.path.isdir(f) and f.startswith('experiment_')
]
experiments.sort()
print('Experinments found:', *experiments)

data_frames = []

for exp in experiments:
  exp_dir = 'experiment_' + exp
  print(exp)
  # original model
  model_file = os.path.join(exp_dir, 'original', 'model.h5')
  if not os.path.exists(model_file):
    print(model_file, 'not found skipping', exp)
    continue
  layers = parse_model(model_file)
  # create dataframes
  columns = ['Model', 'Sparsity']
  for i, l in enumerate(layers):
    columns.append(l['name'] + ' units' if 'units' in l else 'filters')
    columns.append('Reduction' + str(i))
    columns.append('HE ops' + str(i))

  columns.append('MSE')
  columns.append('Time pruning')
  columns.append('Time HE-friendly')
  columns.append('Time PI')
  columns.append('Total HE operations')
  columns.append('Memory PI')

  dict = {k: [] for k in columns}
  # add original model to the dict
  dict['Model'].append('Original')
  dict['Sparsity'].append('-')
  for i, l in enumerate(layers):
    dict[columns[(i * 3) + 2]].append(l['value'])
    dict['Reduction' + str(i)].append('-')
    dict['HE ops' + str(i)].append('-')

  # dict['MSE'].append()  #we cheat here and do that later
  dict['Time pruning'].append('-')
  dict['Time HE-friendly'].append('-')
  dict['Time PI'].append('-')
  dict['Total HE operations'].append('-')
  dict['Memory PI'].append('-')

  # start parsing the result files
  do_he_friendly = True
  for sparsity in SPARSITIES:

    result_file = os.path.join(exp_dir, 'results', f'results_{sparsity}.txt')
    if not os.path.exists(result_file):
      print(result_file, 'not found. skipping sparsity:', sparsity)
      continue
    results = parse_result_file(result_file)
    # determine the metric used
    metric = list(results['Original Model'].keys())[0].split()[0]

    # check if want to add the values for the he friendly and original model
    if not parse_pi_results('hefriendly', os.path.join(exp_dir, 'results')):
      print(
          f'hefriendly not found. in {os.path.join(exp_dir, "results")} skipping'
      )
    if do_he_friendly and parse_pi_results('hefriendly',
                                           os.path.join(exp_dir, 'results')):
      do_he_friendly = False
      # original model
      dict['MSE'].append(results['Original Model'][f'{metric} Test'])
      # add the HE friendly model

      # for the HE friendly model we need some values from the pi results files
      pi_results = parse_pi_results('hefriendly',
                                    os.path.join(exp_dir, 'results'))

      dict['Model'].append('HE-Friendly')
      dict['Sparsity'].append('-')
      for i, l in enumerate(layers):
        key = columns[(i * 3) + 2]
        pi_layer = pi_results['layers'][i]
        assert pi_layer[
            'type'] in key, f'layer mismatch expected: {key} got {pi_layer["type"]}'
        dict[key].append(l['value'])
        dict['Reduction' + str(i)].append('-')
        dict['HE ops' + str(i)].append(int(pi_layer['total_ctxt_ops']))

      dict['MSE'].append(results['HE-Friendly Model'][f'{metric} Test'])
      dict['Time pruning'].append('-')
      dict['Time HE-friendly'].append(results['Time HE-Friendly Model'])
      dict['Time PI'].append(pi_results['private_inference'])
      dict['Total HE operations'].append(
          int(pi_results['total_ciphertext_operations']))
      dict['Memory PI'].append(pi_results['max_memory'])

    # do the proper pruned files
    pi_results = parse_pi_results(f'pruned_{sparsity}',
                                  os.path.join(exp_dir, 'results'))
    if not pi_results or not os.path.exists(
        os.path.join(exp_dir, 'pruned', f'model_{sparsity}.h5')):
      if not pi_results:
        print(
            f'pruned_{sparsity} not found. in {os.path.join(exp_dir, "results")} skipping'
        )
      if not os.path.exists(
          os.path.join(exp_dir, 'pruned', f'model_{sparsity}.h5')):
        print(os.path.join(exp_dir, 'pruned', f'model_{sparsity}.h5'),
              'not found skipping')
      continue
    layers = parse_model(os.path.join(exp_dir, 'pruned',
                                      f'model_{sparsity}.h5'))

    dict['Model'].append(f'Pruned {sparsity}%')
    dict['Sparsity'].append(results['Final Sparsity'])
    for i, l in enumerate(layers):
      value = l['value']
      index = (i * 3) + 2
      reduction = dict[columns[index]][0] / value
      key = columns[index]
      pi_layer = pi_results['layers'][i]
      assert pi_layer[
          'type'] in key, f'layer mismatch expected: {key} got {pi_layer["type"]}'
      dict[columns[index]].append(value)
      dict['Reduction' + str(i)].append(reduction)
      dict['HE ops' + str(i)].append(int(pi_layer['total_ctxt_ops']))

    dict['MSE'].append(results['Pruned Model'][f'{metric} Test'])
    dict['Time pruning'].append(results['Time Pruned Model'])
    dict['Time HE-friendly'].append('-')
    dict['Time PI'].append(pi_results['private_inference'])
    dict['Total HE operations'].append(
        int(pi_results['total_ciphertext_operations']))
    dict['Memory PI'].append(pi_results['max_memory'])

  try:
    df = pandas.DataFrame(dict)
  except Exception as e:
    # print(e)
    # print(exp, 'dataframe messed up. not creating a table')
    # # print(json.dumps(dict, indent=2))
    # for key, value in dict.items():
    #   print(key, len(value))
    continue
  # now that we have all the data in a data frame we can do fun things with

  df = df.rename(columns={
      "MSE": metric.upper(),
  })

  data_frames.append((exp, df.copy()))

  # print(df)
  df = df.rename(
      columns={
          "Time pruning": "TP",
          "Time HE-friendly": "THEF",
          'Total HE operations': 'HE ops'
      })
  df.to_latex(
      os.path.join('tables', f'{exp}.tex'),
      float_format="{:0.2f}".format,
      index=False,
      caption=
      f'Results for {exp}  Time pruning (TP), Time HE-friendly (THEF), Total HE operations (HE ops)'
  )

  # first we create tables
  df = df.set_index('Model')
  df_t = df.T
  # print(df_t)
  # shorten and replace some names
  columns = df_t.columns
  columns = {c: c.replace('Pruned ', '') for c in columns}
  df_t.rename(columns=columns, inplace=True)
  df_t.rename(columns={'HE-Friendly': 'HEF'}, inplace=True)

  # print(df_t.index)
  # rebuild index
  n_dense = 0
  n_conv = 0
  d = {}
  for i in df_t.index:
    if 'conv' in i.lower():
      d[i] = 'Conv ' + str(n_conv)
      n_conv += 1
    if 'dense' in i.lower():
      d[i] = 'Dense ' + str(n_dense)
      n_dense += 1

  df_t.rename(index=d, inplace=True)

  # combine the layer , reduction, and he ops
  delete_me = []
  for idx in df_t.index:
    if 'conv' in idx.lower() or 'dense' in idx.lower():
      # get the Reductioni and HE opsi row
      i = idx.split()[1]
      row = df_t.loc[idx]
      red_row = df_t.loc['Reduction' + i]
      heo_row = df_t.loc['HE ops' + i]

      # build the new row
      def float_format(x):
        if isinstance(x, str):
          return x
        return f'{x:.1f}'

      new_row = [
          f'{r} / {float_format(red)} / {he}'
          for r, red, he in zip(row, red_row, heo_row)
      ]
      # update row
      df_t.loc[idx] = new_row
      # flag for deletion
      delete_me.append('Reduction' + i)
      delete_me.append('HE ops' + i)

  df_t.drop(delete_me, inplace=True)

  df_t.to_latex(
      os.path.join('tables', f'{exp}_T_.tex'),
      float_format="{:0.2f}".format,
      caption=
      f'Results for {exp}  Time pruning (TP), Time HE-friendly (THEF), Total HE operations (HE ops), HE-Friendly (HEF). For Dense and Conv layers the table shows # units or filters / reduction factor / HE operations',
      label=f'tab:{exp}_results')

# generate figures
import matplotlib.pyplot as plt

# private infernce time vs sparsity
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Time PI'])[1:]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Private Inference in s')
plt.savefig(os.path.join('figures', 'pi_sparsity.pdf'))
plt.clf()

# private infernce time reduction % vs sparsity
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Time PI'])[1:]
  y = [x / y[0] for x in y]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Private inference time reduction')
plt.savefig(os.path.join('figures', 'pi_reduction_sparsity.pdf'))
plt.clf()

# sparsity vs pruning time
x = SPARSITIES
for exp, df in data_frames:
  y = list(df['Time pruning'])[2:]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Time Pruning in s')
plt.savefig(os.path.join('figures', 'time_pruning_sparsity.pdf'))
plt.clf()

# sparsity vs he-ops
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Total HE operations'])[1:]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Total HE operations')
plt.savefig(os.path.join('figures', 'total_heops_sparsity.pdf'))
plt.clf()

# sparsity vs he-ops reduction
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Total HE operations'])[1:]
  y = [x / y[0] for x in y]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Total HE operations reduction')
plt.savefig(os.path.join('figures', 'total_heops_reduction_sparsity.pdf'))
plt.clf()

# sparsity vs memory
# 'Memory PI'
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Memory PI'])[1:]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Memory requirement private inference in GB')
plt.savefig(os.path.join('figures', 'memory_sparsity.pdf'))
plt.clf()

# sparsity vs memory_reduction
# 'Memory PI'
x = [0] + SPARSITIES
for exp, df in data_frames:
  y = list(df['Memory PI'])[1:]
  y = [x / y[0] for x in y]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Reduction in memory for private inference')
plt.savefig(os.path.join('figures', 'memory_reduction_sparsity.pdf'))
plt.clf()

# sparsity vs performance
x = [0] + SPARSITIES
for exp, df in data_frames:
  mse = 'MSE' in df.columns
  print('is MSE', mse)
  if mse:
    y = list(df['MSE'])
    baseline = y[0]
    y = [baseline / x for x in y[1:]]
  else:
    y = list(df['ACC'])
    baseline = y[0]
    y = [x / baseline for x in y[1:]]
  plt.plot(x, y, 'o-', label=exp)
plt.legend()
plt.xlabel('Sparsity in %')
plt.ylabel('Perfromance compared to baseline')
plt.savefig(os.path.join('figures', 'performance_sparsity.pdf'))
plt.clf()
