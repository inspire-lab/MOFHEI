import os
import sys
import subprocess
import time
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(50)
from main_mlsurgery import *
import pandas
import json

translator = {
    'mnist-hepex-ae1': 'MNIST-AE1',
    'mnist-hepex-ae2': 'MNIST-AE2',
    'mnist-hepex-ae3': 'MNIST-AE3',
    'cifar10-modified-lenet': 'CIFAR-10-MLeNet',
    'electrical-stability-fcnet': 'EGSS-FcNet',
    'mnist-lenet': 'MNIST-LeNet',
    'x-ray-modified-lenet': 'X-Ray-LeNet'
}

markers = {
    'mnist-hepex-ae1': 'vb--',
    'mnist-hepex-ae2': '^g--',
    'mnist-hepex-ae3': '<r--',
    'cifar10-modified-lenet': 'oc-',
    'electrical-stability-fcnet': '*m-',
    'mnist-lenet': 'Py-',
    'x-ray-modified-lenet': 'xk-'
}
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

  max_mem = []
  private_inference = []
  encryption_error = []
  for file in files:
    with open(os.path.join(dir, file), 'r') as f:
      d = json.loads(f.read())
    max_mem.append(d['max_memory'])
    private_inference.append(d['private_inference'])
    encryption_error.append(d['encyrption_error'])
    if res is None:
      res = d
      continue
    res['max_memory'] += d['max_memory']
    res['private_inference'] += d['private_inference']
    res['encyrption_error'] += d['encyrption_error']
  res['max_memory'] /= len(files)
  res['private_inference'] /= len(files)
  res['encyrption_error'] /= len(files)

  # print(dir, file_base)
  # print('\tmax_memory std:', np.std(max_mem), 'var:', np.var(max_mem))
  # print('\tprivate_inference std:', np.std(private_inference), 'var:',
  #       np.var(private_inference))
  # print('\tencryption_error std:', np.std(encryption_error), 'var:',
  #       np.var(encryption_error))

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

    # currently not in a layer. could be an activation function or somethiong
    if len(layer) == 0:
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


# # get all available experiments
# current_dir = os.listdir()

# experiments = [
#     f.replace('experiment_', '')
#     for f in current_dir
#     # if os.path.isdir(f) and f == 'experiment_mnist-hepex-ae1'
#     if os.path.isdir(f) and f.startswith('experiment_')
# ]
# experiments.sort()
# print('Experinments found:', *experiments)

experiments = [
    'mnist-lenet',
    'x-ray-modified-lenet',
    'cifar10-modified-lenet',
    'electrical-stability-fcnet',
    'mnist-hepex-ae1',
    'mnist-hepex-ae2',
    'mnist-hepex-ae3',
]

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
    if 'units' in l:
      columns.append(l['name'] + ' units')
    else:
      columns.append(l['name'] + ' filters')
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
    if do_he_friendly:
      do_he_friendly = False
      # original model
      dict['MSE'].append(results['Original Model'][f'{metric} Test'])
      # add the HE friendly model

      # for the HE friendly model we need some values from the pi results files
      pi_results = parse_pi_results('hefriendly',
                                    os.path.join(exp_dir, 'results'))
      if not pi_results:
        dict['Model'].append('HE-Friendly')
        dict['Sparsity'].append('-')
        for i, l in enumerate(layers):
          key = columns[(i * 3) + 2]
          dict[key].append(l['value'])
          dict['Reduction' + str(i)].append('-')
          dict['HE ops' + str(i)].append('N/A')

        dict['MSE'].append(results['HE-Friendly Model'][f'{metric} Test'])
        dict['Time pruning'].append('-')
        dict['Time HE-friendly'].append(results['Time HE-Friendly Model'])
        dict['Time PI'].append('N/A')
        dict['Total HE operations'].append('N/A')
        dict['Memory PI'].append('N/A')

      else:
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

  with open(os.path.join('tables', f'{exp}.tex'), 'r+') as f:
    # reaplce table
    text = f.read().replace("\\begin{table}", "\\begin{table*}")
    text = text.replace("\\end{table}", "\\end{table*}")
    # add resize box
    text = text.replace("\\begin{tabular}",
                        "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
    text = text.replace("\\end{tabular}", "\\end{tabular}}")

    f.seek(0)
    f.write(text)

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
  count = 0

  def float_format(x):
    if isinstance(x, str):
      return x
    return f'{x:.1f}'

  def to_scienttific(x):
    try:
      x = int(x)
    except:
      return x
    digits = int(np.log10(x))
    if digits < 4:
      return x
    x /= 10**digits
    return (float_format(x) + 'e' + str(digits))

  for idx in df_t.index:
    if 'conv' in idx.lower() or 'dense' in idx.lower():
      # get the Reductioni and HE opsi row

      # i = idx.split()[1]
      row = df_t.loc[idx]
      red_row = df_t.loc['Reduction' + str(count)]
      heo_row = df_t.loc['HE ops' + str(count)]

      # build the new row
      new_row = [
          f'{r} / {float_format(red)} / {to_scienttific(he)}'
          for r, red, he in zip(row, red_row, heo_row)
      ]
      # update row
      df_t.loc[idx] = new_row
      # flag for deletion
      delete_me.append('Reduction' + str(count))
      delete_me.append('HE ops' + str(count))
      count += 1

  df_t.drop(delete_me, inplace=True)
  # delete the original comlumn
  df_t.drop('Original', axis='columns', inplace=True)
  # delete the Time pruning comlumn
  df_t.drop('THEF', axis='rows', inplace=True)

  # reformat some rows
  def to_int(x):
    if isinstance(x, float):
      return round(x)
    else:
      return x

  df_t.loc['HE ops'] = [to_scienttific(x) for x in df_t.loc['TP']]
  df_t.loc['TP'] = [to_int(x) for x in df_t.loc['TP']]
  df_t.loc['Memory PI'] = [to_int(x) for x in df_t.loc['Memory PI']]

  # rename first columns
  df_t.rename(index={
      'Time PI': 'TPI',
      'HE ops': 'HEO',
      'Memory PI': 'MPI'
  },
              inplace=True)

  if metric == 'ACC':
    metric_string = 'accuracy (ACC)'
  else:
    metric_string = 'mean squared error (MSE)'

  df_t.to_latex(os.path.join('tables', f'{exp}_T_.tex'),
                float_format="{:0.2f}".format,
                caption=f"""Results for the HE-friendly (HEF) {translator[exp]} 
                models and different layer-wise sparsities (columns). The table
                shows the final Sparsity of the model, the prunable layers and 
                their units (filters for conv) / reduction factor / number HE 
                operations, {metric}, time needed to prune the model TP,
                private inference latency (TPI), total number of HE operations 
                (HEO), and max memory required to perform private inference (MPI)
                """,
                label=f'tab:{exp}_results')

  with open(os.path.join('tables', f'{exp}_T_.tex'), 'r+') as f:
    # reaplce table
    text = f.read().replace("\\begin{table}", "\\begin{table*}")
    text = text.replace("\\end{table}", "\\end{table*}")
    # add resize box
    text = text.replace("\\begin{tabular}",
                        "\\resizebox{\\textwidth}{!}{\\begin{tabular}")
    text = text.replace("\\end{tabular}", "\\end{tabular}}")
    text = text.replace("Model", '')

    f.seek(0)
    f.write(text)

# stick all tables in one file
files = os.listdir('tables')

# not transposed tables
non_t_tables = []
# transposed tables
t_tables = []
for file in files:
  if not file.endswith('.tex'):
    continue
  cont = False
  for exp in experiments:
    cont = cont or exp in file
  if not cont:
    continue
  with open(os.path.join('tables', file)) as f:
    if '_T_' in file:
      t_tables.append(f.read())
    else:
      non_t_tables.append(f.read())

with open('tables/all_tables.tex', 'w') as f:
  f.write('\n\n\n'.join(non_t_tables))

with open('tables/all_T_tables.tex', 'w') as f:
  f.write('\n\n\n'.join(t_tables))

# generate table that compares the origianl models and the
d = {
    'Model': [],
    'Metric': [],
    'Org.': [],
    'HEf': [],
    'Change': [],
    'HEf (s)': []
}
for exp, df in data_frames:
  d['Model'].append(translator[exp].replace('-modified-lenet',
                                            '').replace('electrical-stability',
                                                        'EGSS'))
  if 'ACC' in df.columns:
    d['Metric'].append('Acc.')
    metric = 'ACC'
  else:
    d['Metric'].append('MSE')
    metric = 'MSE'
  # get the metric for the original model and hef model
  metric_org = df.iloc[0][metric]
  metric_hef = df.iloc[1][metric]
  change = abs(metric_org - metric_hef) / metric_org

  d['Org.'].append(metric_org)
  d['HEf'].append(metric_hef)
  d['Change'].append(change)
  d['HEf (s)'].append(df.iloc[1]['Time HE-friendly'])

# create a dataframe and latex table
df = pandas.DataFrame(d)

tex_string = df.to_latex(os.path.join('tables', f'org_vs_hef.tex'),
                         float_format="{:0.2f}".format,
                         index=False,
                         caption=f"""
      Comparison of the original model (Org.) and the HE-friendly model (HEf) in
      terms of: performance metric (either Accuracy (Acc.) or Mean squared Error
      (MSE)), the relative change in the metric, and the time to transform the 
      original model into an HE-friendly one in seconds (HEf (s)). 
      """,
                         label=f'tab:org_vs_hef')

# generate figures
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def clean_plot(x, y):
  _x = []
  _y = []
  for v0, v1 in zip(x, y):
    try:
      v1 = float(v1)
      _x.append(v0)
      _y.append(v1)
    except:
      pass
  return _x, _y


label_font_size = 'xx-large'
label_fontweight = 'bold'
tick_size = 'xx-large'


# private infernce time vs sparsity
def pi_sparsity(fig, ax):
  for exp, df in data_frames:
    x = [0] + SPARSITIES
    y = list(df['Time PI'])[1:]
    _x, _y = clean_plot(x, y)
    ax.plot(_x, _y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference In Seconds',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


pi_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'pi_sparsity.pdf'))
plt.clf()


# log y
def pi_sparsity_log(fig, ax):
  for exp, df in data_frames:
    x = [0] + SPARSITIES
    y = list(df['Time PI'])[1:]
    _x, _y = clean_plot(x, y)
    _y = np.log10(_y).astype(float)
    ax.plot(_x, _y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference In $log_{10}$ Seconds',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


pi_sparsity_log(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'pi_sparsity_log.pdf'))
plt.clf()


# private infernce time reduction % vs sparsity
def pi_reduction_sparsity(fig, ax):
  for exp, df in data_frames:
    x = [0] + SPARSITIES
    y = list(df['Time PI'])[1:]
    x, y = clean_plot(x, y)
    y = [x / y[0] for x in y]
    ax.plot(x, y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference Time Reduction',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


pi_reduction_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'pi_reduction_sparsity.pdf'))
plt.clf()


# sparsity vs pruning time
def time_pruning_sparsity(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Time pruning'])[2:]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Pruning Time In Seconds',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


time_pruning_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'time_pruning_sparsity.pdf'))
plt.clf()


#log
def time_pruning_sparsity_log(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Time pruning'])[2:]
    _x, _y = clean_plot(x, y)
    _y = np.log10(_y).astype(float)
    ax.plot(_x, _y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Pruning Time in $log_{10}$ Seconds',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


time_pruning_sparsity_log(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'time_pruning_sparsity_log.pdf'))
plt.clf()


# sparsity vs he-ops
def total_heops_sparsity(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Total HE operations'])[1:]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Total HE Operations',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


total_heops_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'total_heops_sparsity.pdf'))
plt.clf()


# log
def total_heops_sparsity_log(fig, ax):
  for exp, df in data_frames:
    print(exp)
    x = SPARSITIES
    y = list(df['Total HE operations'])[1:]
    _x, _y = clean_plot(x, y)
    _y = np.log10(_y).astype(float)
    ax.plot(_x, _y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Total HE Operations $log_{10}$',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


total_heops_sparsity_log(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'total_heops_sparsity_log.pdf'))
plt.clf()


# sparsity vs he-ops reduction
def total_heops_reduction_sparsity(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Total HE operations'])[1:]
    x, y = clean_plot(x, y)
    y = [x / y[0] for x in y]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Total HE Operations Reduction',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


total_heops_reduction_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'total_heops_reduction_sparsity.pdf'))
plt.clf()


# sparsity vs memory
# 'Memory PI'
def memory_sparsity(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Memory PI'])[1:]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference Memory In GB',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


memory_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'memory_sparsity.pdf'))
plt.clf()


#loglegend
def memory_sparsity_log(fig, ax):
  for exp, df in data_frames:
    x = [0] + SPARSITIES
    y = list(df['Memory PI'])[1:]
    _x, _y = clean_plot(x, y)
    _y = np.log10(_y).astype(float)
    ax.plot(_x, _y, markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference Memory In $log_{10}$ GB ',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


memory_sparsity_log(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'memory_sparsity_log.pdf'))
plt.clf()


# sparsity vs memory_reduction
# 'Memory PI'
def memory_reduction_sparsity(fig, ax):
  for exp, df in data_frames:
    x = SPARSITIES
    y = list(df['Memory PI'])[1:]
    x, y = clean_plot(x, y)
    y = [x / y[0] for x in y]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Private Inference Memory Reduction',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


memory_reduction_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'memory_reduction_sparsity.pdf'))
plt.clf()


# sparsity vs performance
def performance_sparsity(fig, ax):
  x = [0] + SPARSITIES
  for exp, df in data_frames:
    mse = 'MSE' in df.columns
    if mse:
      y = list(df['MSE'])
      x, y = clean_plot(x, y)
      baseline = y[0]
      y = [baseline / x for x in y[1:]]
    else:
      y = list(df['ACC'])
      x, y = clean_plot(x, y)
      baseline = y[0]
      y = [x / baseline for x in y[1:]]
    ax.plot(*clean_plot(x, y), markers[exp], label=translator[exp])
  ax.set_xlabel('Layer-wise Sparsity In %',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.set_ylabel('Perfromance compared to baseline',
                fontsize=label_font_size,
                fontweight=label_fontweight)
  ax.tick_params(axis='x', labelsize=tick_size)
  ax.tick_params(axis='y', labelsize=tick_size)
  ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


performance_sparsity(plt.gcf(), plt.gca())
# plt.gca().legend()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'performance_sparsity.pdf'))
plt.clf()

legend_handles = [
    plt.gca().plot([1], [1], markers[exp], label=translator[exp])[0]
    for exp, _ in data_frames
]
plt.clf()

plt.gca().axis('off')
plt.gca().legend(handles=legend_handles, loc='center', prop={'size': 'x-large'})
plt.tight_layout()
plt.tight_layout()
plt.savefig(os.path.join('figures', 'one_legend_to_rule_them_all.pdf'))

# # create on big figure
scale = 3.
fig, axs = plt.subplots(
    nrows=2,
    ncols=4,
    # sharex=True,
    figsize=[6.4 * scale, 4.8 * scale * 0.5])
axs = axs.reshape(-1)
pi_sparsity(fig, axs[0])
pi_reduction_sparsity(fig, axs[1])
# time_pruning_sparsity(fig, axs[2])
total_heops_sparsity(fig, axs[2])
total_heops_reduction_sparsity(fig, axs[3])
memory_sparsity(fig, axs[4])
memory_reduction_sparsity(fig, axs[5])
performance_sparsity(fig, axs[6])
axs[7].axis('off')
axs[7].legend(handles=legend_handles, loc='center')
fig.tight_layout()
fig.savefig(os.path.join('figures', 'all.pdf'))

# log scale
fig, axs = plt.subplots(
    nrows=2,
    ncols=4,
    # sharex=True,
    figsize=[6.4 * scale, 4.8 * scale * 0.5])
axs = axs.reshape(-1)
pi_sparsity_log(fig, axs[0])
pi_reduction_sparsity(fig, axs[1])
time_pruning_sparsity_log(fig, axs[2])
total_heops_sparsity_log(fig, axs[2])
total_heops_reduction_sparsity(fig, axs[3])
memory_sparsity_log(fig, axs[4])
memory_reduction_sparsity(fig, axs[5])
performance_sparsity(fig, axs[6])
axs[7].axis('off')
axs[7].legend(handles=legend_handles, loc='center', prop={'size': 'x-large'})
fig.tight_layout()
fig.savefig(os.path.join('figures', 'all_log.pdf'))