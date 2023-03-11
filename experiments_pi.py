import os
import sys
import subprocess
import time
import itertools

N_RUNS = 5

SPARSITIES = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 0]  # 0 is special value
#that runs the
# he-friendly model

# get all available experiments
current_dir = os.listdir()

experiments = [
    f.replace('experiment_', '')
    for f in current_dir
    if os.path.isdir(f) and f.startswith('experiment_')
]
experiments.sort()
print('Experinments found:', *experiments)


# function to call a subprocess and run the experiment
def run(experiment, sparsity):
  # create arguments
  if sparsity == 0:
    s = ['-O']
  else:
    s = ['-s', str(sparsity)]
  args = [
      'python3',  #
      'private_inference.py',
      experiment,
      *s,
      '--parallel_encryption',
      '--log_memory',
      '--count_operations',
      '--clear_memory'
  ]

  completed = subprocess.run(args, capture_output=True, cwd=os.curdir)
  if completed.returncode != 0:
    print('Process failed: \nstdout:')
    print(completed.stdout.decode())
    print('stderr:')
    print(completed.err.decode())
    print('exiting')
    exit(1)


# check how many experiments we need to run
experiments_to_run = [[] for _ in range(N_RUNS)]
for exp in experiments:
  exp_dir = 'experiment_' + exp
  results = os.listdir(os.path.join(exp_dir, 'results'))
  for sparsity in SPARSITIES:
    prefix = 'hefriendly_' if sparsity == 0 else 'pruned_' + str(sparsity)
    files = [f for f in results if f.startswith(prefix)]
    # number of runs we need to perform
    n = N_RUNS - len(files)
    if n <= 0:
      continue
    # append to the experiments to run
    for i in range(n):
      experiments_to_run[i].append((exp, sparsity))

experiments_to_run = list(itertools.chain(*experiments_to_run))

print(experiments_to_run)
exit()

# run experiments
time_tracker = []
for i, (exp, sparsity) in enumerate(experiments_to_run):
  if i == 0:
    sys.stdout.write(
        f'\rrunning: {exp} sparsity={sparsity} progress {i+1}/{len(experiments_to_run)}'
    )
  else:
    # clear line
    avg_time = sum(time_tracker) / len(time_tracker)
    time_remaining = avg_time * (len(experiments_to_run) - n)
    sys.stdout.write(
        '\r                                                                                 '
    )
    sys.stdout.write(
        f'\rrunning: {exp} sparsity={sparsity} progress {i+1}/{len(experiments_to_run)}. estimated time remaining {int(time_remaining)}s'
    )
  start = time.time()
  run(exp, sparsity)
  time_tracker.append(time.time() - start)
