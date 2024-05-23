import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--large', action='store_true')
parser.add_argument('--tasks', nargs='+')
args = parser.parse_args()

base_dir = os.path.join('results', 'large') if args.large else 'results'
for task in args.tasks:
    task_dir = os.path.join(base_dir, task)
    best_f1 = 0
    best_config = 0
    for config in os.listdir(task_dir):
        with open(os.path.join(task_dir, config, 'result.txt'), 'r') as f:
            f1 = float(f.readlines()[-2].strip().split()[-1])
            if f1 > best_f1:
                best_f1 = f1
                best_config = config
    print("%s, f1: %f, config: %s" % (task, best_f1, best_config))
