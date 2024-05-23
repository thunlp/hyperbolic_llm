import os
from argparse import ArgumentParser
import re
import numpy as np


def main(args):
    for task in args.tasks:
        if task == 'mrpc' or task == 'qqp':
            pat = re.compile(r'[^_]f1 = (0\.\d+)')
        else:
            pat = re.compile(r'best_result = (0\.\d+)')

        best = 0
        best_lr = 0
        best_std = 0
        for lr in ['lr%de-5' % i for i in range(1, 6)]:
            results = []
            for seed in [0, 1, 2, 3, 4]:
                seed_best = 0
                logfile = os.path.join('results', 'large' if args.large else '', task, lr, 'seed' + str(seed), 'log')
                if not os.path.exists(logfile):
                    continue
                with open(logfile, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        result = pat.search(line)
                        if result is not None:
                            metric = float(result.group(1))
                            if metric > seed_best:
                                seed_best = metric
                    results.append(seed_best)
            if np.mean(results) > best:
                best = np.mean(results)
                best_std = np.std(results)
                best_lr = lr
        print("Task: %s, best lr:%s, best dev: %f, std: %f" % (task, best_lr, best, best_std))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tasks', nargs='+', default=['mrpc', 'qqp', 'sst-2', 'rte', 'mnli', 'qnli', 'cola'])
    parser.add_argument('--large', action='store_true')
    parser.add_argument('--degree1', action='store_true')
    args = parser.parse_args()
    main(args)
