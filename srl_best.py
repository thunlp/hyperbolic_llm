import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--path', type=str, required=True)
# parser.add_argument('--dataset', type=str, choices=['conll2005', 'conll2009', 'conll2012'])
args = parser.parse_args()

best_dev = 0
best_test = None
best_config = None
for dir in os.listdir(args.path):
    ckpt_path = os.path.join(args.path, dir)
    with open(os.path.join(ckpt_path, 'best_dev.txt'), 'r') as f:
        dev = float(f.readline().strip())
        if dev > best_dev:
            best_dev = dev
            best_config = open(os.path.join(ckpt_path, 'args.txt'), 'r').readline().strip()
            with open(os.path.join(ckpt_path, 'test_result.txt'), 'r') as f:
                best_test = f.readlines()

print(best_config)
print("Dev: %f" % best_dev)
print(str(best_test))
