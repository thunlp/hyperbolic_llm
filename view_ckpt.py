import torch
from run_pretraining import pretraining_dataset, WorkerInitObj
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--input', type=str, required=True)
args = parser.parse_args()

tmp = torch.load(args.input, 'cpu')

def inspect(name):
    print('\n%s'%name)
    for k, v in tmp['model'].items():
        if name in k:
            print(k, v.exp())

# for k, v in tmp['model'].items():
#     if 'scale' in k:
#         print(k, v.exp())
inspect('scale')
inspect('dense_act.scale')
inspect('query.scale')
inspect('key.scale')
inspect('value.scale')
inspect('output.residual.scale')

