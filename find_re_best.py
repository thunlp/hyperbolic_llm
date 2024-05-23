import re
import os

def find_best(file):
    if not os.path.exists(file):
        return 0, None
    with open(file, 'r') as f:
        best = 0
        best_config = None
        while True:
            try:
                config = f.readline()
                f.readline()
                f1 = float(f.readline().split('=')[-1])
                if f1 > best:
                    best = f1
                    best_config = config
                f.readline()
            except:
                break
    return best, best_config
        
def find_best_dataset(dataset):
    best, config = find_best('results/large/' + dataset + '/result.txt')
    best2, config2 = find_best('results/large/'+ dataset + '_2/result.txt')
    if best2 > best:
        best = best2
        config = config2
    print(config)
    print("%s:%f" % (dataset, best))

find_best_dataset('tacred')
find_best_dataset('tacrev')
find_best_dataset('retacred')

