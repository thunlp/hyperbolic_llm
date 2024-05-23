import re

pat = re.compile(r'F1: (\d+\.\d+)')

with open('results/squad/results.txt', 'r') as f:
    best = 0
    for line in f:
        result = pat.search(line)
        if result:
            f1 = float(result.group(1))
            if f1 > best:
                best = f1

print(best)
