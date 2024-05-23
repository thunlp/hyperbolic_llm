from argparse import ArgumentParser
import matplotlib.pyplot as plt

parser = ArgumentParser()
parser.add_argument('--input', required=True)
# parser.add_argument('--coef', type=float, default=0)
parser.add_argument('--step', type=int, default=50)
parser.add_argument('--limit', type=int, default=0)
args = parser.parse_args()

losses = []
last_loss = 0
with open(args.input) as f:
    string = 'average_loss'
    for line in f:
        pos = line.strip().find(string)
        if pos != -1:
            pos += len(string) + 3
            loss = line[pos: pos + 6].replace(':', '')
            # losses.append(args.coef * last_loss + (1 - args.coef) * float(loss))
            # last_loss = float(loss)
            losses.append(float(loss))
            if args.limit != 0 and len(losses) * args.step >= args.limit:
                break
plt.plot(range(0, len(losses) * args.step, args.step), losses)
plt.savefig('loss.pdf')