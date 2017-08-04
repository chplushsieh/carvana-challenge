
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("num_start", help="start number")
parser.add_argument("num_end",   help="end number")
args = parser.parse_args()

num_start = int(args.num_start)
num_end   = int(args.num_end)

def times_divisible_by(dividend, divisor):
    if (dividend % divisor) == 0:
        return 1 + times_divisible_by(dividend/divisor, divisor)
    else:
        return 0

for num in range(num_start, num_end):
    two_pow = times_divisible_by(num, 2)

    if two_pow > 3:
        print("{} can be divided by 2^{}".format(num, two_pow))
