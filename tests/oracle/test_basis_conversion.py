"""Independent unbounded CRT sums exercise the compiled coefficient switch."""
import math
import random
import subprocess
import sys

from ntt_oracle import find_primes

rng = random.Random(0xB09)
queries, answers = [], []
witnesses = 0


def case(source, target, values, dimension=8, invalid=False):
    global witnesses
    queries.append(' '.join(map(str, [dimension, len(source), len(target)] + source + target + values)))
    if invalid:
        answers.append('invalid')
        return
    product = math.prod(source)
    hats = [product // modulus for modulus in source]
    inverses = [pow(hat, -1, modulus) for hat, modulus in zip(hats, source)]
    lifted = [sum(hat * (values[tower * dimension + index] * inverse % modulus)
                  for tower, (hat, inverse, modulus) in enumerate(zip(hats, inverses, source)))
              for index in range(dimension)]
    witnesses += sum(value % target[0] != (value % product) % target[0] for value in lifted)
    answers.append(' '.join(str(value % modulus) for modulus in target for value in lifted))


for bits in (30, 50, 55, 60, 63):
    primes = find_primes(65536, 43, bits)
    for count in (1, 2, 3, 8, 16, 24, 32, 40):
        source, target = primes[:count], primes[-3:]
        for reverse in (False, True):
            if reverse:
                source, target = target, source
            for pattern in range(4):
                values = [(0 if pattern == 0 else modulus - 1 if pattern == 1 else
                           int(index == 7) if pattern == 2 else rng.randrange(modulus))
                          for modulus in source for index in range(8)]
                case(source, target, values)
case([17, 97], [5, 113], [16] * 8 + [96] * 8)
case([17, 17], [97], [0] * 16, invalid=True)
case([17], [17], [0] * 8, invalid=True)
case([15], [97], [0] * 8, invalid=True)
case([], [97], [], invalid=True)
case([17], [], [0] * 8, invalid=True)
case([17], [97], [17] * 8, invalid=True)
case([17], [97], [0] * 3, dimension=3, invalid=True)
for modulus in (0, 1, 2, 1 << 63):
    case([modulus], [97], [0] * 8, invalid=True)
primes = find_primes(65536, 42, 30)
case(primes[:41], [primes[-1]], [0] * (41 * 8), invalid=True)
case([primes[-1]], primes[:41], [0] * 8, invalid=True)
result = subprocess.run([sys.argv[1]], input='\n'.join(queries) + '\n',
                        capture_output=True, text=True, check=True)
actual = result.stdout.splitlines()
if actual != answers:
    for index, (got, want) in enumerate(zip(actual, answers)):
        if got != want:
            raise AssertionError(f'case {index}: got {got}, expected {want}')
    raise AssertionError('wrong result count')
assert witnesses > 0
print(f'BASIS BIGINT PASS: {len(answers)} cases, {witnesses} canonical-CRT witnesses')
