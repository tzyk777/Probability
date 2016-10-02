import time
from itertools import combinations
from math import factorial

import numpy as np

np.random.seed(0)
# Be careful to set K, it may kill your computer!!!
K = 25
N = 100
r = np.random.uniform(0, 1, K)
r = r/sum(r)


def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('%s function took %0.3f ms' % (f.__name__, (time2-time1)*1000.0))
        return ret
    return wrap


# Inclusion-exclusion theorem
@timing
def inclusion_exclusion(r, K, N):
    a = [list(combinations(r, i)) for i in range(1, K+1)]
    z = [sum(np.power(1-sum(x), N) for x in b) for b in a]
    z = [np.power(-1, idx)*z[idx] for idx in range(len(z))]
    return 1 - sum(z)


def NcR(n, k):
    numerator = factorial(n)
    denominator = (factorial(k)*factorial(n-k))
    answer = numerator/denominator
    return answer


# Conditional probability approach
@timing
def conditional_probability(r, K, N):
    Matrix = np.zeros((N+1, K+1))
    Matrix[1:,1] = 1
    for col in range(2, K+1):
        probs = r[:col]/sum(r[:col])
        pr = probs[-1]
        pre = 1 - pr
        for row in range(col, N+1):
            start_point = 1
            end_point = row - col + 1
            p = 0
            for j in range(start_point,  end_point+1):
                p += Matrix[row-j, col-1]*NcR(row, j)*np.power(pr, j)*np.power(pre, row-j)
            Matrix[row, col] = p
    return Matrix[N, K]


probability_1 = inclusion_exclusion(r, K, N)
print(probability_1)
probability_2 = conditional_probability(r, K, N)
print(probability_2)
