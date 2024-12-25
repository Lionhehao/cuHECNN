import numpy as np
import time
import math
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
from resource import getrusage as resource_usage, RUSAGE_SELF

class hmm(object):
    def __init__(self, cipherSize):
        self.cipherSize = cipherSize
        self.HE = Pyfhel()
        self.HE.contextGen(
            scheme="ckks", n=cipherSize, scale=2**30, qi_sizes=[45, 30, 30, 30, 30, 40]
        )
        self.HE.keyGen()
        self.HE.relinKeyGen()
        self.HE.rotateKeyGen()

    def generate_l(self, column, case = 1):
        if case == 1:
            l = np.zeros(self.m * self.n)
            for i in range(column, self.m * self.n, self.n):
                l[i] = 1
        else:
            l = np.zeros(self.m * self.n * self.l)
            for i in range(column, self.m * self.n * self.l, self.l):
                l[i] = 1
        return l
    
    def generate_r(self, row):
        r = np.zeros(self.l * self.n)
        for i in range(row * self.n, (row + 1) * self.n):
            r[i] = 1
        return r
    
    def generate_mask(self, first, last):
        mask = np.zeros(self.m * self.n * self.l)
        for i in range(first * self.l, last * self.l):
            mask[i] = 1
        return mask
            
    def generate_filter(self):
        filter = np.zeros(self.m * self.n * self.l)
        for i in range(self.n):
            for j in range(self.m):
                filter[i * self.m * self.l + j * self.l + (j + i) % self.n] = 1
        return filter
    
    def matrixMul(self, matrixA, matrixB):
        self.m = matrixA.shape[0]
        self.l1 = matrixA.shape[1]
        self.l2 = matrixB.shape[0]
        self.n = matrixB.shape[1]

        if self.l1 != self.l2:
            print("cannot do matrix multiplication, dimention does not match")
            exit()

        self.l = self.l1

        if self.l1 < self.n:
            matrixA = np.pad(matrixA, ((0, 0), (0, self.n - self.l)))
            flatA = matrixA.copy().flatten(order="C")
            flatB = matrixB.copy().flatten(order="C")
            ctA = self.HE.encrypt(flatA)
            ctB = self.HE.encrypt(flatB)
            start = time.time()
            # A
            for i in range(self.l):
                
                plain_tmp = self.HE.encodeFrac(
                    self.generate_l(i)
                )
                
                if i == 0:
                    ctA1 = ctA * plain_tmp
                else:
                    tmp = ctA * plain_tmp
                    tmp >>= (self.m * self.n * i - i) % self.cipherSize / 2
                    ctA1 += tmp
            
            for i in range(int(math.log2(self.n))):
                tmp = ctA1.copy()
                tmp >>= 2**i
                ctA1 += tmp
            
            # B
            for i in range(self.l):
                plain_tmp = self.HE.encodeFrac(
                    self.generate_r(i)
                )
                if i == 0:
                    ctB1 = ctB * plain_tmp
                else:
                    tmp = ctB * plain_tmp
                    tmp >>= self.m * self.n * i - self.n * i
                    ctB1 += tmp

            for i in range(int(math.log2(self.m))):
                tmp = ctB1.copy()
                tmp >>= 2**i * self.n
                ctB1 += tmp

            # C
            ctC = ctA1 * ctB1
            self.HE.relinearize(ctC)
            for i in range(int(math.log2(self.l))):
                tmp = ctC.copy()
                tmp <<= (2**i * self.m * self.n) % (self.cipherSize / 2)
                ctC += tmp
            end = time.time()
            print("Ours: ", end - start, "s")
        else:
            matrixB = matrixB.transpose()
            flatA = matrixA.copy().flatten(order="C")
            flatB = matrixB.copy().flatten(order="C")
            ctA = self.HE.encrypt(flatA)
            ctB = self.HE.encrypt(flatB)
            start = time.time()
            # A
            for i in range(int(math.log2(self.n))):
                tmp = ctA.copy()
                tmp >>= self.m * self.l * 2**i
                ctA += tmp

            # B
            if self.n < self.m:
                ctB1 = ctB.copy()
                cur_n = self.n
                while self.m - cur_n >= self.n:
                    tmp = ctB1.copy()
                    tmp >>= cur_n * self.l
                    ctB1 += tmp
                    cur_n *= 2
            else:
                ctB1 = ctB.copy()
                cur_n = self.n
            
            for i in range(self.n):
                plain_tmp1 = self.HE.encodeFrac(
                    self.generate_mask(i, min(i + self.m, cur_n))
                )
                if i == 0:
                    ctB2 = ctB1 * plain_tmp1
                else:
                    tmp = ctB1 * plain_tmp1
                    tmp >>= i * self.m * self.l - i * self.l
                    ctB2 += tmp
                
                if self.m + i > cur_n:
                    plain_tmp2 = self.HE.encodeFrac(
                        self.generate_mask(0, self.m + i - cur_n)
                    )
                    tmp = ctB1 * plain_tmp2
                    tmp >>= i * self.m * self.l- i * self.l + cur_n * self.l
                    ctB2 += tmp
            
            ctC = ctA * ctB2
            self.HE.relinearize(ctC)
            
            for i in range(int(math.log2(self.l))):
                tmp = ctC.copy()
                tmp <<= 2**i
                ctC += tmp
            
            plain_tmp = self.HE.encodeFrac(
                    self.generate_l(0, 2)
                )
            ctC *= plain_tmp

            for i in range(int(math.log2(self.n))):
                tmp = ctC.copy()
                tmp >>= 2**i
                ctC += tmp

            filter = self.HE.encodeFrac(self.generate_filter())
            ctC *= filter
            for i in range(int(math.log2(self.n))):
                tmp = ctC.copy()
                tmp <<= 2**i * self.l * self.m
                ctC += tmp
            
            end = time.time()
            print("Ours: ", end - start, "s")
            
if __name__ == "__main__":
    a = 64
    b = 1
    c = 64

    print("a: ", a, " b: ", b, " c: ", c)
    cipherSize = 2**13

    matrixA = np.zeros((a, b))
    matrixB = np.zeros((b, c))
    for i in range(matrixA.shape[0]):
        for j in range(matrixA.shape[1]):
            matrixA[i][j] = np.random.randint(low=1, high=9)

    for i in range(matrixB.shape[0]):
        for j in range(matrixB.shape[1]):
            matrixB[i][j] = np.random.randint(low=1, high=9)

    print(matrixA)
    print(matrixB)
    # Multiplies the input matrices using permutation matrices.
    trueC = np.matmul(matrixA, matrixB)
    print(trueC)
    MatrixA = matrixA.copy()
    MatrixB = matrixB.copy()

    hmm(cipherSize).matrixMul(MatrixA, MatrixB)