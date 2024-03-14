import math
import numpy as np
import numbers
from vector import Vector

class Matrix(object):
    def __init__(self, elements):
        self.elements = elements
        self.rows = len(elements)
        self.cols = len(elements[0])

    def printEle(self):
        for row in self.elements:
            print(row)

    def __mul__(self, other):
        if isinstance(other, Matrix):
            self.multMatrix(other)
        elif isinstance(other, numbers.Real):
            self.multScaler(other)
        else:
            raise TypeError("Unsupported type for multiplication")
    
    def multMatrix(self, other):
        if self.cols != other.rows:
            raise ValueError("Incompatible matrix dimensions for multiplication")

        result = [[0 for _ in range(other.cols)] for _ in range(self.rows)]

        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    result[i][j] += self.elements[i][k] * other.elements[k][j]

        return Matrix(result)

    def multScaler(self, scalar):
        result = [[element * scalar for element in row] for row in self.elements]
        return Matrix(result)

    def dotMAT(self, other):
        if (self.rows != 1 and self.cols != 1) or (other.rows != 1 and other.cols != 1) or self.cols != other.cols:
            raise ValueError("Incompatible matrix dimensions for dot product")

        result = sum(self.elements[0][i] * other.elements[0][i] for i in range(self.cols))
        return result

    def crossMAT(self, other):
        if self.rows != 1 or other.rows != 1 or self.cols != 3 or other.cols != 3:
            raise ValueError("Incompatible matrix dimensions for cross product")

        result = [
            self.elements[0][1] * other.elements[0][2] - self.elements[0][2] * other.elements[0][1],
            self.elements[0][2] * other.elements[0][0] - self.elements[0][0] * other.elements[0][2],
            self.elements[0][0] * other.elements[0][1] - self.elements[0][1] * other.elements[0][0]
        ]
        return Matrix([result])
