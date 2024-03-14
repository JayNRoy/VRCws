from matrix import Matrix
import math

class TranslateMatrix(Matrix):
    # Digits is a LIST of translation integers correlating to the 2D or 3D dimensions etc.
    # Eg. A 3D translation matrix would have 3 digits, 5D --> 5 digits: [4, 1, 4, 5, 6]
    # Then each matrix would homogenously have an extra dimension
    def __init__(self, digits):
        elements = []
        for i in range(len(digits)):
            row = [0] * i
            row.append(1)
            row = row + ([0] * (len(digits) - 1 - i))
            row.append(digits[i])
            elements.append(row)
        # Homogeneous point
        row = ([0] * len(digits)) + [1]
        elements.append(row)
        super().__init__(elements)

class RotateMatrix(Matrix):
    # Restricted to the 3D homogeneous space (becomes a matrix of 4 dimensions)
    # Axis is a string arguement out of "x", "y" and "z"
    # theta is an angle given in radians
    # Then each matrix would homogenously have an extra dimension
    def __init__(self, theta, axis):
        cosine = math.cos(theta)
        sine = math.sin(theta)
        elements = []
        for i in range(4):
            row = [0] * i
            row.append(1)
            row = row + ([0] * (4 - 1 - i))
            elements.append(row)
        if axis.lower() == "x":
            # Rotation on the x-axis
            elements[1][1] = cosine
            elements[1][2] = -1 * sine
            elements[2][1] = sine
            elements[2][2] = cosine
        elif axis.lower() == "y":
            # Rotation on the y-axis
            elements[0][0] = cosine
            elements[0][2] = sine
            elements[2][0] = -1 * sine
            elements[2][2] = cosine
        elif axis.lower() == "z":
            # Rotation on the z-axis
            elements[0][0] = cosine
            elements[0][1] = -1 * sine
            elements[1][0] = sine
            elements[1][1] = cosine
        else:
            raise ValueError("Invalid axis provided for rotation matrix.")
        super().__init__(elements)

class ScaleMatrix(Matrix):
    # Digits is a list of scaler integers for each coordinate component
    # Eg. [1, 4, 2] for a 3D point
    # Then each matrix would homogenously have an extra dimension
    def __init__(self, digits):
        elements = []
        for i in range(len(digits)):
            row = [0] * i
            row.append(digits[i])
            row = row + ([0] * (len(digits) - i))
            elements.append(row)
        # Homogeneous point
        row = ([0] * len(digits)) + [1]
        elements.append(row)
        super().__init__(elements)

"""
Test:
elements = [
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, 0],
    [0, 0, 0, 1]
]

"""