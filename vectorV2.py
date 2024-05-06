import math
import numbers

class Vector:
    """ A vector with useful vector/matrix operations.
    """
    def __init__(self, *args):
        self.components = args if args else (0, 0, 0)  # Default to a zero 3D vector

    @property
    def x(self):
        return self.components[0] if len(self.components) > 0 else 0

    @x.setter
    def x(self, val):
        if len(self.components) > 0:
            self.components = (val,) + self.components[1:]
        else:
            self.components = (val,)

    @property
    def y(self):
        return self.components[1] if len(self.components) > 1 else 0

    @y.setter
    def y(self, val):
        if len(self.components) > 1:
            self.components = (self.components[0], val) + self.components[2:]
        else:
            self.components = (self.components[0], val) if len(self.components) > 0 else (0, val)

    @property
    def z(self):
        return self.components[2] if len(self.components) > 2 else 0

    @z.setter
    def z(self, val):
        # Ensure there are enough components
        if len(self.components) < 3:
            self.components += (0,) * (3 - len(self.components))
        self.components = self.components[:2] + (val,)

    @property
    def w(self):
        return self.components[3] if len(self.components) > 3 else 0
    
    @w.setter
    def w(self, val=1):
        if len(self.components) < 4:
            self.components += (0,) * (4 - len(self.components))
        self.components = self.components[:3] + (val,)

    def norm(self):
        return math.sqrt(sum(comp ** 2 for comp in self.components))

    def normalize(self):
        norm = self.norm()
        return Vector(*(comp / norm for comp in self.components))

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def cross(self, other):
        assert len(self.components) == len(other.components) == 3, "Cross product only for 3D vectors."
        return Vector(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

    def __mul__(self, other):
        if isinstance(other, Vector):
            return self.dot(other)
        elif isinstance(other, numbers.Real):
            return Vector(*(comp * other for comp in self.components))

    def __truediv__(self, other):
        if isinstance(other, numbers.Real):
            return Vector(*(comp / other for comp in self.components))
    
    def __add__(self, other):
        if isinstance(other, Vector):
            return Vector(*(a + b for a, b in zip(self.components, other.components)))

    def __sub__(self, other):
        if isinstance(other, Vector):
            return Vector(*(a - b for a, b in zip(self.components, other.components)))

    def __len__(self):
        return len(self.components)

    def __iter__(self):
        return iter(self.components)

    def __repr__(self):
        return f"Vector({', '.join(map(str, self.components))})"
