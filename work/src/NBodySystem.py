"""Something."""
from . import Hamiltonian


class NBodies(Hamiltonian):
    """Something."""

    def __init__(self, G):
        """Something."""
        self._G = G

    @property
    def G(self):
        """Something."""
        return self._G

    def evaluate(self, q, p, m):
        """Something."""
        T = (0.5 * (p**2).sum(1) / m).sum()
        U = 0
        for i in range(q.shape[0]):
            for j in range(i + 1, q.shape[0]):
                U -= self.G * m[i] * m[j] / (q[i] - q[j]).norm()
        return T + U