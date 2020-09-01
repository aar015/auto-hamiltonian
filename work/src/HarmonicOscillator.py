"""Something."""
from . import Hamiltonian


class HarmonicOscillator(Hamiltonian):
    """Something."""

    def __init__(self, k):
        """Something."""
        self._k = k

    @property
    def k(self):
        """Something."""
        return self._k

    def evaluate(self, q, p, m):
        """Something."""
        T = (0.5 * (p**2).sum(1) / m).sum()
        U = (0.5 * self.k * q**2).sum()
        return T + U