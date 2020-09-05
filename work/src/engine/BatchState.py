"""Something."""
import torch
from .State import State


class BatchState(State):
    """Something."""

    def __len__(self):
        """Something."""
        return len(self.t)

    def __getitem__(self, index):
        """Something."""
        return BatchState(*self._state[:3])

    def copy(self):
        """Something."""
        new_q = self.q.clone().detach().requires_grad_(True)
        new_p = self.p.clone().detach().requires_grad_(True)
        new_t = self.t.clone().detach()
        return BatchState(new_q, new_p, new_t)