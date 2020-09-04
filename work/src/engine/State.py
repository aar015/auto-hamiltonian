"""Something."""
import torch


class State(object):
    """Something."""

    def __init__(self, q, p, t, dtype=torch.float32, device=torch.device('cpu')):
        """Something."""
        self._state = [torch.tensor(q, dtype=dtype, requires_grad=True, device=device),
                       torch.tensor(p, dtype=dtype, requires_grad=True, device=device),
                       t, None]

    def __len__(self):
        """Something."""
        return 4

    def __getitem__(self, index):
        """Something."""
        return self._state[index]

    def __repr__(self):
        """Something."""
        return '\n'.join(['q: ' + str(self.q),
                          'p: ' + str(self.p),
                          't: ' + str(self.t),
                          'H: ' + str(self.H),
                          'dH/dq: ' + str(self.q.grad),
                          'dH/dp: ' + str(self.p.grad)])

    @property
    def q(self):
        """Something."""
        return self[0]

    @property
    def p(self):
        """Something."""
        return self[1]

    @property
    def t(self):
        """Something."""
        return self[2]

    @property
    def H(self):
        """Something."""
        return self[3]

    @property
    def dtype(self):
        """Something."""
        if self.q.dtype != self.p.dtype:
            raise Exception('''Generalized Coordinate Tensor and Conjugate Momenta Tensor are no
                               longer of the same type.''')
        return self.q.dtype

    @property
    def device(self):
        """Something."""
        if self.q.device != self.p.device:
            raise Exception(''''Generalized Coordinate Tensor and Conjugate Momenta Tensor are no
                                longer on the same device. If you are trying to move the state to
                                another device use state.to(device) instead of state.q.to(device) and
                                state.p.to(device).''')
        return self.q.device

    def zero_grad(self):
        """Something."""
        if self.q.grad is not None:
            self.q.grad.zero_()
        if self.p.grad is not None:
            self.p.grad.zero_()

    def to(self, device):
        """Something."""
        self._state[0] = self.q.to(device)
        self._state[1] = self.p.to(device)

    @torch.no_grad()
    def advance_(self, dq_dt, dp_dt, dt):
        """Something."""
        self._state[0].add_(dq_dt, alpha=dt)
        self._state[1].add_(dp_dt, alpha=dt)
        self._state[2] += dt