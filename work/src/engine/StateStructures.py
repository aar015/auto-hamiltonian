"""Something."""
import torch


class State(object):
    """Something."""

    def __init__(self, q, p, t, device=None, dtype=None):
        """Something."""
        if type(q) is torch.Tensor and type(p) is torch.Tensor and type(t) is torch.Tensor():
            if device is None:
                if q.device != p.device or p.device != t.device:
                    raise Exception('''Generalized Coordinate Tensor, Conjugate Momenta Tensor, and Time
                                       Tensor must be on the same device.''')
                device = q.device
            if dtype is None:
                if q.dtype != p.dtype or p.dtype != t.dtype:
                    raise Exception('''Generalized Coordinate Tensor, Conjugate Momenta Tensor, and Time
                                       Tensor must be of the same type.''')
                dtype = q.dtype
            self._state = [q.to(device=device, dtype=dtype).requires_grad_(True),
                           p.to(device=device, dtype=dtype).requires_grad_(True),
                           t.to(device=device, dtype=dtype).requires_grad_(True)]
        else:
            if dtype is None:
                dtype = torch.float32
            if device is None:
                device = torch.device('cpu')
            self._state = [torch.tensor(q, device=device, dtype=dtype, requires_grad=True),
                           torch.tensor(p, device=device, dtype=dtype, requires_grad=True),
                           torch.tensor(t, device=device, dtype=dtype)]

    def __len__(self):
        """Something."""
        return 3

    def __getitem__(self, index):
        """Something."""
        return self._state[index]

    def __repr__(self):
        """Something."""
        return '\n'.join(['q: ' + str(self.q),
                          'p: ' + str(self.p),
                          't: ' + str(self.t)])

    @property
    def q(self):
        """Something."""
        return self._state[0]

    @property
    def p(self):
        """Something."""
        return self._state[1]

    @property
    def t(self):
        """Something."""
        return self._state[2]

    @property
    def device(self):
        """Something."""
        if self.q.device != self.p.device or self.p.device != self.t.device:
            raise Exception(''''Generalized Coordinate Tensor, Conjugate Momenta Tensor, and Time Tensor
                                are no longer on the same device. This error is likely a result of
                                directly modifing the internal state representation state._state.''')
        return self.q.device

    @property
    def dtype(self):
        """Something."""
        if self.q.dtype != self.p.dtype or self.p.device != self.t.device:
            raise Exception('''Generalized Coordinate Tensor, Conjugate Momenta Tensor, and Time Tensor
                               are no longer of the same type. This error is likely a result of directly
                               modifing the internal state representation state._state.''')
        return self.q.dtype

    def to(self, device=None, dtype=None):
        """Something."""
        if all([(device is None or device == self.device),
                (dtype is None or dtype == self.dtype)]):
            return self
        new_q = self.q.to(device=device, dtype=dtype)
        new_p = self.p.to(device=device, dtype=dtype)
        new_t = self.t.to(device=device, dtype=dtype)
        return State(new_q, new_p, new_t)

    def copy(self):
        """Something."""
        new_q = self.q.clone().detach().requires_grad_(True)
        new_p = self.p.clone().detach().requires_grad_(True)
        new_t = self.t.clone().detach()
        return State(new_q, new_p, new_t)

    @torch.no_grad()
    def zero_grad_(self):
        """Something."""
        if self.q.grad is not None:
            self.q.grad.zero_()
        if self.p.grad is not None:
            self.p.grad.zero_()

    @torch.no_grad()
    def step(self, dq_dt, dp_dt, dt):
        """Something."""
        new_q = self.q.add(dq_dt, alpha=dt).requires_grad_(True)
        new_p = self.p.add(dp_dt, alpha=dt).requires_grad_(True)
        new_t = self.t.add(dt)
        return State(new_q, new_p, new_t)

    @torch.no_grad()
    def step_(self, dq_dt, dp_dt, dt):
        """Something."""
        self._state[0].add_(dq_dt, alpha=dt)
        self._state[1].add_(dp_dt, alpha=dt)
        self._state[2].add_(dt)


class BatchState(State):
    """Something."""

    def __init__(self, q, p, t, device=None, dtype=None):
        """Something."""
        super(BatchState, self).__init__(q, p, t, device, dtype)
        assert len(self.q.shape) >= 2
        assert len(self.p.shape) >= 2
        assert len(self.t.shape) == 1

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

    @torch.no_grad()
    def step(self, dq_dt, dp_dt, dt):
        """Something."""
        new_q = self.q.add(dq_dt, alpha=dt).requires_grad_(True)
        new_p = self.p.add(dp_dt, alpha=dt).requires_grad_(True)
        new_t = self.t.add(dt)
        return State(new_q, new_p, new_t)