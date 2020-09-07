"""Something."""
import torch


class State(object):
    """Object to represent state in phase space."""

    def __init__(self, q=None, p=None, t=None, shape=None, device=None, dtype=None):
        """Initialize State."""
        # Calculate device and dtype settings
        if device is None:
            if any([type(x) is not torch.Tensor for x in [q, p, t]]):
                device = torch.device('cpu')
            elif q.device == p.device and p.device == t.device:
                device = q.device
            else:
                raise Exception('''Either input tensors must be on the same device or you
                                   must provide an explicit device.''')
        if dtype is None:
            if any([type(x) is not torch.Tensor for x in [q, p, t]]):
                dtype = torch.float32
            elif q.dtype == p.dtype and p.dtype == t.dtype:
                dtype = q.dtype
            else:
                raise Exception('''Either input tensors must be of the same type or you
                                   must provide an explicit type.''')
        # Build tensors based on shape if q, p, or t missing
        if q is None or p is None or t is None:
            if shape is None:
                raise Exception('''Either provide input tensors q, p, and t or provide
                                   shape to build q and p. ''')
            q, p, t = self._build_from_shape(shape, device, dtype)
        # Cast q, p, and t to tensors if neccesary
        if any([type(x) is not torch.Tensor for x in [q, p, t]]):
            q = torch.tensor(q, device=device, dtype=dtype)
            p = torch.tensor(p, device=device, dtype=dtype)
            t = torch.tensor(t, device=device, dtype=dtype)
        # Check dimensions
        self._check_dim(q, p, t)
        # Save State
        self._state = (q.to(device=device, dtype=dtype).requires_grad_(True),
                       p.to(device=device, dtype=dtype).requires_grad_(True),
                       t.to(device=device, dtype=dtype))

    def _check_dim(self, q, p, t):
        if q.shape != p.shape:
            raise Exception('''Dimensions of q and p do not match.''')
        if t.shape != ():
            raise Exception('''t is not a scalar.''')

    def _build_from_shape(self, shape, device, dtype):
        q = torch.zeros(shape, device=device, dtype=dtype)
        p = torch.zeros(shape, device=device, dtype=dtype)
        t = torch.zeros((), device=device, dtype=dtype)
        return q, p, t

    def __len__(self):
        """Something."""
        return 3

    def __getitem__(self, index):
        """Something."""
        if type(index) is not tuple:
            return self._state[index]
        return (self.q[index[1:]], self.p[index[1:]], self.t)[index[0]]

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
        return type(self)(q=new_q, p=new_p, t=new_t)

    def copy(self):
        """Something."""
        new_q = self.q.clone().detach().requires_grad_(True)
        new_p = self.p.clone().detach().requires_grad_(True)
        new_t = self.t.clone().detach()
        return type(self)(q=new_q, p=new_p, t=new_t)

    @torch.no_grad()
    def zero_grad(self):
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
        return type(self)(q=new_q, p=new_p, t=new_t)

    @torch.no_grad()
    def step_(self, dq_dt, dp_dt, dt):
        """Something."""
        self._state[0].add_(dq_dt, alpha=dt)
        self._state[1].add_(dp_dt, alpha=dt)
        self._state[2].add_(dt)


class StateList(State):
    """Object to represent list of states in phase space."""

    def __init__(self, q=None, p=None, t=None, shape=None, device=None, dtype=None):
        """Something."""
        super(StateList, self).__init__(q, p, t, shape, device, dtype)

    def _check_dim(self, q, p, t):
        if q.shape != p.shape:
            raise Exception('''Dimensions of q and p do not match.''')
        if len(t.shape) != 1:
            raise Exception('''t must be a 1D Tensor''')
        if t.shape[0] != q.shape[0]:
            raise Exception('''First diminsion of t and q/p do not match.''')

    def _build_from_shape(self, shape, device, dtype):
        q = torch.zeros(shape, device=device, dtype=dtype)
        p = torch.zeros(shape, device=device, dtype=dtype)
        t = torch.zeros(shape[0], device=device, dtype=dtype)
        return q, p, t

    def __len__(self):
        """Something."""
        return len(self.t)

    def __getitem__(self, index):
        """Something."""
        if type(index) is not tuple:
            if type(index) is not slice:
                return State(self.q[index], self.p[index], self.t[index])
            return StateList(self.q[index], self.p[index], self.t[index])
        elif len(index) == 2:
            return (self.q[index[0]], self.p[index[0]], self.t[index[0]])[index[1]]
        return (self.q[(index[0], *index[2:])],
                self.p[(index[0], *index[2:])],
                self.t[index[0]])[index[1]]

    def __setitem__(self, index, state):
        """Something."""
        self._state[0][index] = state.q
        self._state[1][index] = state.p
        self._state[2][index] = state.t