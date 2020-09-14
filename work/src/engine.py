"""Something."""
import functools
import torch


def hamiltonian(func):
    """Decorate Hamiltonian Function."""

    @functools.wraps(func)
    def wrapper(state, *args, **kwargs):
        if type(state) is State:
            q = state.q.unsqueeze(0)
            p = state.p.unsqueeze(0)
            t = state.t.unsqueeze(0)
            return func(q, p, t, *args, **kwargs)[0]
        elif type(state) is StateList:
            q = state.q
            p = state.p
            t = state.t
            return func(q, p, t, *args, **kwargs)
        else:
            raise Exception('Must pass State or StateList to Hamiltonian')

    return wrapper


class State(object):
    """Object to represent state in phase space."""

    def __init__(self, q, p, t):
        """Initialize State."""
        self._state = (q.requires_grad_(True), p.requires_grad_(True), t)
        self.device
        self.dtype
        self.shape

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
        names = ['q', 'p', 't']
        return '\n'.join([name + ': ' + str(x) for name, x, in zip(names, self)])

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
            raise Exception('Found tensors on multiple devices.')
        return self.q.device

    @property
    def dtype(self):
        """Something."""
        if self.q.dtype != self.p.dtype or self.p.device != self.t.device:
            raise Exception('Found tensors of multiple types.')
        return self.q.dtype

    @property
    def shape(self):
        """Something."""
        if self.q.shape != self.p.shape:
            raise Exception('''Dimensions of q and p do not match.''')
        if self.t.shape != ():
            raise Exception('''t is not a scalar.''')
        return self.q.shape

    @torch.no_grad()
    def to(self, device=None, dtype=None):
        """Something."""
        if (device is None or device == self.device) and (dtype is None or dtype == self.dtype):
            return self
        return type(self)(q=self.q.to(device=device, dtype=dtype),
                          p=self.p.to(device=device, dtype=dtype),
                          t=self.t.to(device=device, dtype=dtype))

    @torch.no_grad()
    def copy(self):
        """Something."""
        return type(self)(q=self.q.clone(), p=self.p.clone(), t=self.t.clone())

    @torch.no_grad()
    def zero_grad(self):
        """Something."""
        if self.q.grad is not None:
            self.q.grad.zero_()
        if self.p.grad is not None:
            self.p.grad.zero_()

    def dq_dt(self, a=0.0, b=1.0):
        """Something."""
        return a * self.q.grad + b * self.p.grad

    def dp_dt(self, c=-1.0, d=0.0):
        """Something."""
        return c * self.q.grad + d * self.p.grad

    @torch.no_grad()
    def step(self, dt, dq_dt=None, dp_dt=None, a=0.0, b=1.0, c=-1.0, d=0.0):
        """Something."""
        if dq_dt is None:
            dq_dt = self.dq_dt(a=a, b=b)
        if dp_dt is None:
            dp_dt = self.dp_dt(c=c, d=d)
        new_q = self.q.add(dq_dt, alpha=dt).requires_grad_(True)
        new_p = self.p.add(dp_dt, alpha=dt).requires_grad_(True)
        new_t = self.t.add(dt)
        return type(self)(q=new_q, p=new_p, t=new_t)

    @torch.no_grad()
    def step_(self, dt, dq_dt=None, dp_dt=None, a=0.0, b=1.0, c=-1.0, d=0.0):
        """Something."""
        if dq_dt is None:
            dq_dt = self.dq_dt(a=a, b=b)
        if dp_dt is None:
            dp_dt = self.dp_dt(c=c, d=d)
        self._state[0].add_(dq_dt, alpha=dt)
        self._state[1].add_(dp_dt, alpha=dt)
        self._state[2].add_(dt)


class StateList(State):
    """Object to represent list of states in phase space."""

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

    @property
    def shape(self):
        """Something."""
        if self.q.shape != self.p.shape:
            raise Exception('''Dimensions of q and p do not match.''')
        if len(self.t.shape) != 1:
            raise Exception('''t must be a 1D Tensor''')
        if self.t.shape[0] != self.q.shape[0]:
            raise Exception('''First diminsion of t and q/p do not match.''')
        return self.q.shape