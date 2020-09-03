"""Something."""
import torch
from torch.nn.functional import mse_loss


class Engine(object):
    """Something."""

    def __init__(self, q0, p0, step_size, drag=0.01, target=0,
                 loss=mse_loss, mode='conserve', **params):
        """Something."""
        self._q = torch.tensor(q0, requires_grad=True)
        self._p = torch.tensor(p0, requires_grad=True)
        self._t = 0
        self._H = None
        self._step_size = step_size
        self._drag = drag
        self._target = target
        self._loss = mse_loss
        self._mode = mode
        for key in params:
            params[key] = torch.tensor(params[key])
        self._params = params
        self.disable_history()

    @property
    def q(self):
        """Something."""
        return self._q

    @property
    def p(self):
        """Something."""
        return self._p

    @property
    def t(self):
        """Something."""
        return self._t

    @property
    def state(self):
        """Something."""
        return (self.q, self.p, self.t)

    @property
    def step_size(self):
        """Something."""
        return self._step_size

    @step_size.setter
    def step_size(self, new_step_size):
        """Something."""
        self._step_size = new_step_size

    @property
    def drag(self):
        """Something."""
        return self._drag

    @drag.setter
    def drag(self, new_drag):
        """Something."""
        self._drag = new_drag

    @property
    def target(self):
        """Something."""
        return self._target

    @target.setter
    def target(self, new_target):
        """Something."""
        self._target = new_target

    @property
    def loss(self):
        """Something."""
        return self._loss

    @loss.setter
    def loss(self, new_loss):
        """Something."""
        self._loss = new_loss

    @property
    def mode(self):
        """Something."""
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        """Something."""
        return self._mode

    @property
    def params(self):
        """Something."""
        return self._params

    @property
    def history(self):
        """Something."""
        return self._history

    def dq_dt(self):
        """Something."""
        return self.p.grad

    def dp_dt(self, H=None):
        """Something."""
        if self.mode == 'conserve':
            return -1 * self.q.grad
        elif self.mode == 'dissipate':
            return -1 * self.q.grad - self._drag * self.p.grad
        elif self.mode == 'target':
            if H is None:
                raise Exception('Need to Provide Closure to Use Target Mode')
            with torch.enable_grad():
                H = H.clone().detach().requires_grad_(True)
                loss = self.loss(H, self.target)
                loss.backward()
            return -1 * self.q.grad - H.grad * self.p.grad

    @torch.no_grad()
    def step(self, closure=None):
        """Something."""
        H = None
        if closure is not None:
            with torch.enable_grad():
                H = closure()
        self.q.add_(self.dq_dt(), alpha=self._step_size)
        self.p.add_(self.dp_dt(H), alpha=self._step_size)
        self._t += self._step_size
        if self._history is not None:
            self._history._append(self.q, self.p, self.t, H)

    def zero_grad(self):
        """Something."""
        if self.q.grad is not None:
            self.q.grad.zero_()
        if self.p.grad is not None:
            self.p.grad.zero_()

    def evaluate(self, hamiltonian, **kwargs):
        """Something."""
        return hamiltonian(*self.state, **self.params, **kwargs)

    def enable_history(self, num_entries, H=None):
        """Something."""
        if num_entries <= 0:
            self.disable_history()
        else:
            self._history = _StateHistory(num_entries + 1, self.q.shape, self.p.shape)
            self._history._append(self.q, self.p, self.t, H)

    def disable_history(self):
        """Something."""
        self._history = None


class _StateHistory(object):

    def __init__(self, num_entries, q_shape, p_shape):
        self._q_history = torch.zeros((num_entries, *q_shape))
        self._p_history = torch.zeros((num_entries, *p_shape))
        self._t_history = torch.zeros(num_entries)
        self._H_history = torch.zeros(num_entries)
        self._num_entries = num_entries
        self._entry_count = 0

    def __len__(self):
        return self._entry_count

    def __getitem__(self, index):
        if type(index) is not tuple:
            return (self._q_history[index], self._p_history[index],
                    self._t_history[index], self._H_history[index])
        elif len(index) == 2:
            return (self._q_history[index[0]], self._p_history[index[0]],
                    self._t_history[index[0]], self._H_history[index[0]])[index[1]]
        else:
            return (self._q_history[(index[0], *index[2:])], self._p_history[(index[0], *index[2:])],
                    self._t_history[index[0]], self._H_history[index[0]])[index[1]]

    def _append(self, q, p, t, H=None):
        if self._entry_count == self._num_entries:
            raise Exception('History Overflow')
        self._q_history[self._entry_count] = q.detach().clone()
        self._p_history[self._entry_count] = p.detach().clone()
        self._t_history[self._entry_count] = t
        if H is not None:
            self._H_history[self._entry_count] = H
        self._entry_count += 1