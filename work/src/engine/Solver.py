"""Something."""
import torch


class Solver(object):
    """Something."""

    def __init__(self, drag=1, target=0, rate=1, mode='conserve'):
        """Something."""
        self._drag = drag
        self._target = target
        self._rate = rate
        self._mode = mode

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
    def rate(self):
        """Something."""
        return self._rate

    @rate.setter
    def rate(self, new_rate):
        """Something."""
        self._rate = new_rate

    @property
    def mode(self):
        """Something."""
        return self._mode

    @mode.setter
    def mode(self, new_mode):
        """Something."""
        return self._mode

    def _dq_dt(self, state):
        """Something."""
        return state.p.grad

    def _dp_dt(self, state):
        """Something."""
        dp_dt = -1 * state.q.grad
        if self.mode == 'dissipate':
            dp_dt -= self._drag * state.p.grad
        elif self.mode == 'target':
            dp_dt += self.rate * (self.target - state.H.detach()) * state.p.grad
        return dp_dt

    def time_derivatives(self, state, create_graph=False):
        """Something."""
        if create_graph:
            return self._dq_dt(state), self._dp_dt(state)
        with torch.no_grad():
            return self._dq_dt(state), self._dp_dt(state)