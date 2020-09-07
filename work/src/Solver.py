"""Something."""
from . import StateList


class Solver(object):
    """Something."""

    def __init__(self, drag=0, target=0, rate=0):
        """Something."""
        self._drag = drag
        self._target = target
        self._rate = rate
        self._epsilon = 1e-6

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

    def _dq_dt(self, state):
        """Something."""
        return state.p.grad

    def _dp_dt(self, state, H=None):
        """Something."""
        dp_dt = -1 * state.q.grad
        if abs(self._drag) > self._epsilon:
            dp_dt -= self._drag * state.p.grad
        elif abs(self.rate) > self._epsilon:
            dp_dt += self.rate * (self.target - H.detach()) * state.p.grad
        return dp_dt

    def time_derivatives(self, state, H=None):
        """Something."""
        return self._dq_dt(state), self._dp_dt(state, H)

    def trajectory(self, initial_state, hamiltonian, num_steps, time_step, **kwargs):
        """Something."""
        trajectory = StateList(shape=(num_steps, *initial_state.q.shape))
        state = initial_state
        for index in range(num_steps):
            trajectory[index] = state
            H = hamiltonian(state, **kwargs)
            H.backward()
            dq_dt, dp_dt = self.time_derivatives(state, H)
            state = state.step(dq_dt, dp_dt, time_step)
        return trajectory