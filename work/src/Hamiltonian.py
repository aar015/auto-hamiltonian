"""Something."""
import functools
from . import State, StateList


def hamiltonian_func(func):
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