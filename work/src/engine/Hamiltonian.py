"""Something."""
import functools


def hamiltonian_func(batch=False):
    """Decorate Hamiltonian Function."""

    def decorator_wrapper(func):

        if batch:
            raise Exception('Batch Evaluation Not Implemented Yet.')

            @functools.wraps(func)
            def wrapper(state, *args, **kwargs):
                pass

        else:
            @functools.wraps(func)
            def wrapper(state, *args, **kwargs):
                H = func(*state[:3], *args, **kwargs)
                state._state[3] = H
                return H

        return wrapper

    return decorator_wrapper