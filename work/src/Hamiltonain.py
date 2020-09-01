"""Something."""
import torch
from abc import ABC, abstractmethod


class Hamiltonian(ABC):
    """Something."""

    @abstractmethod
    def evaluate(self, q, p):
        """Something."""
        pass

    def step(self, q, p, time_step, **kwargs):
        """Something."""
        if q.grad is not None:
            q.grad.zero_()
        if p.grad is not None:
            p.grad.zero_()
        hamiltonian = self.evaluate(q, p, **kwargs)
        hamiltonian.backward()
        next_q = (q + time_step * p.grad).detach().requires_grad_(True)
        next_p = (p - time_step * q.grad).detach().requires_grad_(True)
        return next_q, next_p

    def predict(q, p, time, time_step):
        """Something."""
        num_steps = int(sim_time / time_step)
        for index in range(1, num_steps):
            q, p = self.step(q, p, time_step, **kwargs)
        return q, p

    def simulate(self, q, p, sim_time, time_step, **kwargs):
        """Something."""
        num_steps = int(sim_time / time_step)
        time = torch.linspace(0, sim_time, num_steps, device=q.device)
        q_hist = torch.empty((num_steps, *q.shape), device=q.device)
        p_hist = torch.empty((num_steps, *p.shape), device=p.device)
        hamiltonian_hist = torch.empty(num_steps, device=q.device)
        q_hist[0] = q.detach()
        p_hist[0] = p.detach()
        with torch.no_grad():
            hamiltonian_hist[0] = self.evaluate(q, p, **kwargs)
        for index in range(1, num_steps):
            q, p = self.step(q, p, time_step, **kwargs)
            q_hist[index] = q.detach()
            p_hist[index] = p.detach()
            with torch.no_grad():
                hamiltonian_hist[index] = self.evaluate(q, p, **kwargs)
        return time, hamiltonian_hist, q_hist, p_hist
