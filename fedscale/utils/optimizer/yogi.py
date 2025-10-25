import torch
import numpy as np


class YoGi:
    def __init__(self, eta=0.5, tau=1e-3, beta=0.9, beta2=0.99):
        self.eta = eta
        self.tau = tau
        self.beta = beta

        self.v_t = []
        self.m_t = []
        self.beta2 = beta2

    def update(self, gradients):
        update_gradients = []
        if not self.v_t:
            self.v_t = [torch.full_like(g, self.tau) for g in gradients]
            self.m_t = [torch.full_like(g, 0.0) for g in gradients]

        for idx, gradient in enumerate(gradients):
            gradient_square = gradient**2

            self.m_t[idx] = self.beta * self.m_t[idx] + (1.0 - self.beta) * gradient

            self.v_t[idx] = self.v_t[idx] - (
                1.0 - self.beta2
            ) * gradient_square * torch.sign(self.v_t[idx] - gradient_square)
            yogi_learning_rate = self.eta / (torch.sqrt(self.v_t[idx]) + self.tau)

            update_gradients.append(-yogi_learning_rate * self.m_t[idx])

        if len(update_gradients) == 0:
            update_gradients = gradients

        return update_gradients

    def state_dict(self):
        """Return a serialisable snapshot of the optimizer state."""
        return {
            "eta": self.eta,
            "tau": self.tau,
            "beta": self.beta,
            "beta2": self.beta2,
            "m_t": [t.detach().cpu().clone() for t in self.m_t],
            "v_t": [t.detach().cpu().clone() for t in self.v_t],
        }

    def load_state_dict(self, state):
        """Restore optimizer state from `state_dict` output."""
        if not state:
            return
        self.eta = state.get("eta", self.eta)
        self.tau = state.get("tau", self.tau)
        self.beta = state.get("beta", self.beta)
        self.beta2 = state.get("beta2", self.beta2)
        self.m_t = [tensor.clone() for tensor in state.get("m_t", [])]
        self.v_t = [tensor.clone() for tensor in state.get("v_t", [])]
