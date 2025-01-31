import numpy as np

import torch
import torch.nn as nn


## PyTorch
class KSparseTorch(nn.Module):
    """
    K-sparse PyTorch layer.

    Based on https://gist.github.com/harryscholes/ed3539ab21ad34dc24b63adc715a97e0 .

    Parameters
    ----------
    sparsity_levels
        np.ndarray, sparsity levels per epoch calculated by `calculate_sparsity_levels`
    """

    def __init__(self, sparsity_levels):
        super(KSparseTorch, self).__init__()
        self.sparsity_levels = sparsity_levels
        self.k = nn.Parameter(torch.tensor(self.sparsity_levels[0]), requires_grad=False)

    def forward(self, inputs):
        """Make inputs sparse."""
        kth_largest = torch.kthvalue(inputs.view(inputs.size(0), -1), self.k.item(), dim=-1, keepdim=True)[0].view(-1, 1, 1, 1)
        sparse_inputs = inputs * (inputs >= kth_largest).float()
        return sparse_inputs

    def extra_repr(self):
        return f'k={self.k.item()}'

class UpdateSparsityLevelTorch:
    """Update sparsity level at the beginning of each epoch."""

    def __init__(self, model):
        self.model = model

    def on_epoch_begin(self, epoch):
        """
        Update sparsity level.
        """
        try:
            layer = next(m for m in self.model.modules() if isinstance(m, KSparseTorch))
            layer.k.data = torch.tensor(layer.sparsity_levels[epoch])
            print(f"set k to {layer.k.item()}")
        except StopIteration:
            pass

def calculate_sparsity_levels_torch(initial_sparsity, final_sparsity, n_epochs):
    """Calculate sparsity levels per epoch.

    # Arguments
        initial_sparsity: int
        final_sparsity: int
        n_epochs: int
    """
    return np.hstack(
        (
            np.linspace(initial_sparsity, final_sparsity, n_epochs // 2, dtype=np.uint),
            np.repeat(final_sparsity, (n_epochs // 2) + 1),
        )
    )[:n_epochs]