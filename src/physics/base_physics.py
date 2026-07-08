import torch.nn as nn

class Physics(nn.Module):
    """
    Abstract Base Class for Physics (PDEs).
    Inherits from nn.Module to support learnable parameters (Inverse Problems).
    """
    def __init__(self):
        super().__init__()

    def residual(self, model, x, t):
        """
        Compute the PDE residual.
        :param model: The Neural Network (FCN)
        :param x: Spatial coordinate (N, 1)
        :param t: Time coordinate (N, 1)
        :return: Residual tensor (N, 1)
        """
        raise NotImplementedError

    def analytical_solution(self, x, t):
        """
        Return analytical solution if available.
        """
        return None

    def get_param_errors(self):
        """
        Return dictionary of parameter errors (for logging).
        """
        return {}

    def sample_pde(self, n, device):
        """Sample n collocation points for PDE residual"""
        raise NotImplementedError

    def sample_ic(self, n, device):
        """Sample n points for Initial Condition"""
        raise NotImplementedError

    def sample_bc(self, n, device):
        """Sample n points for Boundary Condition"""
        raise NotImplementedError

    def sample_data(self, n, device):
        """Sample n points for Data Loss (Inverse Problem)"""
        return None, None # Optional default

    def bc_residual(self, model, *args):
        """
        Compute BC residual. Default is Dirichlet (u=0).
        For Robin/Neumann, override this.
        Returns: Residual tensor to be squared.
        """
        u = model(*args)
        return u
