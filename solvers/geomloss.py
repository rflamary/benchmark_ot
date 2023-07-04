from benchopt import BaseSolver, safe_import_context
from benchopt.stopping_criterion import SufficientProgressCriterion

# Protect the import with `safe_import_context()`. This allows:
# - skipping import to speed up autocompletion in CLI.
# - getting requirements info when all dependencies are not installed.
with safe_import_context() as import_ctx:
    import jax.numpy as jnp
    import numpy as np

    # Import Geomloss which is based on pytorch.
    import torch
    from geomloss import SamplesLoss
    


class Solver(BaseSolver):

    # Name to select the solver in the CLI and to display the results.
    name = 'GeomLoss'

    install_cmd = 'conda'
    requirements = ['torch', 'pykeops', 'pip:geomloss', 'ott-jax']

    # List of parameters for the solver. The benchmark will consider
    # the cross product for each key in the dictionary.
    parameters = {
        'blur': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
        'use_gpu': [True, False],
    }

    stopping_criterion = SufficientProgressCriterion(patience=10, eps=1e-12)

    def skip(self, **kwargs):
        # we skip the solver if use_gpu is True and no GPU is available
        if self.use_gpu and not torch.cuda.is_available():
            return True, "No GPU available"
        return False, None

    def set_objective(self, x, a, y, b):
        # Create a ott problem based on jax to compute the output \
        # of the solver.

        # Store the problem in torch to use GeomLoss.
        # Use the GPU when it is available.
        device = 'cuda' if self.use_gpu else 'cpu'

        self.x, self.a, self.y, self.b = [
            torch.from_numpy(t).float().to(device=device)
            for t in (x, a, y, b)
        ]

    def get_next(self, n_iter):
        return n_iter + 1

    def run(self, n_iter):
        # content of `sinkhorn_tensorized` from
        # https://github.com/jeanfeydy/geomloss/blob/main/geomloss/sinkhorn_samples.py
        x, y, a, b = self.x, self.y, self.a, self.b

        N, D = x.shape
        M, Dp = y.shape
        assert D == Dp
        assert a.shape == (N,)
        assert b.shape == (M,)

        diameter = 3
        
        if True:
            scaling = np.exp((np.log(self.blur) - np.log(diameter)) / (n_iter + 2))
        else:
            scaling = 0.9

        OT_solver = SamplesLoss(
            "sinkhorn",
            p=2,
            blur=self.blur,
            scaling=scaling,
            debias=False,
            potentials=True,
            verbose=True,
        )

        f_ba, g_ab = OT_solver(a, x, b, y)

        self.f_ba = f_ba.view_as(a)
        self.g_ab = g_ab.view_as(b)

        assert self.f_ba.shape == (N,)
        assert self.g_ab.shape == (M,)


    def get_result(self):
        # Return the result from one optimization run.
        x2_i = (self.x**2).sum(dim=1)
        y2_j = (self.y**2).sum(dim=1)
        C_ij = self.x @ self.y.T
        C_ij = (x2_i[:, None] + y2_j[None, :]) / 2 - C_ij

        f = self.f_ba
        g = self.g_ab

        K_ij = ((f[:,None] + g[None,:] - C_ij) / self.blur).exp()
        P_ij = K_ij * (self.a[:,None] * self.b[None,:])
        
        return P_ij.detach().cpu().numpy()
