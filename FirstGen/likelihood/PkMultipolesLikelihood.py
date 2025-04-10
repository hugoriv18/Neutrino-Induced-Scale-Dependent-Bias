import numpy as np
from cobaya.likelihood import Likelihood
from scipy.linalg import cho_factor, cho_solve
from .get_model import compute_multipoles_model

basepath = '/path/to/data/'

class PkMultipolesLikelihood(Likelihood):
    """
    Cobaya-compatible likelihood for power spectrum multipoles (P0, P2, P4).
    Uses a fixed data vector and covariance matrix.
    """
    required_parameters = ["h", "omega_b", "omega_cdm", "ns", "log10As", "Mnu", "b1"]
    # These will be passed from YAML
    kmax: float
    zeff: float

    def initialize(self):
        # Set fixed redshift
        self.zeff = float(self.zeff)
        # kmax mode
        self.kmax = float(self.kmax)
        # Fixed file paths to data
        k_full = np.loadtxt(basepath + "k.txt")
        Pk0_full = np.loadtxt(basepath + "Pk_0.txt")[:, 0]
        Pk2_full = np.loadtxt(basepath + "Pk_2.txt")[:, 0]
        Pk4_full = np.loadtxt(basepath + "Pk_4.txt")[:, 0]
        cov_full = np.loadtxt(basepath + "cov.txt")
        # Cut the k modes
        mask = k_full <= self.kmax
        self.k_arr = k_full[mask] # Shape: (N,)
        # Cut the multipoles
        Pk0 = Pk0_full[mask]
        Pk2 = Pk2_full[mask]
        Pk4 = Pk4_full[mask]
        self.data_vector = np.concatenate([Pk0, Pk2, Pk4]) # Shape: (3N,)
        # Cut the matrix
        kcenters = k_full  # x-axis values used to index the covariance matrix
        ellscov = (0, 2, 4)  # multipole orders in the full data vector
        klim = {ell: (k_full[0], self.kmax, 0.005) for ell in ellscov}
        # Apply the cut
        # Cut the cov matrix
        def cut_matrix(cov, xcov, ellscov, xlim):
            assert len(cov) == len(xcov) * len(ellscov), 'Input matrix has size {}, different than {} x {}'.format(len(cov), len(xcov), len(ellscov))
            indices = []
            for ell, xlim in xlim.items():
                index = ellscov.index(ell) * len(xcov) + np.arange(len(xcov))
                index = index[(xcov >= xlim[0]) & (xcov <= xlim[1])]
                indices.append(index)
            indices = np.concatenate(indices, axis=0)
            return cov[np.ix_(indices, indices)]
        self.cov = cut_matrix(cov_full, kcenters, ellscov, klim) # Shape: (3N, 3N)
        # Precompute Cholesky decomposition
        self.cov = 0.5 * (self.cov + self.cov.T)
        self.cov += 1e-8 * np.eye(len(self.cov))
        self.cov_chol = cho_factor(self.cov, lower=True)
        print(">>> PkMultipolesLikelihood initialized")

    def get_requirements(self):
        return {}

    def logp(self, **params_values):
        # Build parameter vector in correct order
        theta = [
            params_values["h"],
            params_values["omega_b"],
            params_values["omega_cdm"],
            params_values["ns"],
            params_values["log10As"],
            params_values["Mnu"],
            params_values["b1"]
        ]
        # Model prediction
        k_model, P0_model, P2_model, P4_model = compute_multipoles_model(theta, self.zeff, self.kmax)
        # Build full model vector (should be same length as data_vector)
        model_vector = np.concatenate([P0_model, P2_model, P4_model])
        # Compute residuals and chi^2
        delta = self.data_vector - model_vector
        chi2 = np.dot(delta, cho_solve(self.cov_chol, delta))
        return -0.5 * chi2
