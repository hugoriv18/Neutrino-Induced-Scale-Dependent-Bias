import numpy as np
import camb
from camb import model
basepath = '/path/to/data/'

def compute_multipoles_model(CosmoParams, zeff = 0.8, kmax = 0.1):
    """
    Compute the power spectrum multipoles (P0, P2, P4) using the NISDB model.
    """
    # Cosmological parameters
    h, omega_b, omega_cdm, ns, log10As, Mnu, b1 = CosmoParams
    # Getting ready CAMB
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=h*100, ombh2=omega_b, omch2=omega_cdm, omk=0.0, 
                    tau=0.0561, standard_neutrino_neff=3.04, num_massive_neutrinos=1, 
                    neutrino_hierarchy='normal', mnu=Mnu, nnu=3.04)
    pars.InitPower.set_params(ns=ns, As=np.exp(log10As)/1e10)
    pars.set_matter_power(kmax=0.5, redshifts=[zeff, zeff + 0.001])
    results = camb.get_results(pars)
    # Get Power spectrum of matter
    k_full, _, PK = results.get_linear_matter_power_spectrum(hubble_units=True, k_hunit=True)
    Pk = PK[0,:]
    # Get Power spectrum of baryons+CDM at the two redshift (for computing the growth rate f_cb(k) later)
    _, _, PK_cb = results.get_linear_matter_power_spectrum(var1='delta_nonu', var2='delta_nonu', hubble_units=True, k_hunit=True)
    Pk_cb = PK_cb[0,:]
    Pk_cb_z2 = PK_cb[1,:]
    # Get transfer functions of cb and matter
    trans = results.get_matter_transfer_data()
    T_m = trans.transfer_data[model.Transfer_tot-1,:,0]
    T_cb = trans.transfer_data[model.Transfer_nonu-1,:,0]
    # Computing f_cb(k) around z = zeff
    a1 = 1.0 / (1.0 + zeff)
    a2 = 1.0 / (1.0 + zeff + 0.001)
    dlnP = np.log(Pk_cb_z2) - np.log(Pk_cb)
    dln_a = np.log(a2) - np.log(a1)
    f_cb = 0.5 * dlnP / dln_a
    # Get the bias of galaxies with respect to the total matter
    bm = b1*(T_cb/T_m)
    # Get the effective growth rate with respect to the total matter
    f_eff = f_cb*(T_cb / T_m)
    # Define mu values
    mu_vals = np.linspace(-1, 1, len(Pk))
    # Computing the power spectrum with RSDs 
    P_k_mu = np.array([bm**2*(1 + f_eff/bm * mu_i**2)**2 * Pk for mu_i in mu_vals])
    P_k_mu =  P_k_mu.T
    # mu_vals = np.linspace(-1, 1, P_k_mu.shape[1])  # Ensure match with P_k_mu's shape
    # Define Legendre polynomials of Legendre
    def L0(mu):
        return np.ones_like(mu)
    def L2(mu):
        return 0.5 * (3 * mu**2 - 1)
    def L4(mu):
        return (35 * mu**4 - 30 * mu**2 + 3) / 8
    # Compute Legendre polynomials
    L_ell0 = L0(mu_vals)
    L_ell2 = L2(mu_vals)
    L_ell4 = L4(mu_vals)
    # Multiply each row in P_k_mu by the Legendre polynomials
    integrand0 = P_k_mu * L_ell0[np.newaxis, :]
    integrand2 = P_k_mu * L_ell2[np.newaxis, :]
    integrand4 = P_k_mu * L_ell4[np.newaxis, :]
    # Integrate over μ using trapezoidal rule along axis=1 (μ-axis)
    integral0 = np.trapz(integrand0, x=mu_vals, axis=1)
    integral2 = np.trapz(integrand2, x=mu_vals, axis=1)
    integral4 = np.trapz(integrand4, x=mu_vals, axis=1)
    # Multipole prefactors
    P_ell0 = 0.5 * integral0
    P_ell2 = 2.5 * integral2
    P_ell4 = 4.5 * integral4
    # Interpolate Pk, Pk_cb and transfer funcions in the bining of k
    k_fn = (basepath + 'k.txt')
    k_arr = np.loadtxt(k_fn)
    mask = k_arr <= kmax
    k_arr = k_arr[mask]
    P_ell0  = np.interp(k_arr, k_full, P_ell0)
    P_ell2  = np.interp(k_arr, k_full, P_ell2)
    P_ell4  = np.interp(k_arr, k_full, P_ell4)
    # Return multipoles
    return k_arr, P_ell0, P_ell2, P_ell4
