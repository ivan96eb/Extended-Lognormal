import numpy as np 
from scipy.special import eval_legendre
from scipy.integrate import simpson
from numpy.polynomial.hermite import hermgauss
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
from .Gn import Gn

def get_gh_nodes_weights(n_nodes):
    t, w = hermgauss(n_nodes)
    y_nodes = np.sqrt(2.0) * t
    y_weights = w / np.sqrt(np.pi)
    return y_nodes, y_weights

def F_gauss_hermite_single(n, params_i, params_j, xi_g,
                           n_nodes=40,
                           precomputed=None):
    """
    Compute E[Gn(x_i)*Gn(x_j)] where (x_i,x_j) ~ N(0, Sigma) with corr xi_g.
    - n: label passed to Gn
    - params_i, params_j: Gn parameters
    - precomputed: optional tuple (y_nodes, y_weights) to reuse
    """
    if precomputed is None:
        y_nodes, y_weights = get_gh_nodes_weights(n_nodes)
    else:
        y_nodes, y_weights = precomputed

    cov = np.array([[1.0, xi_g],
                    [xi_g, 1.0]])
    L = np.linalg.cholesky(cov)  # map y -> x: x = L @ y

    # build tensor product nodes and weights (vectorized)
    YI, YJ = np.meshgrid(y_nodes, y_nodes, indexing='ij')
    WI, WJ = np.meshgrid(y_weights, y_weights, indexing='ij')

    # stack and map to x coordinates
    Ystack = np.stack([YI.ravel(), YJ.ravel()], axis=1)   # (M,2)
    Xstack = (L @ Ystack.T).T                             # (M,2)

    xi_vals = Xstack[:, 0]
    xj_vals = Xstack[:, 1]

    # Ensure Gn is vectorized and fast
    Gn_i = Gn(xi_vals, n, params_i)
    Gn_j = Gn(xj_vals, n, params_j)

    integrand = Gn_i * Gn_j * (WI * WJ).ravel()
    result = integrand.sum()
    return result

def build_lookup_table(n, params_i,params_j, xi_g_values, pre,nnodes=20):
    """Precompute F for various xi_g values"""
    results = []
    for xi_g in xi_g_values:
        result = F_gauss_hermite_single(n, params_i,params_j, xi_g, nnodes,pre)
        results.append(result)
    return np.array(results)


def C_NG_to_C_G(cl_NG, fitted_params, N_bins, N, 
                          xig_grid_size=100, quad_order=3, Nnodes=16, 
                          n_jobs=-1, v=0):
    """
    OPTIMIZED VERSION using Gauss-Legendre quadrature with precomputed Legendre polynomials.
    
    Parameters:
    -----------
    cl_NG : array
        Non-Gaussian power spectrum
    fitted_params : array
        Parameters for Gn function for each bin
    N_bins : int
        Number of redshift bins
    N : int
        Parameter for Gn function
    xig_grid_size : int, optional
        Number of points in xi_g interpolation grid (default: 100)
    quad_order : int, optional
        Gauss-Legendre quadrature order multiplier (default: 3)
        Total points = quad_order * lmax
    Nnodes : int, optional
        Number of Gauss-Hermite nodes for F integral (default: 16)
    n_jobs : int, optional
        Number of parallel jobs. -1 uses all cores (default: -1)
    v : int, optional
        Verbosity level for joblib (default: 0)
    
    Returns:
    --------
    cl_G : array
        Gaussian power spectrum
    """    
    cl_G = np.zeros_like(cl_NG)
    
    # Determine number of quadrature points needed
    lmax_cl = cl_NG.shape[-1] - 1
    n_quad = quad_order * lmax_cl
    
    # Get Gauss-Legendre quadrature points and weights
    mu, w = np.polynomial.legendre.leggauss(n_quad)
    
    # Precompute Legendre polynomials 
    ell_array = np.arange(lmax_cl + 1)
    P_ell = np.array([eval_legendre(ell, mu) for ell in ell_array])

    # Setup for lookup table
    xi_g_grid = np.linspace(-0.99999, 0.99999, xig_grid_size)
    pre = get_gh_nodes_weights(Nnodes)
    
    def process_pair_optimized(i, j):
        """Process a single (i,j) bin pair with precomputed Legendre polynomials"""
        params_i = fitted_params[i]
        params_j = fitted_params[j]
        
        # Build lookup table for F(xi_g)

        
        # C_l -> xi_NG using precomputed Legendre polynomials
        ell_col = ell_array[:, np.newaxis]
        arg = (2*ell_col + 1) * P_ell * cl_NG[i, j, :, np.newaxis]
        xi_NG = np.sum(arg, axis=0) / (4*np.pi)
        
        if N == '2':
            alpha_i, beta_i = params_i
            alpha_j, beta_j = params_j
            # ξ_NG = β_i × β_j × (exp(α_i×α_j×ξ_G) - 1)
            # => ξ_G = log(1 + ξ_NG/(β_i×β_j)) / (α_i×α_j)
            xi_G = np.log(1 + xi_NG / (beta_i * beta_j)) / (alpha_i * alpha_j)
        else:
            F_values = build_lookup_table(N, params_i, params_j, xi_g_grid, pre, Nnodes)
            F_to_xi_g = interp1d(F_values, xi_g_grid, kind='linear',
                                fill_value='extrapolate')
            xi_G = F_to_xi_g(xi_NG)
        
        # Backward transform: xi_G -> C_l using precomputed Legendre polynomials
        integrand = P_ell * xi_G[np.newaxis, :]
        clG_ij = 2 * np.pi * np.sum(w[np.newaxis, :] * integrand, axis=1)
        clG_ij[:2] = 1e-20
        
        return i, j, clG_ij
    
    # Generate list of pairs to process
    pairs = [(i, j) for i in range(N_bins) for j in range(i + 1)]
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=v)(
        delayed(process_pair_optimized)(i, j) for i, j in pairs
    )
    
    # Fill the matrix
    for i, j, clG_ij in results:
        cl_G[i, j] = clG_ij
        cl_G[j, i] = clG_ij
    
    # Force l=0,1 to be diagonal
    for l in [0, 1]:
        cl_G[:, :, l] = 1e-20 * np.eye(N_bins)
    
    return cl_G

def diagnose_cl_G(cl_G):
    problematic_ells = []
    min_eigval_per_ell = []

    for test_l in range(0, cl_G.shape[2]):  # Start from l=2 since you set l=0,1 to 1e-20
        eigvals = np.linalg.eigvalsh(cl_G[:,:,test_l])
        min_eigval = np.min(eigvals)
        min_eigval_per_ell.append(min_eigval)
        
        if min_eigval <= 0:
            problematic_ells.append(test_l)
            print(f"⚠️  l={test_l}: min eigenvalue = {min_eigval:.6e}")

    if len(problematic_ells) == 0:
        print("✓ All matrices are positive definite!")
        print(f"Minimum eigenvalue across all l: {np.min(min_eigval_per_ell):.6e}")
    else:
        print(f"\nFound {len(problematic_ells)} problematic ell values")
        print(f"First few: {problematic_ells[:10]}")