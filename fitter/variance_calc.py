import numpy as np 
from numpy.polynomial.hermite import hermgauss
from .Gn import Gn

def get_gh_nodes_weights(n_nodes):
    t, w = hermgauss(n_nodes)
    y_nodes = np.sqrt(2.0) * t
    y_weights = w / np.sqrt(np.pi)
    return y_nodes, y_weights

def integrand(x,N,params):
    return Gn(x,N,params)**2

def var_pdf(N,params,n_nodes=10):
    quads,weights = get_gh_nodes_weights(n_nodes)
    integral = weights*integrand(quads,N,params) 
    return integral.sum()

def var_cl(cl):
    ell    = np.arange(cl.shape[0])
    return np.sum((2*ell+1)*cl)/(4*np.pi)

def compute_A(cl,N,fitted_params,N_bins):
    variance_pdf = np.array([var_pdf(N,fitted_params[i]) for i in range(N_bins)])
    variance_cl  = np.array([var_cl(cl[i,i]) for i in range(N_bins)])
    return (variance_pdf - variance_cl )

def compute_alpha_ij(Ai,Aj,c_ii,c_jj):
    arg1 = Ai/c_ii
    arg2 = Aj/c_jj
    arg3 = (Ai*Aj)/(c_ii*c_jj)
    return np.sqrt(1+arg1+arg2+arg3)

def correct_cl(cl, N, fitted_params, N_bins, A=None,diag_only=True):
    cl_corrected = cl.copy()  # Start with original
    #if A == None:
    #    A = compute_A(cl, N, fitted_params, N_bins)
    
    if diag_only:
        # Only correct auto-spectra
        for i in range(N_bins):
            alpha_ii = compute_alpha_ij(A[i], A[i], cl[i,i], cl[i,i])
            cl_corrected[i,i] = alpha_ii * cl[i,i]
    else:
        # Correct everything (original approach)
        for i in range(N_bins):
            for j in range(i+1):
                alpha_ij = compute_alpha_ij(A[i], A[j], cl[i,i], cl[j,j])
                cl_corrected_ij = alpha_ij * cl[i,j]
                cl_corrected[i,j] = cl_corrected_ij
                cl_corrected[j,i] = cl_corrected_ij
    
    for l in [0, 1]:
        cl_corrected[:, :, l] = 1e-20 * np.eye(N_bins)
    
    return cl_corrected



