import numpy as np 
import healpy as hp
import time
from multiprocessing import Pool
from .transform_cls import C_NG_to_C_G,diagnose_cl_G
from .mocker import get_y_maps,get_kappa,get_kappa_pixwin

def correct_mult_cl(cl,A):
    N_bins = cl.shape[0]
    cl_correct = np.zeros_like(cl)
    for i in range(N_bins):
        for j in range(i+1):
            cl_ij = np.sqrt(A[i]*A[j])*cl[i,j]
            cl_correct[i,j] = cl_ij
            cl_correct[j,i] = cl_ij
    return cl_correct

def cl_mock_avg(cl_NG,cl_G,fitted_params,pixwin,pixwin_ell_filter,N,N_mocks=200,Nside=256,N_bins=4,gen_lmax=767):
    cl_arr  = np.zeros((N_mocks,N_bins,N_bins,gen_lmax+1))
    for mock in range(N_mocks):
        if mock % 50 == 0:
            print(f'Working on mock {mock}')
        y_maps, _      = get_y_maps(cl_G, Nside, N_bins, gen_lmax)
        kappa_mock     = get_kappa_pixwin(y_maps, N_bins, N, fitted_params,Nside,pixwin_ell_filter)
        for i in range(N_bins):
            for j in range(i+1):
                c_ij = hp.anafast(kappa_mock[i], kappa_mock[j], lmax=gen_lmax)
                cl_arr[mock,i,j] = c_ij
                cl_arr[mock,j,i] = c_ij
    perdiff_arr = np.zeros_like(cl_arr)
    for mock in range(N_mocks):
        perdiff_arr[mock]=(cl_arr[mock]/(cl_NG*pixwin**2)) 
    average_ratio = np.average(perdiff_arr,axis=0)
    return average_ratio

def process_single_mock(args):
    """Process a single mock iteration"""
    mock, cl_G, fitted_params, pixwin, pixwin_ell_filter, N, Nside, N_bins, gen_lmax = args
    
    y_maps, _ = get_y_maps(cl_G, Nside, N_bins, gen_lmax)
    kappa_mock = get_kappa_pixwin(y_maps, N_bins, N, fitted_params, Nside, pixwin_ell_filter)
    
    cl_single = np.zeros((N_bins, N_bins, gen_lmax+1))
    for i in range(N_bins):
        for j in range(i+1):
            c_ij = hp.anafast(kappa_mock[i], kappa_mock[j], lmax=gen_lmax)
            cl_single[i, j] = c_ij
            cl_single[j, i] = c_ij
    
    return cl_single

def cl_mock_avg_parallel(cl_NG, cl_G, fitted_params, pixwin, pixwin_ell_filter, N, N_mocks=200, Nside=256, N_bins=4, gen_lmax=767, n_processes=8):
    """
    Parallel version of cl_mock_avg
    
    Parameters:
    -----------
    n_processes : int, optional
        Number of processes to use. If None, uses all available CPUs.
    """
    # Prepare arguments for each mock
    args_list = [(mock, cl_G, fitted_params, pixwin, pixwin_ell_filter, N, Nside, N_bins, gen_lmax) 
                 for mock in range(N_mocks)]
    
    # Process mocks in parallel
    with Pool(processes=n_processes) as pool:
        cl_list = pool.map(process_single_mock, args_list)
    
    # Convert list to array
    cl_arr = np.array(cl_list)
    
    # Calculate perdiff and average
    perdiff_arr = cl_arr / (cl_NG * pixwin**2)
    average_ratio = np.average(perdiff_arr, axis=0)
    
    return average_ratio

def Acoeff(average_ratio):
    Nbins = average_ratio.shape[0]
    beta = np.zeros(Nbins)
    for i in range(Nbins):
        beta[i]=np.average(average_ratio[i, i, 10:300])
    return 1/beta

def debiaser(cl_NG,N,params,pixwin,pixwinellfilter,N_iter=3,Nmocks=200):
    Nbins = cl_NG.shape[0]
    cl_NG_corr = cl_NG 
    for i in range(N_iter):
        print(f'Iteration {i}')
        cl_G       = C_NG_to_C_G(cl_NG_corr,params,Nbins,N)
        diagnose_cl_G(cl_G)
        avg_ratio = cl_mock_avg(cl_NG,cl_G,params,pixwin,pixwinellfilter,N,Nmocks)
        A         = Acoeff(avg_ratio)
        print('beta=',1/A)
        cl_NG_corr = correct_mult_cl(cl_NG_corr,A)
    return cl_NG_corr