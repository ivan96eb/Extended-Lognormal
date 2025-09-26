from scipy.stats import gaussian_kde
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from scipy.interpolate import UnivariateSpline

def histogram_pdf_spline(y_samples, x_eval, bins=100, smoothing=0.1, k=3):
    """
    Histogram-based PDF estimation using spline interpolation
    Smoother than linear interpolation but still much faster than KDE
   
    Parameters:
    -----------
    y_samples : array-like
        The sample data
    x_eval : scalar or array-like
        Points at which to evaluate the PDF
    bins : int, default=100
        Number of histogram bins
    smoothing : float, default=0.1
        Smoothing parameter for spline (0 = interpolation, >0 = smoothing)
        Higher values create smoother curves
    k : int, default=3
        Degree of the spline (1=linear, 2=quadratic, 3=cubic, etc.)
        Must be <= 5
   
    Returns:
    --------
    float or array
        PDF values at x_eval points
    """
    # Create histogram
    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
   
    # Create spline interpolator
    # Use smoothing spline to get smoother curves than linear interpolation
    spline = UnivariateSpline(bin_centers, hist, k=k, s=smoothing)
   
    # Evaluate spline
    if np.isscalar(x_eval):
        result = float(spline(x_eval))
        # Ensure non-negative (splines can sometimes go slightly negative)
        return max(result, 1e-10)
    else:
        result = spline(x_eval)
        # Ensure non-negative values
        return np.maximum(result, 1e-10)


class GPTG:
    """A class for general point transforms of Gaussian"""
    def __init__(self, n, map=None, n_samples=500000):
        np.random.seed(42)
        self.x_samples = np.random.normal(0, 1, n_samples)
        self.n         = n
        self.map = map

    def Gn(self,params):
        if self.n == '2':
            alpha, beta = params
            y_samples = beta * np.exp(alpha * self.x_samples - 0.5 * alpha**2) - beta
            return y_samples
        
        elif self.n == '3':
            a, b, c = params
            arg = np.exp(a * self.x_samples - 0.5 * a**2) + b*self.x_samples + c
            n = 1/np.mean(arg)
            y_samples = n * arg - 1
            return y_samples      
         
        elif self.n == '4':
            a1, a2, t, x0 = params
            arg1 = np.exp(a1*self.x_samples - 0.5*a1**2)
            arg2 = (1 + np.exp((self.x_samples - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2 
            n = 1/np.mean(arg)
            y_samples = n * arg - 1
            return y_samples
        
        elif self.n == '5':
            a1, a2, b, t, x0 = params
            arg1 = np.exp(a1*self.x_samples - 0.5*a1**2) + b*self.x_samples
            arg2 = (1 + np.exp((self.x_samples - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2 
            n = 1/np.mean(arg)
            y_samples = n * arg - 1
            return y_samples

    def Gn_pdf(self, params):
        y_samples = self.Gn(params)
        kde = gaussian_kde(y_samples)
        def pdf_evaluator(x):
            return kde(x)[0] if np.isscalar(x) else kde(x)
       
        return pdf_evaluator
    
    def G2_pdf_fit(self, params, x):
        y_samples = self.Gn(params)
        return histogram_pdf_spline(y_samples, x, 1000)
       
    def cost_function(self, params, f):
        """Cost function for optimization"""
        try:
            pdf_values = f(params, self.map)
            # Handle case where pdf_values might be very small
            log_pdf = np.log(np.maximum(pdf_values, 1e-15))
            return -np.sum(log_pdf)
        except Exception as e:
            print(f"Error in cost function: {e}")
            return np.inf  # Return large value if evaluation fails
   
    def fit(self, fitting_func, initial_params, bounds=None, method='L-BFGS-B'):
        """
        Fit the specified G function to the data by minimizing the cost function.
       
        Parameters:
        -----------
        fitting_func : str
            Which G function to fit ('2', '3', '4', or '5')
        initial_params : array-like
            Initial parameter values for optimization
        bounds : list of tuples, optional
            Bounds for each parameter (min, max). If None, no bounds are applied.
        method : str, default='L-BFGS-B'
            Optimization method to use
       
        Returns:
        --------
        scipy.optimize.OptimizeResult
            The optimization result containing the fitted parameters
        """
        # Validate input
        if self.map is None:
            raise ValueError("No data provided. Set self.map before fitting.")
        
        # Select the appropriate fitting function
        if fitting_func == '2':
            f = self.G2_pdf_fit
        elif fitting_func == '3':
            raise NotImplementedError("G3 function not implemented yet")
        elif fitting_func == '4':
            raise NotImplementedError("G4 function not implemented yet")
        elif fitting_func == '5':
            raise NotImplementedError("G5 function not implemented yet")
        else:
            raise ValueError(f"Unknown fitting function: {fitting_func}. Must be '2', '3', '4', or '5'")
       
        # Minimize the cost function
        try:
            result = minimize(
                fun=self.cost_function,
                x0=initial_params,
                args=(f,),
                method=method,
                bounds=bounds
            )
            return result
        except Exception as e:
            print(f"Optimization failed: {e}")
            raise