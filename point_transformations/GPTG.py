from scipy.stats import gaussian_kde
import numpy as np
from scipy.optimize import minimize
from scipy import stats
from .histograms.histograms import histogram_pdf_quadratic as histogram

class GPTG:
    """A class for general point transforms of Gaussians"""
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
    
    def Gn_pdf_fit(self, params, x):
        y_samples = self.Gn(params)
        return histogram(y_samples, x, 200)
       
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
   
    def fit(self, initial_params, bounds=None, method='L-BFGS-B'):
        # Validate input
        if self.map is None:
            raise ValueError("No data provided. Set self.map before fitting.")
        f = self.Gn_pdf_fit
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