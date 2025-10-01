import numpy as np
from scipy.optimize import minimize

def smart_initialization(x_data, y_data, n):
    """
    Generate smart initial parameters based on data characteristics
    
    Parameters:
    -----------
    x_data : array of x values
    y_data : array of y values  
    n : str, which G function ('2', '3', '4', '5')
    
    Returns:
    --------
    initial_params : array of initial parameter values
    """
    
    # Basic statistics
    y_mean = np.mean(y_data)
    y_std = np.std(y_data)
    y_skew = np.mean(((y_data - y_mean) / y_std)**3) if y_std > 0 else 0
    
    # Check if there's asymmetry (where does y cross zero or change behavior)
    if len(y_data[y_data > 0]) > 0 and len(y_data[y_data < 0]) > 0:
        # Find approximate location where y transitions
        zero_crossing_idx = np.argmin(np.abs(y_data))
        x0_estimate = x_data[zero_crossing_idx]
    else:
        x0_estimate = 0.0
    
    # Estimate exponential growth rate from tails
    # Use the relationship: if y ~ exp(a*x), then log(y+offset) ~ a*x
    # Focus on positive tail
    upper_mask = x_data > 1.0
    if np.sum(upper_mask) > 5:
        x_upper = x_data[upper_mask]
        y_upper = y_data[upper_mask]
        # Shift y to be positive
        y_shifted = y_upper - np.min(y_upper) + 1
        try:
            # Linear fit in log space
            coeffs = np.polyfit(x_upper, np.log(y_shifted), 1)
            a_estimate = coeffs[0]
            # Clamp to reasonable range
            a_estimate = np.clip(a_estimate, 0.1, 3.0)
        except:
            a_estimate = 0.5
    else:
        a_estimate = 0.5
    
    # Estimate linear trend from center region
    center_mask = (x_data > -1.0) & (x_data < 1.0)
    if np.sum(center_mask) > 3:
        x_center = x_data[center_mask]
        y_center = y_data[center_mask]
        coeffs = np.polyfit(x_center, y_center, 1)
        b_estimate = coeffs[0]
        c_estimate = coeffs[1]
    else:
        b_estimate = 0.0
        c_estimate = 0.0
    
    # Generate initial parameters based on model type
    if n == '2':
        # G2: alpha, beta
        # beta controls scale, alpha controls exponential rate
        beta_estimate = y_std * 0.5  # Scale parameter
        alpha_estimate = a_estimate
        return np.array([alpha_estimate, beta_estimate])
    
    elif n == '3':
        # G3: a, b, c
        # a: exponential term, b: linear term, c: constant
        return np.array([a_estimate, b_estimate * 0.5, c_estimate * 0.5])
    
    elif n == '4':
        # G4: a1, a2, t, x0
        # a1: left tail behavior, a2: right tail behavior
        # t: transition steepness, x0: transition location
        
        # If skewed right, make a2 > a1
        if y_skew > 0:
            a1_estimate = a_estimate * 0.7
            a2_estimate = a_estimate * 1.3
        else:
            a1_estimate = a_estimate * 1.3
            a2_estimate = a_estimate * 0.7
        
        t_estimate = 1.0  # Moderate transition steepness
        
        return np.array([a1_estimate, a2_estimate, t_estimate, x0_estimate])
    
    elif n == '5':
        # G5: a1, a2, b, t, x0
        # Combines linear term with asymmetric exponential
        
        if y_skew > 0:
            a1_estimate = a_estimate * 0.7
            a2_estimate = a_estimate * 1.3
        else:
            a1_estimate = a_estimate * 1.3
            a2_estimate = a_estimate * 0.7
        
        t_estimate = 1.0
        
        return np.array([a1_estimate, a2_estimate, b_estimate * 0.5, t_estimate, x0_estimate])
    
    else:
        raise ValueError(f"Unknown model type: {n}")


def fit_gn_to_data(x_data, y_data, n, initial_params=None, n_samples=500000):
    """
    Fit a Gn transformation to (x, y) data points
    
    Parameters:
    -----------
    x_data : array of x values (should be approximately standard normal)
    y_data : array of y values (transformed data)
    n : str, which G function to use ('2', '3', '4', '5')
    initial_params : initial parameter guess (if None, uses smart initialization)
    n_samples : number of samples for computing normalization constant
    
    Returns:
    --------
    result : optimization result with fitted parameters
    """
    
    # Use smart initialization if not provided
    if initial_params is None:
        initial_params = smart_initialization(x_data, y_data, n)
        print(f"Smart initialization for G{n}: {initial_params}")
    
    # Generate standard normal samples for normalization computation
    np.random.seed(42)
    x_samples = np.random.normal(0, 1, n_samples)
    
    def compute_normalization(params):
        """Compute n such that E[n*arg - 1] = 0"""
        if n == '2':
            return None
        elif n == '3':
            a, b, c = params
            arg = np.exp(a * x_samples - 0.5 * a**2) + b*x_samples + c
            return 1/np.mean(arg)
        elif n == '4':
            a1, a2, t, x0 = params
            arg1 = np.exp(a1*x_samples - 0.5*a1**2)
            arg2 = (1 + np.exp((x_samples - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2
            return 1/np.mean(arg)
        elif n == '5':
            a1, a2, b, t, x0 = params
            arg1 = np.exp(a1*x_samples - 0.5*a1**2) + b*x_samples
            arg2 = (1 + np.exp((x_samples - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2
            return 1/np.mean(arg)
    
    def evaluate_gn(params, x):
        """Evaluate Gn at specific x values"""
        if n == '2':
            alpha, beta = params
            return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta
        elif n == '3':
            a, b, c = params
            arg = np.exp(a * x - 0.5 * a**2) + b*x + c
            norm = compute_normalization(params)
            return norm * arg - 1
        elif n == '4':
            a1, a2, t, x0 = params
            arg1 = np.exp(a1*x - 0.5*a1**2)
            arg2 = (1 + np.exp((x - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2
            norm = compute_normalization(params)
            return norm * arg - 1
        elif n == '5':
            a1, a2, b, t, x0 = params
            arg1 = np.exp(a1*x - 0.5*a1**2) + b*x
            arg2 = (1 + np.exp((x - x0)*t))**((a2-a1)/t)
            arg = arg1 * arg2
            norm = compute_normalization(params)
            return norm * arg - 1
    
    def cost_function(params):
        """Least squares cost"""
        try:
            y_pred = evaluate_gn(params, x_data)
            return np.sum((y_pred - y_data)**2)
        except:
            return np.inf
    
    result = minimize(
        fun=cost_function,
        x0=initial_params,
        method='BFGS'
    )
    
    return result