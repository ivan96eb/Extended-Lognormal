import numpy as np 

def Gn(x, n, params, n_samples=500000):
    """
    Evaluate Gn transformation at given x values
    
    Parameters:
    -----------
    x : array of x values (standard normal inputs)
    n : str, which G function to use ('2', '3', '4', '5')
    params : array of parameters for the transformation
    n_samples : number of samples for computing normalization constant
    
    Returns:
    --------
    y : transformed values
    """
    
    # Generate standard normal samples for normalization computation
    # Use a local RandomState that doesn't affect global np.random
    rng = np.random.RandomState(42)  # ← This creates an isolated random generator
    x_samples = rng.normal(0, 1, n_samples)  # ← Use rng, not np.random
    
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
    
    # Evaluate transformation
    if n == '2':
        alpha, beta = params
        return beta * np.exp(alpha * x - 0.5 * alpha**2) - beta
    
    elif n == '3':
        a, b, c = params
        arg = np.exp(a * x - 0.5 * a**2) + b*x + c
        norm = 1/(1+c)
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
    
    else:
        raise ValueError(f"Unknown model type: {n}")