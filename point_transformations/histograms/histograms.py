import numpy as np
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

def histogram_pdf_spline(y_samples, x_eval, bins, smoothing=0.1, k=3):
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

def histogram_pdf_linear(y_samples, x_eval, bins):
    """
    Histogram-based PDF estimation using linear interpolation
    Fast and simple, but can be jagged
    """
    # Create histogram
    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Linear interpolation
    interpolator = interp1d(bin_centers, hist, kind='linear', 
                           bounds_error=False, fill_value=1e-10)
    
    if np.isscalar(x_eval):
        return float(interpolator(x_eval))
    else:
        return interpolator(x_eval)

def histogram_pdf_quadratic(y_samples, x_eval, bins):
    """
    Histogram-based PDF estimation using quadratic interpolation
    Good balance of smoothness and stability
    """
    # Create histogram
    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Quadratic interpolation
    interpolator = interp1d(bin_centers, hist, kind='quadratic', 
                           bounds_error=False, fill_value=1e-10)
    
    if np.isscalar(x_eval):
        result = float(interpolator(x_eval))
        return max(result, 1e-10)  # Ensure non-negative
    else:
        result = interpolator(x_eval)
        return np.maximum(result, 1e-10)
    
def histogram_pdf_cubic(y_samples, x_eval, bins):
    """
    Histogram-based PDF estimation using cubic interpolation
    Smoothest but can oscillate with sparse data
    """
    # Create histogram
    hist, bin_edges = np.histogram(y_samples, bins=bins, density=True)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    # Cubic interpolation
    interpolator = interp1d(bin_centers, hist, kind='cubic', 
                           bounds_error=False, fill_value=1e-10)
    
    if np.isscalar(x_eval):
        result = float(interpolator(x_eval))
        return max(result, 1e-10)  # Ensure non-negative
    else:
        result = interpolator(x_eval)
        return np.maximum(result, 1e-10)