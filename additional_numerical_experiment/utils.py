from statsmodels.distributions.empirical_distribution import ECDF
import numpy as np

class EmpiricalCDF:
    def __init__(self, data):
        self.data = data
        self.ecdf = ECDF(data)

    def get_quantile_levels(self, x):
        return self.ecdf(x)
    
    def get_quantiles(self, q):
        return np.quantile(self.data, q=q, method='inverted_cdf')