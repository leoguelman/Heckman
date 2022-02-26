"""
Created on Mon Dec 20 14:41:29 2021
@author: lguelman

Missing Data Experiments
"""

import numpy as np
import pandas as pd
import pystan
import multiprocessing
from multivariate_laplace import multivariate_laplace
from scipy.stats import multivariate_t


class MissingDataExperiments:
    """
    
    Generate sythetic missing data 
    
    Parameters
    ----------
    N : number of observations
    alpha: intercept outcome model
    beta: slope outcome model
    delta: intercept missing model
    gamma: slope missing model
    sigma_y: sigma outcome model 
    rho_xz: correlation between features of outcome and pattern of missigness 
    rho_yd: correlation between errors in outcome and pattern of missigness
    m_error: Add measurement error to observed outcome?
    m_error_sigma: sigma value for measurement error (applicable if `m_error=True`).
    distribution: joint distribution between errors in outcome and pattern of missigness.
                  Default is the multivariate normal ('MVN'). Other options include douple
                  exponential/Laplace distribution('Laplace'), and Student ('Student').
    nonlinear_z: If True, the covariate Z enters non-linearly in the true missing data model. 
                  
    """

    def __init__(self, N: int = 100, 
                 alpha=None, beta=None, delta=None, gamma=None, sigma_y=None,
                 rho_xz: float = 0.0, rho_yd: float = 0.0 , m_error: bool = False, m_error_sigma: float = None,
                 distribution: str = 'MVN', nonlinear_z: bool = False,
                 seed: int = 42):
        
        self.N = N
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.gamma = gamma
        self.sigma_y = sigma_y
        self.rho_xz = rho_xz
        self.rho_yd = rho_yd
        self.m_error = m_error
        self.m_error_sigma = m_error_sigma
        self.distribution = distribution
        self.nonlinear_z = nonlinear_z
        self.seed = seed
        
    def __repr__(self):
            
            items = ("%s = %r" % (k, v) for k, v in self.__dict__.items())
            return "<%s: {%s}>" % (self.__class__.__name__, ', '.join(items))

    def generate_data(self):
        
        np.random.seed(self.seed)
        
        cov_xz = np.array([[1,self.rho_xz],[self.rho_xz, 1]])
        x, z = np.random.multivariate_normal(np.array([0,0]), cov_xz, self.N).T
        
        if self.alpha is None:
            self.alpha = np.random.normal(0.0, 1.0, size=1)    
        
        if self.beta is None:
            self.beta = np.random.normal(0.0, 1.0, size=(1, 1))
        else:
            self.beta = np.expand_dims([self.beta], axis=1)
        
        if self.delta is None:
            self.delta = np.random.normal(0.0, 1.0, size=1)    
        
        if self.gamma is None:
            self.gamma = np.random.normal(0.0, 1.0, size=(1, 1))
        else:
            self.gamma = np.expand_dims([self.gamma], axis=1)

        if self.sigma_y is None:
            self.sigma_y = np.random.exponential(1, size=1)
        else:
            self.sigma_y = np.array([self.sigma_y])

        mu_y = self.alpha + np.matmul(np.expand_dims(x, axis=1), self.beta)

        if self.nonlinear_z:
            mu_d = self.delta + np.matmul(np.expand_dims(z**2, axis=1), self.gamma)
        else:
            mu_d = self.delta + np.matmul(np.expand_dims(z, axis=1), self.gamma)
        mu_yd = np.hstack([mu_y, mu_d])
        cov_yd = np.array([[self.sigma_y[0]**2, self.rho_yd * self.sigma_y[0]],
                           [self.rho_yd * self.sigma_y[0],1]])

        y = np.empty([self.N])
        d = np.empty([self.N])
        for i in range(self.N):
            
            if self.distribution == 'MVN':
                y[i], d[i] = np.random.multivariate_normal(mu_yd[i], cov_yd, size = 1).T 
            
            elif self.distribution == 'Laplace':
                y[i], d[i] = multivariate_laplace.rvs(mu_yd[i], cov_yd, size = 1).T 
                
            elif self.distribution == 'Student':
                y[i], d[i] = np.expand_dims(multivariate_t.rvs(mu_yd[i], cov_yd, size = 1).T, 1)
                
        if self.m_error:
            if self.m_error_sigma is None:
                raise ValueError("m_error_sigma value must be provided when m_error=True.")
            else:
                y_obs = np.empty([self.N])
                for i in range(self.N):
                    y_obs[i] = np.random.normal(y[i], self.m_error_sigma)
                
        else:
            y_obs = y.copy()
                                  
        d_obs = 1*(d>0)
        y_obs[d_obs==0] = np.nan  

        df = pd.DataFrame({'y':y, 'd': d, 'y_obs':y_obs, 'd_obs':d_obs, 'x': x, 'z':z})

        # Generate prediction data 
        self.x_new = np.expand_dims(np.arange(-5, 5, 0.1), 1)
 
        self.df = df
        self.mu_y = mu_y
        self.mu_d = mu_d


        return self

    def stan_fit(self, type: str='Heckman', iter=1000, chains=4):

        if type == 'Heckman':
            data_dict = {'N'     : self.N,
                         'N_y'   : len(self.df[self.df.d_obs==1].values),
                         'p'     : 1,
                         'q'     : 1,
                         'X'     : np.expand_dims(self.df.x[self.df.d_obs==1].values,1),
                         'Z'     : np.expand_dims(self.df.z.values, 1),
                         'D'     : self.df.d_obs.values,
                         'y'     : self.df.y_obs[self.df.d_obs==1].values,
                         'N_new' : self.x_new.shape[0],
                         'X_new' : self.x_new,
                        }

            sm = pystan.StanModel('../stan/Heckman.stan') 
           
        elif type == 'OLS':
            data_dict = {'N_y'   : len(self.df[self.df.d_obs==1].values),
                         'p'     : 1,
                         'X'     : np.expand_dims(self.df.x[self.df.d_obs==1].values,1),
                         'y'     : self.df.y_obs[self.df.d_obs==1].values,
                         'N_new' : self.x_new.shape[0],
                         'X_new' : self.x_new,
                        }
            
            sm = pystan.StanModel('../stan/OLS.stan') 

        multiprocessing.set_start_method("fork", force=True)
        fit = sm.sampling(data=data_dict, iter=iter, chains=chains)
        samples = fit.extract(permuted=True)

        self.fit = fit
        self.samples = samples

        return self



    def stan_model_summary(self):

        summary_dict = self.fit.summary()
        summary_fit = pd.DataFrame(summary_dict['summary'],
                                  columns=summary_dict['summary_colnames'],
                                  index=summary_dict['summary_rownames'])

        self.summary_fit = summary_fit

        return self







    
