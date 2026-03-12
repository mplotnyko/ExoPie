import numpy as np
import copy
from exopie.property import exoplanet

class planet(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001], Teq=[1000,10],  N=50000, planet_type=None, **kwargs):
        super().__init__(Mass, Radius, Teq, N, planet_type, **kwargs)

    def run(self,tol=1e-8,host_star=None,**kwargs):
        '''
        Running Monte Carlo simulation of assumed planet model.

        Parameters:
        -----------
            tol: float
                Tolerance of a minimization function, when modeling interior.
            host_star: class
                Host star object with all the properties.
            kwargs:
                parameters provided to the minerology model, ignored if host_star provided.
        Note:
        ----
            1. xSi and xFe parameters are only used for purely rocky model.
            2. The CMF parameter is assumed to be Rocky core mass fraction, such that rcmf = (1-wmf)/cmf.
            3. For H/He model the CMF parameter is fixed for the moment.
        ''' 
        self._check(self._get_radius)
        if host_star:
            print('Using stellar prior to constrain rocky interior:')
            for item in host_star.prior:
                percentile = np.percentile(host_star.dr_star_ratios[item],[16,50,84])
                mu,sigma = percentile[1],(percentile[2]-percentile[0])/2
                print(item+f'~\U0001D4A9({mu:.2f}, {sigma:.2f})' ,end=', ')
            print()
            
            self.host_star = copy.deepcopy(host_star)
            self.xCore_trace = self.host_star.xCore_trace
            self.xWu, self.xSiO2, self.xNi, self.xAl, self.xCa = np.zeros([5,self.N])
            if self.planet_type=='rocky':
                print('Planet is assumed to be rocky, using xSi and xFe to match the stellar abundances.')
                self.host_star.xFe,self.host_star.xSi = self.xFe,self.xSi
            else:
                print('Planet is not purely rocky, setting xSi and xFe to zero.')
                self.host_star.xFe,self.host_star.xSi = np.zeros(self.N),np.zeros(self.N)
            
            
            
            self.stellar_prior = True
        
        if self.planet_type=='rocky':
            xi = [0.325]
            bounds = [[0,1]]
        elif self.planet_type=='water':
            xi = [0.1]
            bounds = [[0,1]]
        elif self.planet_type=='envelope':
            xi = [0.01]
            bounds = [[0.005,0.2]]

        if self.stellar_prior:
            if self.planet_type=='rocky':
                xi=[]
                bounds=[]
            xi+=[0.325,0.2,0,0,0]
            bounds+=[[1e-15,1-1e-15],[1e-15,0.5],[0,0.2],[0,0.2],[0,0.2]]
        
        args = np.asarray([self.Radius,self.Mass,self.CMF,self.Teq,self.xSi,self.xFe]).T
        self._run_MC(self._residual,args,xi=xi,bounds=bounds,tol=tol,**kwargs)
        return self

    def _residual(self,x,args):
        R,M,CMF,Teq,xSi,xFe,star_ratios = args
        if self.stellar_prior:
            k = 0 if self.planet_type=='rocky' else 1
            CMF, Xmgsi, xNi, xAl, xCa = x[k:]
            chem_residual = self.host_star._minerology_residual([CMF, Xmgsi, xNi, xAl, xCa],[star_ratios,xFe,xSi])
        else:
            chem_residual = 0            
        if self.planet_type=='rocky':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,xSi,xFe]).T))**2/1e-6    
        elif self.planet_type=='water':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,CMF]).T))**2/1e-6    
        elif self.planet_type=='envelope':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,Teq]).T))**2/1e-6    
        
        return radius_residual(x[0]) + chem_residual