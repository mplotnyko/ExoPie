import numpy as np
from exopie.property import exoplanet
import warnings

class planet(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001], Teq=[300,0],  N=50000, planet_type=None, **kwargs):
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
        self.xAl = kwargs.get('xAl', 0)
        self.xNi = kwargs.get('xNi', None)
        self.xCa = kwargs.get('xCa', 0)
        self.xWu = kwargs.get('xWu', 0.2)
        self.xSiO2 = kwargs.get('xSiO2', 0.)
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)
        self.host_star = host_star

        self._check(self._get_radius)
        if self.host_star:
            if self.planet_type=='rocky':
                warnings.warn('Planet is assumed to be rocky, using xSi and xFe to match the stellar abundances.')
                host_star.xFe,host_star.xSi = self.xFe,self.xSi
            else:
                host_star.xFe,host_star.xSi = np.zeros(self.N),np.zeros(self.N)

            print('Using stellar prior to constrain rocky interior:',end=' ')
            for item in host_star.prior:
                percentile = np.percentile(host_star.dr_star_ratios[item],[16,50,84])
                mu,sigma = percentile[1],(percentile[2]-percentile[0])/2
                print(item+r'$~\mathcal{N}($'+f'{mu:.2f}, {sigma:.2f})' ,end=', ')
            print()
            self.star_ratios = host_star.dr_star_ratios
        else:
            self.star_ratios = None
        args = np.asarray([self.Radius,self.Mass,self.CMF,self.Teq,self.xSi,self.xFe,self.star_ratios]).T
        
        if self.planet_type=='rocky':
            if not self.host_star:
                xi = [0.325]
                bounds = [[0,1]]
        elif self.planet_type=='water':
            xi = [0.1]
            bounds = [[0,1]]
        elif self.planet_type=='envelope':
            xi = [0.01]
            bounds = [[0.005,0.2]]

        if self.host_star:
            xi+=[0.325,0.2,0,0,0]
            bounds+=[[1e-15,1-1e-15],[1e-15,0.5],[0,0.2],[0,0.2],[0,0.2]]
        
        results = self._run_MC(self._residual,args,xi=xi,bounds=bounds,tol=tol,**kwargs)
        return results

    def _residual(self,x,args):
        R,M,CMF,Teq,xSi,xFe,star_ratios = args
        if self.host_star:
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


    # def host_star_prior(self,star_abundaces,
    #           interior_ratio_constraints=['Fe/Si','Fe/Mg','Mg/Si'],
    #           abundaces_normalization=[7.46,7.55,7.51,6.30,6.43,6.20]):
    #     '''
    #     Convert stellar abundances to planet abundances, using Monte Carlo sampling.
    #     Stellar abundances (X/H in dex) are assumed to be normalized to Asplund 2021 solar abundances.
        
    #     Parameters:
    #     -----------
    #     star_abundaces: list [Fe/H, Mg/H, Si/H, Ca/H, Al/H, Ni/H]
    #         List of host star abundances for main abundances in log scale (e.g., [Fe/H]).
    #     interior_ratio_constraints: list
    #         List of refractory ratios to constrain the minerology of planet equivelent.
    #     abundaces_normalization: list
    #         List of solar abundances for normalization.
    #     Returns:
    #     --------
    #     host_star: object
    #         Host star object with all the properties.
    #     '''    
    #     mu = [55.85e-3,28.09e-3,24.31e-3,40.08e-3,26.98e-3,58.69e-3] # Fe, Mg, Si, Ca, Al, Ni
    #     stellar_abundaces_w = [10**(star_abundaces[i]+abundaces_normalization[i]-12)*mu[i] for i in range(3)]
    #     dr_star,dr_star_ratios = {},{}
    #     for i,item in enumerate(['fe','mg','si','ca','al','ni']):
    #         dr_star[item] = stellar_abundaces_w[i]
    #     interior_ratio_constraints = [item.lower() for item in interior_ratio_constraints]
    #     for item in interior_ratio_constraints:
    #         dr_star_ratios[item] = eval(item, dr_star.copy())

    #     print('Using stellar prior:',end=' ')
    #     [print(item+f'~{dr_star_ratios[item]:.2f}',end=', ') for item in dr_star_ratios.keys()]
    #     print()

    #     return dr_star_ratios,interior_ratio_constraints
