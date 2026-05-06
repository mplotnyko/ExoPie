import numpy as np
import copy
from exopie.property import PlanetProperty

class planet(PlanetProperty):
    '''
`   Wrapper to compute the interior structure parameters (CMF, Fe-MF, WMF) for a given exoplanet 
    using the SUPEREARTH interior structure model developed by Valencia et al. (2006) 
    and updated in Plotnykov & Valencia (2020) as well as the AMF using the H2-He grid 
    based on results using CEPAM (Guillot & Morel 1995; Guillot 2010) and H-He EOS from Saumon et al. (1995).

    This function constructs a predictive model by interpolating the data to estimate 
    the interior based on the mass and radius for rocky, water or thin 
    envelope models. The methodology is detailed in Plotnykov & Valencia (2024).

    Parameters:
    -----------
    planet_type: str
        The type of planet to model, can be 'rocky', 'water', or 'envelope'.
        Default is 'rocky'.
    N: int
        Number of samples to generate, default is 50000. 
        If Mass or Radius posterior is given, N is set to the size [n].
    Mass: list
        Set planet's mass in Earth masses, 
        assumes normal distribution or skew normal distribution.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    Radius: list
        Set planet's radius in Earth radii, 
        assumes normal distribution or skew normal distribution.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    Teq: list
        Set equilibrium temperature of the planet, only for envelope planets.
        Format: [mu, sigma] or posterior size [n].
    CMF: list
        Set rocky core mass fraction (rcmf = (1-wmf)/cmf) of the planet, 
        only for water planets.
        Format: [mu, sigma] or posterior size [n].
    xSi: list
        Set silicon amount in the core, only for rocky planets.
        Format: [a, b] or posterior size [n].
    xFe: list
        Set iron amount in the mantle, only for rocky planets.
        Format: [a, b] or posterior size [n].
    
    Attributes:
    -----------
    self.AMF: array
        Atmosphere mass fraction
    self.WMF: array
        Water mass fraction
    self.CMF: array
        Core mass fraction 
    self.FeMF: array
        Iron mass fraction
    self.SiMF: array
        Silicon mass fraction
    self.MgMF: array
        Magnesium mass fraction
    
    Methods:
    --------
    run()
        Executes the internal structure solver and populates 
        compositional results.
    get_outputs()
        Monte carlo outputs used for corner plotting.
    summary()
        Tabulated summary of planetary interior parameters.
    Notes:
    -----
        The method for planet with H/He envelope is in beta,
        at the moment WMF, CMF, xSi, xFe will be ignored when using 'envelope' planet type.
    '''

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
            self.xWu, self.xSiO2, self.xNi, self.xAl, self.xCa = np.zeros([5,self._N])
            
            if self.planet_type=='rocky':
                print('Planet is assumed to be rocky, using xSi and xFe to match the stellar abundances.')
                self.host_star.xFe,self.host_star.xSi = self.xFe,self.xSi
            else:
                print('Planet is not purely rocky, setting xSi and xFe to zero.')
                self.host_star.xFe,self.host_star.xSi = np.zeros(self._N),np.zeros(self._N)
            self.ratio_keys = list(self.host_star.dr_star_ratios.keys())
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
            xi+=[0.325,0.2]
            bounds+=[[1e-15,1-1e-15],[1e-15,0.5]]

            for flag, key in self.host_star.trace_mapping:
                if flag:
                    xi.append(0.0)
                    bounds.append((0.0, 2.0))
        
        args = np.asarray([self.Radius,self.Mass,self.CMF,self.Teq,self.xSi,self.xFe]).T
        self._run_MC(self._residual,args,xi=xi,bounds=bounds,tol=tol,**kwargs)
        return self        

    def _residual(self,x,args):
        R,M,CMF,Teq,xSi,xFe,star_ratios = args
        
        if self.stellar_prior:
            k = 0 if self.planet_type=='rocky' else 1
            self.host_star.chemistry_kwargs['xSi'] = self.xSi[i]
            self.host_star.chemistry_kwargs['xFe'] = self.xFe[i]
            chem_residual = self.host_star._minerology_residual(x[k:],star_ratios)
        else:
            chem_residual = 0            

        if self.planet_type=='rocky':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,xSi,xFe]).T))**2/1e-6    
        elif self.planet_type=='water':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,CMF]).T))**2/1e-6    
        elif self.planet_type=='envelope':
            radius_residual = lambda x: np.sum(R-self._get_radius(np.asarray([x,M,Teq]).T))**2/1e-6    
        
        return radius_residual(x[0]) + chem_residual