import numpy as np
from exopie.property import exoplanet
from exopie.star import _minerology_residual
from exopie.tools import _rocky_model, _water_model, _envelope_model, get_radius, chemistry
import warnings

class planet(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001], Teq=[300,0],  N=50000, planet_type=None, **kwargs):
        super().__init__(N, Mass, Radius,**kwargs)

        if not planet_type:
            M_test = Mass[0]+Mass[1]
            R_test = Radius[0]
            RTR = get_radius(M_test,cmf=0,xSi=0,xFe=0)
            if RTR>R_test:
                warnings.warn('Planet is inside the rocky region, using purely rocky model.')
                planet_type = 'rocky'
                xSi = kwargs.get('xSi', [0,0.2])
                xFe = kwargs.get('xFe', [0,0.2])
                self.set_xSi(a=xSi[0], b=xSi[1])
                self.set_xFe(a=xFe[0], b=xFe[1])
                self._save_parameters = ['Mass','Radius','CMF','xSi','xFe','FeMF','SiMF','MgMF']   
                self._get_radius = lambda x: _rocky_model(x)
            else:
                warnings.warn('Planet is outside the rocky region and may include volatilies')
                if Teq[0] > 400:
                    warnings.warn(f'The equilibrium temperature is high (Teq~{Teq[0]:.0f}), using gaseous model')
                    planet_type = 'envelope'
                    # CMF not implemented yet
                    # CMF = kwargs.get('CMF', [0.325,0.325]) 
                    # self.set_CMF(a=CMF[0], b=CMF[1])
                    Teq = kwargs.get('Teq', [1000,100])
                    self.set_Teq(mu=Teq[0], sigma=Teq[1])
                    self._save_parameters = ['Mass','Radius','AMF','Teq']
                    self._get_radius = lambda x: _envelope_model(x)
                else:
                    warnings.warn(f'The equilibrium temperature is low (Teq~{Teq[0]:.0f}), using liquid water model')
                    planet_type = 'water'
                    CMF = kwargs.get('CMF', [0.325,0.325])
                    self.set_CMF(a=CMF[0], b=CMF[1])
                    self._save_parameters = ['Mass','Radius','WMF','CMF']
                    self._get_radius = lambda x: _water_model(x)

        else:
            planet_type = planet_type.lower()
        
        self.planet_type = planet_type

    def run(self,host_star=None,**kwargs):
        '''
        Running Monte Carlo simulation of assumed planet model.

        Parameters:
        -----------
            host_star: class
                Host star object with all the properties.
            kwargs:
                parameters provided to host_star_prior function.
        Note:
        ----
            1. xSi and xFe parameters are only used for purely rocky model.
            2. The CMF parameter is assumed to be Rocky core mass fraction, such that rcmf = (1-wmf)/cmf.
            3. For H/He model the CMF parameter is fixed for the moment.
        ''' 
        self._check(self._get_radius)
        def _residual(x,args):
            R,z = args[0],args[1:]
            if not host_star:
                if self.planet_type=='rocky':
                    warnings.warn('Planet is assumed to be rocky, using xSi and xFe to match the stellar abundances.')
                    xFe,xSi = args[1],args[2]
                else:
                    xFe,xSi = 0,0
                minerology = x[1:]
                dr_star_ratios,interior_ratio_constraints = self.host_star_prior(star_abundaces,interior_ratio_constraints,abundaces_normalization)
                arguments = [dr_star_ratios,interior_ratio_constraints,xFe,xSi,xCore_trace]
                chem_residual = _minerology_residual(minerology,arguments)
            else:
                chem_residual = 0

            radius_residual = lambda x,args: np.sum(R-self._get_radius(np.asarray([x,*args]).T))**2/1e-6    
            return radius_residual(x[0],z) + chem_residual
    

        residual = lambda x,param: np.sum(param[0]-self._get_radius(np.asarray([x[0],*param[1:]]).T))**2/1e-6
        self._check(self._get_radius)
        if self.planet_type=='rocky':
            args = np.asarray([self.Radius,self.Mass,self.xSi,self.xFe]).T
            self.CMF = self._run_MC(_residual,args,xi=0.325,bounds=[[0,1]])
        elif self.planet_type=='water':
            args = np.asarray([self.Radius,self.Mass,self.CMF]).T
            self.WMF = self._run_MC(_residual,args,xi=0.1,bounds=[[0,1]])
        elif self.planet_type=='envelope':
            args = np.asarray([self.Radius,self.Mass,self.Teq]).T
            self.AMF = self._run_MC(_residual,args,xi=0.01,bounds=[[0.005,0.2]])
        res = self._run_MC(_residual,args,xi=0.325,bounds=[[0,1]])


    def host_star_prior(self,star_abundaces,
              interior_ratio_constraints=['Fe/Si','Fe/Mg','Mg/Si'],
              abundaces_normalization=[7.46,7.55,7.51,6.30,6.43,6.20]):
        '''
        Convert stellar abundances to planet abundances, using Monte Carlo sampling.
        Stellar abundances (X/H in dex) are assumed to be normalized to Asplund 2021 solar abundances.
        
        Parameters:
        -----------
        star_abundaces: list [Fe/H, Mg/H, Si/H, Ca/H, Al/H, Ni/H]
            List of host star abundances for main abundances in log scale (e.g., [Fe/H]).
        interior_ratio_constraints: list
            List of refractory ratios to constrain the minerology of planet equivelent.
        abundaces_normalization: list
            List of solar abundances for normalization.
        Returns:
        --------
        host_star: object
            Host star object with all the properties.
        '''    
        mu = [55.85e-3,28.09e-3,24.31e-3,40.08e-3,26.98e-3,58.69e-3] # Fe, Mg, Si, Ca, Al, Ni
        stellar_abundaces_w = [10**(star_abundaces[i]+abundaces_normalization[i]-12)*mu[i] for i in range(3)]
        dr_star,dr_star_ratios = {},{}
        for i,item in enumerate(['fe','mg','si','ca','al','ni']):
            dr_star[item] = stellar_abundaces_w[i]
        interior_ratio_constraints = [item.lower() for item in interior_ratio_constraints]
        for item in interior_ratio_constraints:
            dr_star_ratios[item] = eval(item, dr_star.copy())

        print('Using stellar prior:',end=' ')
        [print(item+f'~{dr_star_ratios[item]:.2f}',end=', ') for item in dr_star_ratios.keys()]
        print()

        return dr_star_ratios,interior_ratio_constraints
