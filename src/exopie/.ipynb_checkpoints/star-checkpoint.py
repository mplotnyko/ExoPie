import numpy as np
from exopie.tools import chemistry
from scipy.optimize import minimize
from exopie.property import PlanetProperty

class star(PlanetProperty):
    """
    Wrapper to compute the planet equivelent interior given the host star elemental abundances,
    using chemical interior model that allows for all the major and minor mineralogies. 
    However, for stars that have Mg/Si<0.86 the planet equivelent interior switches to allow SiO2,
    which may not be as representative or real planet composition.
    
    Parameters:
    ----------- 
    N: int
        Number of samples to generate, default is 50000. 
    Fe,Mg,Si,Ca,Al,Ni: lists
        Set host star elemental abundances individually in dex and normalized to Sun.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    prior: list
        List of refractory ratios to constrain the minerology of planet equivelent.
        Default is ['Fe/Si', 'Fe/Mg', 'Mg/Si'], Ca, Al, Ni can be added.
    normalization: list
        List of solar abundances for normalization.
        Default is Asplund 2021 solar abundances for Fe, Si, Mg, Ca, Al, Ni in log scale.
    
    Attributes:
    -----------
    FeMF: array
        Iron mass fraction
    SiMF: array
        Silicon mass fraction
    MgMF: array
        Magnesium mass fraction
    CMF: array
        Core mass fraction equivalent
    
    Methods:
    --------
    to_planet()
        Executes the chemical model to compute the planet equivalent composition.
    radius_equivalent(Mass)
        Find equivalent planet radius for given output.
    get_outputs()
        Monte carlo outputs used for corner plotting.
    summary()
        Tabulated summary of planetary equivalent interior parameters.

    """

    def __init__(self,prior=['Fe/Si', 'Fe/Mg', 'Mg/Si'], normalization=None,N=50000, **kwargs):
        super().__init__(None, None, None, N, 'rocky', **kwargs)
        self.planet_type = 'star'
        self.flag_Ni,self.flag_Ca,self.flag_Al = False,False,False
        # if Ca, Al or Ni of the star is inputed, add the following ratios as prior constraint
        if kwargs.get('Ca') is not None: 
            self.flag_Ca = True
            prior.append('Ca/Mg')
        if kwargs.get('Al') is not None: 
            self.flag_Al = True
            prior.append('Al/Mg')
        if kwargs.get('Ni') is not None: 
            self.flag_Ni = True
            prior.append('Ni/Fe')
        self.trace_mapping = [(self.flag_Ni, 'xNi'),(self.flag_Al, 'xAl'),(self.flag_Ca, 'xCa')]
        self._prior(prior)

        if normalization is None:
            # Assume Asplund 2021 solar abundances if None have been given
            normalization = [7.46, 7.55, 7.51, 6.30, 6.43, 6.20]
        mu = [55.85e-3, 28.09e-3, 24.31e-3, 40.08e-3, 26.98e-3, 58.69e-3]
        elements = ['Fe', 'Si', 'Mg', 'Ca', 'Al', 'Ni']

        self.dr_star_ratios = {}
        for i, item in enumerate(elements):
            val = kwargs.get(item, None) # extract elemnents
            if val is None:
                arr = np.nan
            elif np.size(val) <= 3:
                # Defaults to distribution='normal' as per standard stellar abundances
                arr = self._set_parameter(mu=val[0], sigma=val[1:]) 
            else:
                arr = np.array(val)
            setattr(self, item, (10 ** (arr + normalization[i])) * mu[i])
        # Set the ratios used for constraining the interior
        star = {'Fe':self.Fe,'Mg':self.Mg,'Si':self.Si,
                'Ca':self.Ca,'Al':self.Al,'Ni':self.Ni}
        for i,item in enumerate(self.prior):
            self.dr_star_ratios[item] = eval(self._compiled_priors[i],{},star)
        if 'Mg/Si' not in prior:
            self.dr_star_ratios['Mg/Si'] = self.Mg/self.Si

    def _prior(self,prior):
        """
        Pre-compiles string prior into executable byte-code objects, handiling complex expressions.
        """
        self.prior = prior
        self._compiled_priors = []
        for item in prior:
            # compile(source, filename, mode)
            # 'eval' mode compiles a single expression
            self._compiled_priors.append(compile(item, '<string>', 'eval'))# get compiled eval function 
    
    def _minerology_residual(self,x,args):
        cmf,Xmgsi = x[:2]
        star_ratios = args    
        
        trace_vars = iter(x[2:]) # find out if there are more parameters passed
        for flag, key in self.trace_mapping:
            if flag:
                self.chemistry_kwargs[key] = next(trace_vars)
        
        # check if the stellar Mg/Si is higher compared to minimum 
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,**self.chemistry_kwargs)
        self.chemistry_kwargs['xSiO2'], self.chemistry_kwargs['xWu'] = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)  
        
        # Final results for the prior calculations
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,**self.chemistry_kwargs)
        self.dr_planet = {'Fe': femf, 'Mg': mgmf, 'Si': simf, 'Ca': camf, 'Al': almf, 'Ni': nimf} # planet equivalent  values
        
        res = 0
        for i,item in enumerate(self.prior):
            val = eval(self._compiled_priors[i],{},self.dr_planet) # compute the prior ratio for comparison
            res+=np.sum((star_ratios[item]-val)**2)/1e-6 
        return res 
        
    def to_planet(self,tol=1e-8):
        '''
        Function to compute the planet equivelent of a star,
        given stellar abundances and defined priors
        '''
        model_param = np.zeros([self._N,6])
        planet_data = np.zeros([self._N,6])

        x0 = [0.325,0.2]
        bounds = [[1e-15,1-1e-15],[1e-15,0.5]]
        for flag, key in self.trace_mapping:
            if flag:
                x0.append(0.0)
                bounds.append((0.0, 2.0))
        x0 = np.array(x0) # Convert to array outside the loop

        # Pre-fetch dictionary keys and set-up the input dic
        ratio_keys = list(self.dr_star_ratios.keys())
        self.chemistry_kwargs = {
                    'xSi': 0,
                    'xFe': 0,
                    'xNi': None, # set to defaults
                    'xAl': 0,
                    'xCa': 0,
                    'xWu': 0,
                    'xSiO2': 0,
                    'trace_core': self.xCore_trace,
                }
        self.dr_planet = {}
        trace_keys = ['xNi', 'xAl', 'xCa', 'xWu', 'xSiO2']

        for i in range(self._N):
            self.chemistry_kwargs['xSi'] = self.xSi[i]
            self.chemistry_kwargs['xFe'] = self.xFe[i]
            
            star_ratios = {k: self.dr_star_ratios[k][i] for k in ratio_keys}
            res = minimize(
                self._minerology_residual, 
                x0, 
                args=star_ratios, 
                tol=tol, 
                bounds=bounds
            )

            if res.success:
                planet_data[i] = [self.dr_planet[k] for k in ['Fe', 'Si', 'Mg', 'Ca', 'Al', 'Ni']]
                model_param[i] = [res.x[0], *[self.chemistry_kwargs[k] for k in trace_keys]]
            else:
                model_param[i] = np.nan
                planet_data[i] = np.nan
        
        for i,item in enumerate(['FeMF','SiMF','MgMF','CaMF','AlMF','NiMF']):
            setattr(self,item,planet_data[:,i])
        for i,item in enumerate(['CMF',*trace_keys]):
            setattr(self,item,model_param[:,i])
        self._N = np.sum(~np.isnan(planet_data[:, 0])) # Count successful runs
