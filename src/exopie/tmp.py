import numpy as np
from exopie.tools import chemistry
from scipy.optimize import minimize

class HostStar:
    """
    Initializes and models the properties of a host star to estimate 
    the interior composition of its orbiting exoplanets.
    """

    def __init__(self, star_abundance, 
                 interior_ratio_constraints=None,
                 abundances_normalization=None,
                 N=50000, **kwargs):
        """
        Initialize the HostStar object.

        Args:
            star_abundance (list): Log abundances of [Fe, Si, Mg, Ca, Al, Ni].
                                   Use None for missing elements.
            interior_ratio_constraints (list, optional): Elemental ratios to constrain.
            abundances_normalization (list, optional): Normalization constants (Asplund 2021).
            N (int): Number of Monte Carlo samples to draw.
            **kwargs: Additional parameters for xSi, xFe, and xCore_trace.
        """
        if interior_ratio_constraints is None:
            interior_ratio_constraints = ['Fe/Si', 'Fe/Mg', 'Mg/Si', 'Ni/Fe', 'Ca/Mg', 'Al/Mg']
        if abundances_normalization is None:
            abundances_normalization = [7.46, 7.55, 7.51, 6.30, 6.43, 6.20]

        self.N = N
        self.interior_ratio_constraints = interior_ratio_constraints
        self.elements = ['Fe', 'Si', 'Mg', 'Ca', 'Al', 'Ni']

        # Set stellar abundances, defaulting to -2 if not provided
        for i, item in enumerate(self.elements):
            val = star_abundance[i] if star_abundance[i] is not None else -2
            setattr(self, item, val)

        # Re-fetch the newly set attributes to ensure consistency
        current_abundances = [self.Fe, self.Si, self.Mg, self.Ca, self.Al, self.Ni]
        
        # Atomic masses for Fe, Si, Mg, Ca, Al, Ni (in kg/mol)
        mu = [55.85e-3, 28.09e-3, 24.31e-3, 40.08e-3, 26.98e-3, 58.69e-3] 
        
        # Calculate base stellar abundances
        star = {}
        for i, item in enumerate(self.elements):
            star[item] = 10**(current_abundances[i] + abundances_normalization[i]) * mu[i]

        # Calculate expected stellar ratios based on constraints
        self.dr_star_ratios = {}
        for item in self.interior_ratio_constraints:
            num, den = item.split('/')
            self.dr_star_ratios[item] = star[num] / star[den]

        # Extract and format kwargs parameters
        self.xSi = kwargs.get('xSi', [0, 0.2])
        self.xFe = kwargs.get('xFe', [0, 0.2])
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)

        if len(self.xSi) <= 3:
            self.xSi = self._set_parameter(mu=self.xSi[0], sigma=self.xSi[1:])
        if len(self.xFe) <= 3:
            self.xFe = self._set_parameter(mu=self.xFe[0], sigma=self.xFe[1:])


    def _minerology_residual(self, x, args):
        """
        Objective function to minimize when finding the planet's mineralogy.
        
        Args:
            x (list): Model inputs [cmf, Xmgsi, xNi, xAl, xCa].
            args (tuple): Arguments needed for optimization (dr_star_ratios, xFe, xSi).
            
        Returns:
            float: The computed residual difference between the star and planet ratios.
        """
        cmf, Xmgsi, xNi, xAl, xCa = x
        dr_star_ratios, xFe, xSi = args
        
        # First pass to determine Mg/Si ratio
        femf, simf, mgmf, camf, almf, nimf = chemistry(
            cmf, xSi=self.xSi, xFe=self.xFe, trace_core=self.xCore_trace,
            xNi=xNi, xAl=xAl, xCa=xCa, xWu=0, xSiO2=0
        )
        
        # Determine presence of Wüstite (xWu) or Silica (xSiO2)
        if dr_star_ratios['Mg/Si'] > mgmf / simf:
            xSiO2, xWu = 0, Xmgsi
        else:
            xSiO2, xWu = Xmgsi, 0
            
        # Second pass with updated xWu and xSiO2
        femf, simf, mgmf, camf, almf, nimf = chemistry(
            cmf, xSi=xSi, xFe=xFe, trace_core=self.xCore_trace,
            xNi=xNi, xAl=xAl, xCa=xCa, xWu=xWu, xSiO2=xSiO2
        )
        
        # Dictionary keys capitalized to match interior_ratio_constraints
        dr_planet = {'Fe': femf, 'Mg': mgmf, 'Si': simf, 'Ca': camf, 'Al': almf, 'Ni': nimf}
        
        res = 0
        for item in self.interior_ratio_constraints:
            num, den = item.split('/')
            planet_ratio = dr_planet[num] / dr_planet[den]
            res += np.sum((dr_star_ratios[item] - planet_ratio)**2) / 1e-6
            
        return res 

    def _set_parameter(self, mu, sigma):
        """
        Helper method to generate statistical distributions for parameters.
        Note: Assumes a `_skewposterior` method is defined elsewhere in the class or module.
        """
        if isinstance(sigma, (np.ndarray, list)):
            try:
                # Warning: _skewposterior is called but not defined in the provided snippet.
                return np.random.choice(self._skewposterior(mu, sigma[0], sigma[1], self.N), self.N)
            except Exception:
                return np.random.normal(mu, sigma[0], self.N)    
        else:
            return np.random.normal(mu, sigma, self.N)

    def star_to_planet(self, tol=1e-8):
        """
        Converts stellar abundances to planet mineralogy by optimizing the 
        core mass fraction (CMF) and light element distributions.
        
        Args:
            tol (float): Tolerance for the scipy minimization algorithm.
        """
        model_param = np.zeros([self.N, 8])
        planet_data = np.zeros([self.N, 6])
        
        for i in range(self.N):
            xfe = self.xFe[i] if isinstance(self.xFe, np.ndarray) else self.xFe
            xsi = self.xSi[i] if isinstance(self.xSi, np.ndarray) else self.xSi
            
            star_ratios = {item: self.dr_star_ratios[item] for item in self.dr_star_ratios.keys()}
            
            # Initial guess and bounds for [cmf, Xmgsi, xNi, xAl, xCa]
            initial_guess = [0.325, 0.2, 0, 0, 0]
            bounds = [[1e-15, 1 - 1e-15], [1e-15, 0.5], [0, 0.2], [0, 0.2], [0, 0.2]]
            
            res = minimize(self._minerology_residual, initial_guess, 
                           args=[star_ratios, xfe, xsi], tol=tol, bounds=bounds)
            
            if res.success:
                cmf, Xmgsi, xNi, xAl, xCa = res.x
                
                # Retrieve final chemistry
                femf, simf, mgmf, camf, almf, nimf = chemistry(
                    cmf, xSi=xsi, xFe=xfe, trace_core=self.xCore_trace,
                    xNi=xNi, xAl=xAl, xCa=xCa, xWu=0, xSiO2=0
                )
                
                xSiO2, xWu = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)
                
                # Fetch final planet data mass fractions
                data = chemistry(
                    cmf, xSi=xsi, xFe=xfe, trace_core=self.xCore_trace,
                    xNi=xNi, xAl=xAl, xCa=xCa, xWu=xWu, xSiO2=xSiO2
                )
        
                model_param[i] = [cmf, xsi, xfe, xNi, xAl, xCa, xWu, xSiO2]
                planet_data[i] = data               
                
            else:
                model_param[i] = np.repeat(np.nan, 8)
                planet_data[i] = np.repeat(np.nan, 6)
        
        # Save output arrays to class attributes
        for i, item in enumerate(['FeMF', 'SiMF', 'MgMF', 'CaMF', 'AlMF', 'NiMF']):
            setattr(self, item, planet_data[:, i])
            
        for i, item in enumerate(['CMF', 'xSi', 'xFe', 'xNi', 'xAl', 'xCa', 'xWu', 'xSiO2']):
            setattr(self, item, model_param[:, i])