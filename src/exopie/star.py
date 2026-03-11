import numpy as np
from exopie.tools import chemistry
from scipy.optimize import minimize

class star:
    """
    Initializes and models the properties of a host star to estimate 
    the interior composition of its orbiting exoplanets.
    """

    def __init__(self,Fe=[0,0.04],Si=[0,0.04],Mg=[0,0.04],
                 interior_ratio_constraints=None,
                 abundances_normalization=None,
                 N=50000, **kwargs):
        """
        Initialize the HostStar object.

        Args:
            star_abundance (list): Log abundances of [Fe, Si, Mg, Ca, Al, Ni].
                                   Use None for missing elements.
            interior_ratio_constraints (list, optional): Elemental ratios to constrain.
            abundances_normalization (list, optional): Sollar normalization coeficients.
            N (int): Number of Monte Carlo samples to draw.
            **kwargs: Additional parameters for xSi, xFe, and xCore_trace.
        """
        if interior_ratio_constraints is None:
            interior_ratio_constraints = ['Fe/Si', 'Fe/Mg', 'Mg/Si', 'Ni/Fe', 'Ca/Mg', 'Al/Mg']
        if abundances_normalization is None:
            # Assume Asplund 2021 solar abundances
            abundances_normalization = [7.46, 7.55, 7.51, 6.30, 6.43, 6.20]

        self.N = N
        self.interior_ratio_constraints = interior_ratio_constraints
        elements = ['Fe', 'Si', 'Mg', 'Ca', 'Al', 'Ni']
        for i, item in enumerate(elements):
            star_abundance = kwargs.get(item.lower(), [-2,0])
            if len(star_abundance)<=3:
                star_abundance = self._set_parameter(mu=star_abundance[0], sigma=star_abundance[1:])
            setattr(self,item,star_abundance)

        current_abundances = [self.Fe,self.Si,self.Mg,self.Ca,self.Al,self.Ni]
        # Atomic masses for Fe, Si, Mg, Ca, Al, Ni (in kg/mol)
        mu = [55.85e-3, 28.09e-3, 24.31e-3, 40.08e-3, 26.98e-3, 58.69e-3] 
        star = {}
        for i, item in enumerate(elements):
            star[item] = 10**(current_abundances[i]+abundances_normalization[i])*mu[i]

        self.dr_star_ratios = {}
        for i,item in enumerate(self.interior_ratio_constraints):
            self.dr_star_ratios[item] = eval(item, star.copy())
        self.xSi = kwargs.get('xSi', [0,0.2])
        self.xFe = kwargs.get('xFe', [0,0.2])
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)
        if len(self.xSi)<=3:
            self.xSi = self._set_parameter(mu=self.xSi[0], sigma=self.xSi[1:])
        if len(self.xFe)<=3:
            self.xFe = self._set_parameter(mu=self.xFe[0], sigma=self.xFe[1:])
        
    def star_to_planet(self,tol=1e-8):
        model_param = np.zeros([self.N,8])
        planet_data = np.zeros([self.N,6])
        for i in range(self.N):
            xfe,xsi = self.xFe[i],self.xSi[i]
            star_ratios = {item: self.dr_star_ratios[item] for item in self.dr_star_ratios.keys()}
            
            res = minimize(self._minerology_residual,[0.325,0.2,0,0,0],args=[star_ratios,xfe,xsi],tol=tol,
                        bounds=[[1e-15,1-1e-15],[1e-15,0.5],[0,0.2],[0,0.2],[0,0.2]])
            
            if res.success:
                cmf, Xmgsi, xNi, xAl, xCa = res.x
                model_param[i] = cmf,xsi,xfe,xNi,xAl,xCa,xWu,xSiO2
                femf, simf, mgmf, camf, almf, nimf = chemistry(cmf,xSi=xsi,xFe=xfe,trace_core=self.xCore_trace,
                                    xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
                xSiO2, xWu = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)
                planet_data[i] = chemistry(cmf,xSi=xsi,xFe=xfe,trace_core=self.xCore_trace,
                                        xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
        
                
                
            else:
                model_param[i] = np.repeat(np.nan,8)
                planet_data[i] = np.repeat(np.nan,6)
        
        for i,item in enumerate(['FeMF','SiMF','MgMF','CaMF','AlMF','NiMF']):
            setattr(self,item,planet_data[:,i])
        for i,item in enumerate(['CMF','xSi','xFe','xNi','xAl','xCa','xWu','xSiO2']):
            setattr(self,item,model_param[:,i])

    def _minerology_residual(self,x,args):
        cmf, Xmgsi, xNi, xAl, xCa = x
        dr_star_ratios,xFe,xSi = args
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=self.xSi,xFe=self.xFe,trace_core=self.xCore_trace,
                                xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
        xSiO2, xWu = (0, Xmgsi) if dr_star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=self.xCore_trace,
                                xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
        dr_planet = {'Fe': femf, 'Mg': mgmf, 'Si': simf, 'Ca': camf, 'Al': almf, 'Ni': nimf}
        res = 0
        for item in self.interior_ratio_constraints:
            res+=np.sum(dr_star_ratios[item]-eval(item, dr_planet.copy()))**2/1e-6
        return res 


    def _set_parameter(self, mu, sigma):
        if isinstance(sigma, (np.ndarray, list)):
            try:
                return np.random.choice(self._skewposterior(mu,sigma[0],sigma[1],self.N),self.N)
            except:
                return np.random.normal(mu, sigma[0], self.N)    
        else:
            return np.random.normal(mu, sigma, self.N)
    
    def _skewposterior(self, mu, sigma_up, sigma_lw, N):
        UP = np.random.normal(0,abs(sigma_up),size=N)
        LW = np.random.normal(0,abs(sigma_lw),size=N)
        return mu + np.concatenate([UP[UP>0],LW[LW<0]])