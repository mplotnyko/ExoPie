import numpy as np
from exopie.tools import chemistry
from scipy.optimize import minimize

class star:
    """
    Wrapper to compute the planet equivelent interior given the host star elemental abundances,
    using chemical interior model that allows for all the major and minor mineralogies. 
    However, for stars that have Mg/Si<0.86 the planet equivelent interior switches to allow SiO2,
    which may not be as representative or real planet composition.
    
    Parameters:
    ----------- 
    N: int
        Number of samples to generate, default is 50000. 
    Fe: list
        Set host star Fe abundance in dex and normalized to Sun, 
        assumes normal distribution or skew normal distribution.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    Mg: list
        Set host star Fe abundance in dex and normalized to Sun, 
        assumes normal distribution or skew normal distribution.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    Si: list
        Set host star Fe abundance in dex and normalized to Sun, 
        assumes normal distribution or skew normal distribution.
        Format: [mu, sigma] or [mu, sigma_up, sigma_lw] or posterior size [n].
    prior: list
        List of refractory ratios to constrain the minerology of planet equivelent.
        Default is ['Fe/Si', 'Fe/Mg', 'Mg/Si'],
        but if Ca, Al, Ni is defined the prior is expanded.
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
        Core mass fraction
    
    Methods:
    --------
    to_planet()
        Executes the chemical model to compute the planet equivalent composition.
    """

    def __init__(self,Fe=[0,0.04],Mg=[0,0.04],Si=[0,0.04],prior=None,normalization=None,N=50000, **kwargs):
        self.N = N
        if prior is None:
            prior = ['Fe/Si', 'Fe/Mg', 'Mg/Si'] 
            Ca = kwargs.get('Ca', None)
            Al = kwargs.get('Al', None)
            Ni = kwargs.get('Ni', None)
            if Ca: prior += ['Ca/Mg']
            if Al: prior += ['Al/Mg']
            if Ni: prior += ['Ni/Fe']
        self.prior = prior
        
        if normalization is None:
            # Assume Asplund 2021 solar abundances
            normalization = [7.46, 7.55, 7.51, 6.30, 6.43, 6.20]
        
        star = {}
        self.dr_star_ratios = {}
        # Atomic masses for Fe, Si, Mg, Ca, Al, Ni (in kg/mol)
        mu = [55.85e-3, 28.09e-3, 24.31e-3, 40.08e-3, 26.98e-3, 58.69e-3] 
        dr_abundances = {'Fe': Fe,'Si': Si,'Mg': Mg,'Ca':Ca,'Al':Al,'Ni':Ni}

        for i,item in enumerate(dr_abundances.keys()):
            element = dr_abundances[item]
            if element is None: element = [-2,0]
            if len(element)<=3: element = self._set_parameter(element[0], sigma=element[1:])
            setattr(self,item,element)
            star[item] = 10**(element+normalization[i])*mu[i]

        for i,item in enumerate(self.prior):
            self.dr_star_ratios[item] = eval(item, star.copy())

        self.xSi = kwargs.get('xSi', [0,0.2])
        self.xFe = kwargs.get('xFe', [0,0.2])
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)
        
        if len(self.xSi)<=3:
            self.xSi = self._set_parameter(mu=self.xSi[0], sigma=self.xSi[1:])
        if len(self.xFe)<=3:
            self.xFe = self._set_parameter(mu=self.xFe[0], sigma=self.xFe[1:])
        
    def to_planet(self,tol=1e-8):

        model_param = np.zeros([self.N,8])
        planet_data = np.zeros([self.N,6])
        for i in range(self.N):
            xfe,xsi = self.xFe[i],self.xSi[i]
            star_ratios = {item: float(self.dr_star_ratios[item][i]) for item in self.dr_star_ratios.keys()}
            res = minimize(self._minerology_residual,[0.325,0.2,0,0,0],args=[star_ratios,xfe,xsi],tol=tol,
                        bounds=[[1e-15,1-1e-15],[1e-15,0.5],[0,0.2],[0,0.2],[0,0.2]])
            
            if res.success:
                cmf, Xmgsi, xNi, xAl, xCa = res.x
                femf, simf, mgmf, camf, almf, nimf = chemistry(cmf,xSi=xsi,xFe=xfe,trace_core=self.xCore_trace,
                                    xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
                xSiO2, xWu = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)
                planet_data[i] = chemistry(cmf,xSi=xsi,xFe=xfe,trace_core=self.xCore_trace,
                                        xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
                model_param[i] = cmf,xsi,xfe,xNi,xAl,xCa,xWu,xSiO2
            else:
                model_param[i] = np.repeat(np.nan,8)
                planet_data[i] = np.repeat(np.nan,6)
        
        for i,item in enumerate(['FeMF','SiMF','MgMF','CaMF','AlMF','NiMF']):
            setattr(self,item,planet_data[:,i])
        for i,item in enumerate(['CMF','xSi','xFe','xNi','xAl','xCa','xWu','xSiO2']):
            setattr(self,item,model_param[:,i])

    def _minerology_residual(self,x,args):
        cmf, Xmgsi, xNi, xAl, xCa = x
        star_ratios,xFe,xSi = args
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=self.xCore_trace,
                                xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
        xSiO2, xWu = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=self.xCore_trace,
                                xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
        dr_planet = {'Fe': femf, 'Mg': mgmf, 'Si': simf, 'Ca': camf, 'Al': almf, 'Ni': nimf}
        res = 0
        for item in self.prior:
            res+=np.sum(star_ratios[item]-eval(item, dr_planet.copy()))**2/1e-6
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