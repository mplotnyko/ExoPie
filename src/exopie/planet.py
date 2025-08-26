import numpy as np
from scipy.interpolate import interpn
from scipy.optimize import minimize
from exopie.property import exoplanet, host_star, load_Data
from exopie.tools import chemistry
import warnings


class rocky(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001],  N=50000, **kwargs):
        '''
        Assuming purely rocky planet. xSi and xFe are free parameters.
        '''
        super().__init__(N, Mass, Radius,**kwargs)
        xSi = kwargs.get('xSi', [0,0.2])
        xFe = kwargs.get('xFe', [0,0.2])
        self.set_xSi(a=xSi[0], b=xSi[1])
        self.set_xFe(a=xFe[0], b=xFe[1])
        self._save_parameters = ['Mass','Radius','CMF','xSi','xFe','FeMF','SiMF','MgMF']
        self.type = 'rocky'
        self.Points, self.Radius_Data = load_Data(self.type) # load interpolation fits

    def run(self,star=None,ratio=None,star_norm=None):
        '''
        Run the rocky planet model.

        Parameters:
        -----------
        star: list [Fe/H, Mg/H, Si/H]
            Stellar abundances of Fe/H, Mg/H, Si/H.
        ratio: str
            ratio to constrain the planet chemistry to the star.
            e.g. ratio='Fe/Si,Fe/Mg,Mg/Si' (default=None)
        star_norm: list [Fe/H, Mg/H, Si/H]
            Normalization reference for the stellar abundances. 
        
        Attributes:
        --------
        self.CMF: array
            Core mass fraction 
        self.FeMF: array
            Iron mass fraction
        self.SiMF: array
            Silicon mass fraction
        self.MgMF: array
            Magnesium mass fraction
        '''
        get_R = lambda x: interpn(self.Points, self.Radius_Data, x) # x=cmf,Mass,xSi,xFe
        self._check(get_R)
        args = np.asarray([self.Radius,self.Mass,self.xSi,self.xFe]).T
        if star is None:
            residual = lambda x,param: np.sum(param[0]-get_R(np.asarray([x[0],*param[1:]]).T))**2/1e-6
            self.CMF = self._run_MC(residual,args)
            self.FeMF,self.SiMF,self.MgMF,_,_,_ = chemistry(self.CMF,xSi=self.xSi,xFe=self.xFe)
        elif ratio is None:
            warnings.warn('No target ratio provided. Running without stellar constraint.')
            residual = lambda x,param: np.sum(param[0]-get_R(np.asarray([x[0],*param[1:]]).T))**2/1e-6
            self.CMF = self._run_MC(residual,args)
            self.FeMF,self.SiMF,self.MgMF,_,_,_ = chemistry(self.CMF,xSi=self.xSi,xFe=self.xFe)
        else:
            if star_norm is None:
                star_norm = [7.46,7.55,7.51] # Fe, Mg, Si Asplund 2021
            mu = [55.85e-3,28.09e-3,24.31e-3] # Fe, Mg, Si atomic masses
            star_w = [10**(star[i]+star_norm[i]-12)*mu[i] for i in range(3)]
            
            ratio_split = ratio.lower().split(',') # Fe/Si, Fe/Mg, Mg/Si ....
            dr_star = {'fe': star_w[0],'mg': star_w[1], 'si': star_w[2]}
            print('Using stellar constraints:',end=' ')
            [print(item+f' ({eval(item, dr_star.copy()):.2f})',end=', ') for item in ratio_split]
            print()
            
            def residual(x, param):
                radius_residual = np.sum(param[0] - get_R(np.asarray([x[0],param[1],x[1],x[2]]).T))**2/1e-6
                data = chemistry(x[0], xSi=x[1], xFe=x[2],xWu=x[3])
                dr_planet = {'fe': data[0], 'si': data[1], 'mg': data[2]}
                chem_residual = 0
                for item in ratio_split:
                    chem_residual += np.sum(eval(item, dr_star.copy())-eval(item, dr_planet.copy()))**2/1e-6
                return radius_residual + chem_residual
            
            args = np.asarray([self.Radius,self.Mass]).T
            self.CMF,self.xSi,self.xFe,self.xWu = self._run_MC(residual,args,
                                xi=[0.325,0.1,0.1,0.2],bounds=[[0,1],[0,0.2],[0,0.2],[0,0.5]]).T
            self.FeMF,self.SiMF,self.MgMF,_,_,_ = chemistry(self.CMF,xSi=self.xSi,xFe=self.xFe,xWu=self.xWu)
    
class water(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001], N=50000, **kwargs):
        '''
        Water planet method. xSi, xFe and atm_height will be ignored.
        '''
        super().__init__(N, Mass, Radius, **kwargs)
        CMF = kwargs.get('CMF', [0.325,0.325])
        self.set_CMF(a=CMF[0], b=CMF[1])
        self._save_parameters = ['Mass','Radius','WMF','CMF']
        self.type = 'water'
        self.Points, self.Radius_Data = load_Data(self.type) # load interpolation fits

    def run(self):
        '''
        Run the water planet model.

        Attributes:
        --------
        self.WMF: array
            Water mass fraction
        self.CMF: array
            Rocky core mass fraction (rcmf = (1-wmf)/cmf)
        self.FeMF: array
            Iron mass fraction
        self.SiMF: array
            Silicon mass fraction
        self.MgMF: array
            Magnesium mass fraction
        '''
        get_R = lambda x: interpn(self.Points, self.Radius_Data, x) # x=wmf,Mass,cmf   
        self._check(get_R)
        args = np.asarray([self.Radius,self.Mass,self.CMF]).T
        residual = lambda x,param: np.sum(param[0]-get_R(np.asarray([x[0],param[1],param[2]]).T))**2/1e-6
        self.WMF = self._run_MC(residual,args)
        self.FeMF,self.SiMF,self.MgMF,_,_,_ = chemistry(self.CMF,xSi=0.,xFe=0.,xWu=0.2)

class envelope(exoplanet):
    def __init__(self, Mass=[1,0.001], Radius=[1,0.001], N=50000, **kwargs):
        '''
        Envelope planet method (beta). WMF, CMF, xSi, xFe will be ignored.
        The equilibrium temperature (Teq) is assumed parameter to be set. 
        '''
        super().__init__(N, Mass, Radius, **kwargs)
        # CMF not implemented yet
        # CMF = kwargs.get('CMF', [0.325,0.325]) 
        # self.set_CMF(a=CMF[0], b=CMF[1])
        Teq = kwargs.get('Teq', [1000,100])
        self.set_Teq(mu=Teq[0], sigma=Teq[1])
        self._save_parameters = ['Mass','Radius','AMF','Teq']
        self.type = 'envelope'
        self.Points, self.Radius_Data = load_Data(self.type) # load interpolation fits

    def run(self):
        '''
        Run the envelope planet model.

        Attributes:
        --------
        self.AMF: array
            Atmosphere mass fraction
        self.CMF: array
            Rocky core mass fraction (rcmf = (1-amf)/cmf)
        self.FeMF: array
            Iron mass fraction
        self.SiMF: array
            Silicon mass fraction
        self.MgMF: array
            Magnesium mass fraction
        '''
        get_R = lambda x: interpn(self.Points, self.Radius_Data, x)
        self._check(get_R)
        args = np.asarray([self.Radius,self.Mass,self.Teq]).T
        residual = lambda x,param: np.sum(param[0]-get_R(np.asarray([x[0],param[1],param[2]]).T))**2/1e-6
        self.AMF = self._run_MC(residual,args,bounds=[[0.005,0.2]])
        # self.FeMF,self.SiMF,self.MgMF,_,_,_ = chemistry(self.CMF,xSi=0.,xFe=0.,xWu=0.2)
    
def get_radius(M,cmf=0.325,wmf=None,amf=None,xSi=0,xFe=0.1,Teq=1000):
    '''
    Find the Radius of a planet, given mass and interior parameters.
    
    Parameters:
    -----------
    M: float/array
        Mass of the planet in Earth masses, 
        if array the interior paramaters need to be the same size as M.
    cmf: float/array
        Core mass fraction. 
    wmf: float/array
        Water mass fraction.
        xSi and xFe will be ignored and cmf corresponds to rocky portion only (rcmf).
        Thus rcmf is will keep the mantle to core fraction constant, rather than the total core mass.
    amf: float/array
        Atmosphere mass fraction.
        If None, the planet is assumed to be rocky or water.
    xSi: float/array
        Molar fraction of silicon in the core (between 0-0.2).
    xFe: float/array
        Molar fraction of iron in the mantle (between 0-0.2).
    Teq: float/array
        Equilibrium temperature of the planet (K).
        Only used if amf is not None, otherwise Teq=300K is assumed.
    
    Returns:
    --------
    Radius: float or array
        Radius of the planet in Earth radii.
    '''
    M = np.asarray(M) if isinstance(M, (list, np.ndarray)) else np.array([M])
    n = len(M)    
    cmf = np.asarray(cmf) if isinstance(cmf, (list, np.ndarray)) else np.full(n, cmf)
    if wmf is not None:
        Points, Radius_Data = load_Data('water') # load interpolation fits
        wmf = np.asarray(wmf) if isinstance(wmf, (list, np.ndarray)) else np.full(n, wmf) # if wmf is not an array, assume the same wmf for all masses
        xi = np.asarray([wmf, M, cmf]).T
    elif amf is not None:
        if np.any(Teq<400) or np.any(Teq>2000):
            raise ValueError("Teq must be between 400-2000 K for envelope planets.")
        if np.any(amf<0.005) or np.any(amf>0.2):
            raise ValueError("amf must be between 0.005-0.2 for envelope planets.")
        if np.any(M<0.8) or np.any(M>31):
            raise ValueError("M must be between 0.8-31 Earth masses for envelope planets.")
        Points, Radius_Data = load_Data('envelope')
        amf = np.asarray(amf) if isinstance(amf, (list, np.ndarray)) else np.full(n, amf)
        Teq = np.asarray(Teq) if isinstance(Teq, (list, np.ndarray)) else np.full(n, Teq)
        xi = np.asarray([amf, M, Teq]).T
    else:
        Points, Radius_Data = load_Data('rocky')
        xSi = np.asarray(xSi) if isinstance(xSi, (list, np.ndarray)) else np.full(n, xSi)
        xFe = np.asarray(xFe) if isinstance(xFe, (list, np.ndarray)) else np.full(n, xFe)
        xi = np.asarray([cmf, M, xSi, xFe]).T

    result = interpn(Points, Radius_Data, xi)
    return result if isinstance(M, (list, np.ndarray)) else result[0]

def get_interior(M,R,type=None,cmf=0.325,xSi=0,xFe=0.1,Teq=1000):
    '''
    Find the interior parameters of a planet, given mass and radius.

    Parameters:
    -----------
    M: float/array
        Mass of the planet in Earth masses,
    R: float/array
        Radius of the planet in Earth radii.
    type: str
        Type of planet interior to assume. Options are 'rocky', 'water', 'envelope'.
        If None, the function will first try to fit a rocky planet.
    cmf: float/array
        Core mass fraction. 
        Only used if type is 'water'.
    xSi: float/array
        Molar fraction of silicon in the core (between 0-0.2).
        Only used if type is 'rocky'.
    xFe: float/array
        Molar fraction of iron in the mantle (between 0-0.2).
        Only used if type is 'rocky'.
    Teq: float/array
        Equilibrium temperature of the planet (K).
        Only used if type is 'envelope', otherwise Teq=300K is assumed.
    
    Returns:
    --------
    interior: float or array
        Interior parameter of the planet.
        If type is 'rocky', returns cmf.
        If type is 'water', returns wmf.
        If type is 'envelope', returns amf.
    '''

    M = np.asarray(M) if isinstance(M, (list, np.ndarray)) else np.array([M])
    R = np.asarray(R) if isinstance(R, (list, np.ndarray)) else np.array([R])
    n = len(M)    
    cmf = np.asarray(cmf) if isinstance(cmf, (list, np.ndarray)) else np.full(n, cmf)
    xSi = np.asarray(xSi) if isinstance(xSi, (list, np.ndarray)) else np.full(n, xSi)
    xFe = np.asarray(xFe) if isinstance(xFe, (list, np.ndarray)) else np.full(n, xFe)
    Teq = np.asarray(Teq) if isinstance(Teq, (list, np.ndarray)) else np.full(n, Teq)

    if type=='rocky' or type is None:
        Points, Radius_Data = load_Data('water') # load interpolation fits
        residual = lambda x,param: (param[0]-get_radius(param[1],cmf=x,xSi=param[2],xFe=param[3]))**2/1e-6
        args = np.asarray([R,M,xSi,xFe]).T
    elif type=='water':
        Points, Radius_Data = load_Data('water') # load interpolation fits
        residual = lambda x,param: (param[0]-get_radius(param[1],cmf=param[2],wmf=x))**2/1e-6
        args = np.asarray([R,M,cmf]).T
    elif type=='envelope':
        Points, Radius_Data = load_Data('envelope') # load interpolation fits
        residual = lambda x,param: (param[0]-get_radius(param[1],cmf=0.325,amf=x,Teq=param[2]))**2/1e-6
        args = np.asarray([R,M,Teq]).T
    else:
        raise ValueError("type must be 'rocky', 'water', 'envelope' or None")
    
    res = []
    for i in range(n):
        res.append(minimize(residual,0.325,args=args[i],bounds=[[0,1]]).x[0])
    return res if n>1 else res[0]

def get_rhoe(M,R, **kwargs):
    '''
    Find the planet density normalized to Earth-like planet for the same mass.

    Parameters:
    -----------
    M: float or array
        Mass of the planet in Earth masses.
    R: float or array
        Radius of the planet in Earth radii.
    **kwargs: dict {'cmf': float, 'xSi': float, 'xFe': float}
        Optional parameters for radius calculation, default is Earth-like values.
    Returns:
    --------
    rhoe: float or array
        rho_bulk/rho_earth(M).
    '''
    if not kwargs:
        kwargs = {'cmf': 0.325, 'xSi': 0.2, 'xFe': 0.05}
    # if isinstance(M, (list, np.ndarray)):
    #     rhoe = np.zeros(len(M))
    #     # for i in range(len(M)):
    #     r_earth = get_radius(M,**kwargs)
    #     rhoe[i] = (r_earth/R)**3
    # else:
    r_earth = get_radius(M,**kwargs)
    rhoe = (r_earth/R)**3
    return rhoe

def get_mass(R,cmf=0.325,wmf=None,xSi=0,xFe=0.1):
    '''
    Find the Mass of a planet, given radius and interior parameters.

    Parameters:
    -----------
    R: float or array
        Radius of the planet in Earth radii.
        if array the same interior parameters will be used for all masses.
    cmf: float
        Core mass fraction. 
    wmf: float
        Water mass fraction.
        xSi and xFe will be ignored and cmf corresponds to rocky portion only (rcmf).
        Thus rcmf is will keep the mantle to core fraction constant, rather than the total core mass.
    xSi: float
        Molar fraction of silicon in the core (between 0-0.2).
    xFe: float
        Molar fraction of iron in the mantle (between 0-0.2).
    
    Returns:
    --------
    Mass: float or array
        Mass of the planet in Earth masses.
    '''
    residual = lambda x,param: (param[0]-get_radius(x[0],cmf=param[1],wmf=param[2],xSi=param[3],xFe=param[4]))**2/1e-4
    if isinstance(R, (list, np.ndarray)):
        res = []
        for i in range(R):
            args = [R[i],cmf,wmf,xSi,xFe]
            res.append(minimize(residual,1,args=args,bounds=[[10**-0.5,10**1.3]]).x[0])
    else:
        args = [R,cmf,wmf,xSi,xFe]
        res = minimize(residual,1,args=args,bounds=[[10**-0.5,10**1.3]]).x[0]
    return res

def star_to_planet(Fe,Si,Mg,Ca=-2,Al=-2,Ni=-2,Sun=[7.46,7.51,7.55,6.20,6.30,6.43],
                   xSi=[0,0.2],xFe=[0,0.2],xCore_trace=0.02,tol=1e-8):
    '''
    Convert stellar abundances to planet abundances, using Monte Carlo sampling.
    Stellar abundances (X/H in dex) are assumed to be normalized to Asplund 2021 solar abundances.
    
    Parameters:
    -----------
    Fe, Mg, Si: array
        Stellar abundances of Fe, Si, Mg ratative to solar in dex.
    Ca, Al, Ni: array, optional
        Stellar abundances of Ca, Al, Ni ratative to solar in dex.
        Default is to set to -2 dex.
    Sun: list
        Solar abundances of Fe, Si, Mg, Ca, Al, Ni in log scale.
    xSi: list, array
        Range or array of silicon molar fraction in the core.
    xFe: list, array
        Range or array of iron molar fraction in the mantle.
    xCore_trace: float
        Molar fraction of trace metals in the core.
    tol: float
        Tolerance for the optimization.
    Returns:
    --------
    host_star: object
        Host star object with all the properties.
    '''
    if not isinstance(Fe, (np.ndarray)):
        Fe = np.array([Fe])
    N = len(Fe)
    # Assume Asplund 2021 solar abundances
    star = np.empty([6,N])
    mu = [55.85e-3,28.09e-3,24.31e-3,40.08e-3,26.98e-3,58.69e-3] # Fe, Si, Mg, Ca, Al, Ni atomic masses
    star_abundance = [Fe,Si,Mg,Ca,Al,Ni]
    for i in range(6):
        star[i] = 10**(star_abundance[i]+Sun[i])*mu[i]
    
    if len(xFe)==2:
        xfe = np.random.uniform(xFe[0],xFe[1],N)
    if len(xSi)==2:
        xsi = np.random.uniform(xSi[0],xSi[1],N)

    Fe2Si,Fe2Mg,Mg2Si = star[0]/star[1],star[0]/star[2],star[2]/star[1]
    # if Ni,Ca,Al are not provided, set to very low value so xCa, xAl, xNi are zero
    Ni2Fe,Ca2Mg,Al2Mg = star[5]/star[0],star[3]/star[2],star[4]/star[2]
    star_ratio = [Fe2Si,Fe2Mg,Mg2Si,Ni2Fe,Ca2Mg,Al2Mg]

    def residual(x,param):
        # residual function for Monte Carlo sampling
        cmf, Xmgsi, xNi, xAl, xCa = x
        Fe2Si,Fe2Mg,Mg2Si,Ni2Fe,Ca2Mg,Al2Mg,xFe,xSi = param
        
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=xCore_trace,
                                   xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
        xSiO2, xWu = (0, Xmgsi) if Mg2Si > mgmf / simf else (Xmgsi, 0)
        femf,simf,mgmf,camf,almf,nimf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=xCore_trace,
                                   xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
        res = ( (femf/simf - Fe2Si)**2/1e-4 + (femf/mgmf - Fe2Mg)**2/1e-4 + (mgmf/simf - Mg2Si)**2/1e-4 +
                (nimf/femf - Ni2Fe)**2/1e-3 + (camf/mgmf - Ca2Mg)**2/1e-3 + (almf/mgmf - Al2Mg)**2/1e-3 )
        return res
    
    model_param = np.zeros([N,8])
    planet_data = np.zeros([N,6])
    for i in range(N):
        xFe,xSi = xfe[i],xsi[i]
        param = [Fe2Si[i],Fe2Mg[i],Mg2Si[i],Ni2Fe[i],Ca2Mg[i],Al2Mg[i],xFe,xSi]
        res = minimize(residual,[0.325,0.2,0,0,0],args=param,tol=tol,
                       bounds=[[1e-15,1-1e-15],[1e-15,0.5],[0,0.2],[0,0.2],[0,0.2]])
        
        if res.success:
            cmf, Xmgsi, xNi, xAl, xCa = res.x
            femf,simf,mgmf,nimf,camf,almf = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=xCore_trace,
                                   xNi=xNi,xAl=xAl,xCa=xCa,xWu=0,xSiO2=0)
            xSiO2, xWu = (0, Xmgsi) if Mg2Si[i] > mgmf / simf else (Xmgsi, 0)
            data = chemistry(cmf,xSi=xSi,xFe=xFe,trace_core=xCore_trace,
                                    xNi=xNi,xAl=xAl,xCa=xCa,xWu=xWu,xSiO2=xSiO2)
    
            model_param[i] = cmf,xSi,xFe,xNi,xAl,xCa,xWu,xSiO2
            planet_data[i] = data               
            
        else:
            model_param[i] = np.repeat(np.nan,8)
            planet_data[i] = np.repeat(np.nan,6)
    return host_star(star_abundance,star_ratio,planet_data,model_param)
    
