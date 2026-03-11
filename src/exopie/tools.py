import numpy as np
import pickle
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import minimize
import os



def sigma_cmf(M,R,dM,dR):
    '''
    Analytical function to find the error in core mass fraction.
    '''
    M = M*5.97219e24
    R = R*6.371e6
    rho_bulk = M/(4/3*np.pi*R**3)/1e3
    rho_m,rho_c = 5,10
    return (rho_c/rho_bulk)*np.sqrt(9*dR**2+dM**2)/(rho_c / rho_m - 1)

def delta_cmf(M,R,dM,dR):
    '''
    Analytical function to find difference in core mass fraction.
    '''
    rho_m,rho_c = 5,10
    rho_bulk = M*5.97219e24/(4/3*np.pi*(R*6.371e6)**3)/1e3 
    return (rho_c/rho_bulk)*(dM/M-3*dR/R)/(rho_c / rho_m - 1)

def chemistry(cmf,xSi=0,xFe=0.1,trace_core=0.02,
                xNi=0.1,xAl=0,xCa=0,xWu=0.2,xSiO2=0):
    '''
    Calculate the interior chemistry of a planet.

    Parameters:
    ----------
    cmf: float or array
        Core mass fraction.
    xSi: float or array
        Molar fraction of silicon in the core.
    xFe: float or array
        Molar fraction of iron in the mantle.
    trace_core: float or array
        Molar fraction of trace metals in the core.
    xNi: float or array
        Molar fraction of nickel in the core. If None, assume Fe/Ni ratio is 16.
    xAl: float or array
        Molar fraction of aluminum in the mantle.
    xCa: float or array
        Molar fraction of calcium in the mantle.
    xWu: float or array
        Molar fraction of wustite in the mantle.
    xSiO2: float or array
        Molar fraction of SiO2 in the mantle.

    Returns:
    --------
    FeMF, SiMF, MgMF: list
        Mass fractions of Fe, Si, and Mg.
    '''
    mmf = 1-cmf # mantle mass fraction
    xPv = 1-xWu-xCa-xAl-xSiO2 #molar fraction of porovskite in the mantle
    Fe,Ni,Si,Mg,Ca,Al,O,XCore = [55.85e-3,58.69e-3,28.09e-3,24.31e-3,
                                    40.078e-3,26.98e-3,16e-3,50e-3] # atmoic masses, Xcore stands for other metals in the core
    if xNi is None:
        xNi = (1-xSi-trace_core)/(16*Ni/Fe+1) #based on McDonough&Sun 1995 Fe/Ni ratio is 16 [w]
    xfe_core = 1-xSi-xNi-trace_core #molar fraction of Fe in core
    core_mol = xfe_core*Fe+Si*xSi+Ni*xNi+trace_core*XCore
    man_mol = ( ((1-xFe)*Mg+xFe*Fe+O)*xWu + xCa*(Ca+Si+O*3) + xAl*(Al*2+O*3) +
               xSiO2*(Si+O*2) + xPv*(Mg*(1-xFe)+Fe*xFe+Si+3*O) ) #molar mass of lower mantle
    
    fe_core = cmf*xfe_core*Fe/core_mol
    fe_man = mmf*(xPv+xWu)*xFe*Fe/man_mol 
    
    si_core = cmf*xSi*Si/core_mol
    si_man = mmf*(xPv+xCa+xSiO2)*Si/man_mol

    fe_mass = fe_core+fe_man
    si_mass = si_core+si_man
    mg_mass = mmf*(xPv+xWu)*(1-xFe)*Mg/man_mol
    ca_mass = xCa*Ca*mmf/man_mol
    al_mass = xAl*2*Al*mmf/man_mol
    ni_mass = cmf*xNi*Ni/core_mol
    return fe_mass,si_mass,mg_mass,ca_mass,al_mass,ni_mass

def magnisium_number(xFe,xWu,xCa,xAl):
    '''
    Calculate the magnesium number in the mantle.

    Parameters:
    ----------
    xFe: float
        Molar fraction of iron in the mantle.
    xWu: float
        Molar fraction of wustite in the mantle.
    xCa: float
        Molar fraction of calcium in the mantle.
    xAl: float
        Molar fraction of aluminum in the mantle.
    '''
    xPv = 1-xWu-xCa # calculate the mole fraction of Pv
    Fe_m = xPv*xFe+xWu*xFe #moles of Fe
    Mg_m = xPv*(1-xFe-xAl)+xWu*(1-xFe) #moles of Mg
    return Mg_m/(Fe_m+Mg_m)

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
        radius_interpolation = _water_model
        wmf = np.asarray(wmf) if isinstance(wmf, (list, np.ndarray)) else np.full(n, wmf) # if wmf is not an array, assume the same wmf for all masses
        xi = np.asarray([wmf, M, cmf]).T
    elif amf is not None:
        if np.any(Teq<400) or np.any(Teq>2000):
            raise ValueError("Teq must be between 400-2000 K for envelope planets.")
        if np.any(amf<0.005) or np.any(amf>0.2):
            raise ValueError("amf must be between 0.005-0.2 for envelope planets.")
        if np.any(M<0.8) or np.any(M>31):
            raise ValueError("M must be between 0.8-31 Earth masses for envelope planets.")
        radius_interpolation = _envelope_model
        amf = np.asarray(amf) if isinstance(amf, (list, np.ndarray)) else np.full(n, amf)
        Teq = np.asarray(Teq) if isinstance(Teq, (list, np.ndarray)) else np.full(n, Teq)
        xi = np.asarray([amf, M, Teq]).T
    else:
        radius_interpolation = _rocky_model
        xSi = np.asarray(xSi) if isinstance(xSi, (list, np.ndarray)) else np.full(n, xSi)
        xFe = np.asarray(xFe) if isinstance(xFe, (list, np.ndarray)) else np.full(n, xFe)
        xi = np.asarray([cmf, M, xSi, xFe]).T

    result = radius_interpolation(xi)
    return result if isinstance(M, (list, np.ndarray)) else result[0]

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
    r_earth = get_radius(M,**kwargs)
    rhoe = (r_earth/R)**3
    return rhoe

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
        residual = lambda x,param: (param[0]-get_radius(param[1],cmf=x,xSi=param[2],xFe=param[3]))**2/1e-6
        args = np.asarray([R,M,xSi,xFe]).T
    elif type=='water':
        residual = lambda x,param: (param[0]-get_radius(param[1],cmf=param[2],wmf=x))**2/1e-6
        args = np.asarray([R,M,cmf]).T
    elif type=='envelope':
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

def load_Data(name):
    '''
    Load the data for the rocky and water planets to use in interpolation models.
    '''
    package_dir = os.path.dirname(__file__)
    # load rocky data
    if name == 'rocky':
        with open(package_dir+'/Data/MRdata_rocky.pkl','rb') as f:
            Data = pickle.load(f)
            points = [Data['CMF'],Data['Mass'],Data['xSi'],Data['xFe']]
            Radius = Data['Radius_total'] # tuple of radius data in Re
    # load water data
    elif name == 'water':
        with open(package_dir+'/Data/MRdata_water.pkl','rb') as f:
            Data = pickle.load(f)
            points = [Data['WMF'],Data['Mass'],Data['CMF']]
            Radius = Data['Radius_total'] # tuple of radius data in Re
    # load envelope data (H2-He grid)
    elif name == 'envelope':
        with open(package_dir+'/Data/MRdata_H2.pkl','rb') as f:
            Data = pickle.load(f)
            points = [Data['AMF'],Data['Mass'],Data['Teq']]
            Radius = Data['Radius_total'] # tuple of radius data in Re
    return points,Radius

_DATA_STORE = {}
def get_cached_data(data_type):
    """Internal helper to fetch data from the store, loading it if missing."""
    if data_type not in _DATA_STORE:
        _DATA_STORE[data_type] = load_Data(data_type)
    return _DATA_STORE[data_type]

_rocky_model = RegularGridInterpolator(*get_cached_data('rocky'))
_water_model = RegularGridInterpolator(*get_cached_data('water'))
_envelope_model = RegularGridInterpolator(*get_cached_data('envelope'))