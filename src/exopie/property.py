import numpy as np
from scipy.optimize import minimize
from exopie.tools import chemistry,_rocky_model, _water_model, _envelope_model, get_radius
import warnings


class planet_property:
    '''
    Initialize properties of a planet.
    '''
    def __init__(self, Mass, Radius, Teq, N, CMF=None, WMF=None, AMF=None, xSi=None, xFe=None):
        self._N = N
        self._Mass = np.array(Mass)
        self._Radius = np.array(Radius)
        self._Teq = np.array(Teq)
        self._xSi = np.array(xSi)
        self._xFe = np.array(xFe)
        self._CMF = np.array(CMF)
        self._WMF = np.array(WMF)
        self._AMF = np.array(AMF)    

    @property
    def N(self):
        return self._N
    @property
    def Mass(self):
        return self._Mass
    @Mass.setter
    def Mass(self, value):
        self._Mass = value
    @property
    def Radius(self):
        return self._Radius
    @Radius.setter
    def Radius(self, value):
        self._Radius = value
    @property
    def xSi(self):
        return self._xSi
    @xSi.setter
    def xSi(self, value):
        self._xSi = value
    @property
    def xFe(self):
        return self._xFe
    @xFe.setter
    def xFe(self, value):
        self._xFe = value
    @property
    def CMF(self):
        return self._CMF
    @CMF.setter
    def CMF(self, value):
        self._CMF = value
    @property
    def WMF(self):
        return self._WMF
    @WMF.setter
    def WMF(self, value):
        self._WMF = value
    @property
    def AMF(self):
        return self._AMF
    @AMF.setter
    def AMF(self, value):
        self._AMF = value
    @property
    def Teq(self):
        return self._Teq
    @Teq.setter
    def Teq(self, value):
        self._Teq = value

class exoplanet(planet_property):
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
    CMF: list
        Set rocky core mass fraction (rcmf = (1-wmf)/cmf) of the planet, 
        only for water planets.
        Format: [mu, sigma] or posterior size [n].
    Teq: list
        Set equilibrium temperature of the planet, only for envelope planets.
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
    self.FeMF: array
        Iron mass fraction
    self.CMF: array
        Core mass fraction 
    self.SiMF: array
        Silicon mass fraction
    self.MgMF: array
        Magnesium mass fraction
    Methods:
    --------
    run()
        Executes the internal structure solver and populates 
        compositional results.
    corner()
        Corner plot of selected planet interior parameters.
    summary()
        Tabulated summary of planetary interior parameters.
    Notes:
    -----
        The method for planet with H/He envelope is in beta,
        at the moment WMF, CMF, xSi, xFe will be ignored when using 'envelope' planet type.
    '''

    def __init__(self, Mass, Radius, Teq, N, planet_type,**kwargs):
        N = len(Mass) if len(Mass)>3 else int(N)
        N = len(Radius) if len(Radius)>3 else int(N)      
        super().__init__(Mass, Radius, Teq, N, **kwargs)
        if len(Mass)<=3:
            self.set_Mass(mu=Mass[0], sigma=Mass[1:])
        if len(Radius)<=3:
            self.set_Radius(mu=Radius[0], sigma=Radius[1:])
        # Find what planet type to use
        if planet_type:
            planet_type = planet_type
        else:
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
        self.planet_type = planet_type

    def set_Mass(self, mu=1, sigma=0.001):
        self.Mass = self._set_parameter(mu,sigma)

    def set_Radius(self, mu=1, sigma=0.001):
        self.Radius = self._set_parameter(mu,sigma)

    def set_Teq(self,mu=1000, sigma=100):
        self.Teq = self._set_parameter(mu,sigma)

    def set_xSi(self, a=0, b=0.2):
        self.xSi = np.random.uniform(a,b,self.N)
    
    def set_xFe(self, a=0, b=0.2):
        self.xFe = np.random.uniform(a,b,self.N)
    
    def set_CMF(self, a=0, b=1):
        self.CMF = np.random.uniform(a,b,self.N)

    def _set_parameter(self, mu, sigma):
        if type(sigma) == np.ndarray or type(sigma) == list:
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
    
    def _test(self):
        '''
        check if parameters are accepted by the model.
        '''
        if (self.Mass==None).any():
            raise Exception('Mass must be set.')
        if (self.Radius==None).any():
            raise Exception('Radius must be set.')
        if len(self.Mass) != len(self.Radius):
            raise Exception('Mass and Radius must be of same length.')
    
    def _check(self,get_R):
        self._test()
        # check if parameters are in Mass bounds 
        M_min,M_max = min(self.Points[1]), max(self.Points[1])
        pos = (self.Mass>M_min) & (self.Mass<M_max)
        if sum(pos)==0:
            raise Exception('Mass out of bounds [{:.2f},{:.2f}]'.format(M_min,M_max))
        # check if parameters are in Radius bounds
        if self.planet_type=='rocky':
            pos = pos & (0<=self.xSi) & (self.xSi<=0.2) & (0<=self.xFe) & (self.xFe<=0.2)
            args = np.asarray([self.Mass[pos],self.xSi[pos],self.xFe[pos]])
            xx = [1,0] # bounds for CMF
        elif self.planet_type=='water':
            pos = pos & (0<=self.CMF) & (self.CMF<=1)
            args = np.asarray([self.Mass[pos],self.CMF[pos]])
            xx = [0,1] # bounds for WMF
        elif self.planet_type=='envelope':
            pos = pos & (0<=self.Teq) & (self.Teq<=2000)
            args = np.asarray([self.Mass[pos],self.Teq[pos]])
            xx = [0.005,0.2] # bounds for AMF
        else:
            raise Exception("Type must be 'rocky', 'water' or 'envelope'")
            
        pos = ( (self.Radius[pos] > get_R(np.asarray([np.repeat(xx[0],self._N),*args]).T)) &
                (self.Radius[pos] < get_R(np.asarray([np.repeat(xx[1],self._N),*args]).T)))
        self._N = sum(pos)
        
        if self._N==0:
            raise Exception('Wrong planet type, no M-R pair in bounds')
        if self.planet_type=='rocky':
            for item in  ['Mass','Radius','xSi','xFe']:
                setattr(self, item, getattr(self, item)[pos])        
        elif self.planet_type=='water':
            for item in  ['Mass','Radius','CMF']:
                setattr(self, item, getattr(self, item)[pos])
        elif self.planet_type=='envelope':
            for item in  ['Mass','Radius','Teq']:
                setattr(self, item, getattr(self, item)[pos])
        
    def _run_MC(self,residual,args,bounds=[[0,1]],xi=0.3,tol=1e-8,**kwargs):
        res = []
        for i in range(self.N):
            res.append(minimize(residual,xi,args=args[i],bounds=bounds).x,tol=tol)
        res = np.asarray(res).T
        if self.planet_type=='rocky': self.CMF = res[0]
        if self.planet_type=='water': self.WMF = res[0]
        if self.planet_type=='envelope': self.AMF = res[0]

        if self.host_star:
            k = 0 if self.planet_type=='rocky' else 1
            self.CMF, Xmgsi, self.xNi, self.xAl, self.xCa = res[k:]
            
            femf,simf,mgmf,camf,almf,nimf = chemistry(
                self.CMF,xSi=self.xSi,xFe=self.xFe,
                trace_core=self.xCore_trace,xNi=self.xNi,
                xAl=self.xAl,xCa=self.xCa,xWu=0,xSiO2=0)
            pos = self.host_star.dr_star_ratios['Mg/Si'] > mgmf / simf
            self.xSiO2,self.xWu = np.zeros([self._N,2])
            self.xWu = Xmgsi[pos]
            self.xSiO2 = Xmgsi[~pos]

        self.FeMF,self.SiMF,self.MgMF,self.CaMF,self.AlMF,self.NiMF = chemistry(
                    self.CMF,xSi=self.xSi,xFe=self.xFe,
                    trace_core=self.xCore_trace,xNi=self.xNi,
                    xAl=self.xAl,xCa=self.xCa,xWu=self.xWu,xSiO2=self.xSiO2)
        if not isinstance(xi, (list, np.ndarray)):
            return np.asarray(res).flatten()
        else:
            return np.asarray(res)

    def corner(self, Data=['Mass', 'Radius'], corner_data=None, 
               labels=None, bins=50, smooth=True, show_titles=True, **kwargs):
        '''
        Corner plot of the planet interior parameters.
    
        Parameters:
        -----------
        Data: list
            list of supported parameters to be plot.
            ['Mass', 'Radius', 'FeMF', 'SiMF', 'MgMF', 'CMF', 'WMF', 'AMF', 'Fe/Si', 'Fe/Mg', 'xSi', 'xFe']
        corner_data: array
            Data to plot. If None, the data will be extracted from the model.
        other: 
            Other parameters to be passed to the corner.corner function.
        
        Returns:
        --------
        fig, axs: matplotlib figure and axis objects.
        '''
        try:
            import corner
        except:
            raise ImportError("corner.py is not installed. Please install it with pip install corner")
        if corner_data is None:
            data = self.__dict__
            corner_data = []
            for item in Data:
                if item == 'Fe/Mg':
                    corner_data.append(data['FeMF'] / data['MgMF'])
                elif item == 'Fe/Si':
                    corner_data.append(data['FeMF'] / data['SiMF'])
                else:
                    try:
                        corner_data.append(data[f'_{item}'])
                    except:
                        corner_data.append(data[f'{item}'])
            labels = [item if item not in ['Mass', 'Radius', 'FeMF'] else {'Mass': 'M', 'Radius': 'R', 'FeMF': 'Fe-MF'}[item] for item in Data]
            try:
                pos = data['FeMF']/data['MgMF']<15 # remove all the cases where Fe/Mg goes to inf_
            except:
                pos = np.ones(self.N,dtype=bool)
            corner_data = np.array(corner_data).T[pos]
        fig = corner.corner(corner_data, labels=labels, bins=bins, smooth=smooth, show_titles=show_titles, **kwargs)
        n = len(corner_data.T)
        axs = np.array(fig.axes).reshape((n,n))
        return fig, axs
    def summary(self):
        '''
        Tabulated summary of planetary interior parameters.
        '''
        if self.planet_type=='rocky':
            x = 'FeMF'
        elif self.planet_type=='water':
            x = 'WMF'
        elif self.planet_type=='envelope':
            x = 'AMF'
        
        summary_table = ''
        table = {} # 
        for item in ['Mass', 'Radius', x, 'CMF', 'Fe/Si', 'Fe/Mg']:
            if item == 'Fe/Mg':
                table[item] = self.FeMF / self.MgMF
            elif item == 'Fe/Si':
                table[item] = self.FeMF / self.SiMF
            else:
                table[item] = getattr(self, f'_{item}')
        # make pandas 

