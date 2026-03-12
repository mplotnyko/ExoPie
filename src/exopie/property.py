import numpy as np
from scipy.optimize import minimize
from exopie.tools import chemistry,_rocky_model, _water_model, _envelope_model, get_radius, get_cached_data
import warnings


class planet_property:
    '''
    Initialize properties of a planet.
    '''
    def __init__(self, Mass, Radius, Teq, N, WMF=None, AMF=None, CMF=[0.325,0.325], xSi=[0,0.2], xFe=[0.,0.2],**kwargs):
        N = len(Mass) if len(Mass)>3 else int(N)
        N = len(Radius) if len(Radius)>3 else int(N)      
        self._N = N
        self.N_total = N
        if len(Mass)<=3:
            Mass = self.set_Mass(mu=Mass[0], sigma=Mass[1:])
        if len(Radius)<=3:
            Radius = self.set_Radius(mu=Radius[0], sigma=Radius[1:])
        if len(Teq)<=3:
            Teq = self.set_Teq(mu=Teq[0], sigma=Teq[1:])
        if len(CMF)==2:
            CMF = self.set_CMF(a=CMF[0], b=CMF[1])
        if len(xSi)==2:
            xSi = self.set_xSi(a=xSi[0], b=xSi[1])
        if len(xFe)==2:
            xFe = self.set_xFe(a=xFe[0], b=xFe[1])
        
        self._Mass = Mass
        self._Radius = Radius
        self._Teq = Teq
        self._CMF = np.array(CMF)
        self._xSi = np.array(xSi)
        self._xFe = np.array(xFe)
        self._WMF = np.array(WMF)
        self._AMF = np.array(AMF)
        self.xAl = kwargs.get('xAl', 0)
        self.xNi = kwargs.get('xNi', None)
        self.xCa = kwargs.get('xCa', 0)
        self.xWu = kwargs.get('xWu', 0.2)
        self.xSiO2 = kwargs.get('xSiO2', 0.)
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)
        self.FeMF = None
        self.stellar_prior = False

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

    def set_Mass(self, mu=1, sigma=0.001):
        return  self._set_parameter(mu,sigma)

    def set_Radius(self, mu=1, sigma=0.001):
        return  self._set_parameter(mu,sigma)

    def set_Teq(self,mu=1000, sigma=100):
        return  self._set_parameter(mu,sigma)

    def set_xSi(self, a=0, b=0.2):
        return  np.random.uniform(a,b,self.N)
    
    def set_xFe(self, a=0, b=0.2):
        return np.random.uniform(a,b,self.N)
    
    def set_CMF(self, a=0, b=1):
        return np.random.uniform(a,b,self.N)

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
    corner()
        Corner plot of selected planet interior parameters.
    summary()
        Tabulated summary of planetary interior parameters.
    Notes:
    -----
        The method for planet with H/He envelope is in beta,
        at the moment WMF, CMF, xSi, xFe will be ignored when using 'envelope' planet type.
    '''

    def __init__(self, Mass, Radius, Teq, N, planet_type, CMF=[0.325,0.325],**kwargs):
        super().__init__(Mass, Radius, Teq, N, **kwargs)
        # Find what planet type to use, if not specified
        if planet_type is None:
            M_test = Mass[0]+Mass[1]
            R_test = Radius[0]
            RTR = get_radius(M_test,cmf=0,xSi=0,xFe=0)
            if RTR>R_test:
                print('Planet is inside the rocky region, using purely rocky model.')
                planet_type = 'rocky'
            else:
                print('Planet is outside the rocky region and may include volatilies')
                if np.mean(self.Teq) > 400:
                    print(f'The equilibrium temperature is high (Teq~{Teq[0]:.0f}), using gaseous model')
                    planet_type = 'envelope'# CMF not implemented yet and is assumed fixed
                    
                else:
                    print(f'The equilibrium temperature is low (Teq~{Teq[0]:.0f}), using liquid water model')
                    planet_type = 'water'
        if planet_type == 'envelope':
            self._get_radius = lambda x: _envelope_model(x)
        elif planet_type == 'water':
            self._get_radius = lambda x: _water_model(x)
        elif planet_type == 'rocky':
            self._get_radius = lambda x: _rocky_model(x)
        else:
            warnings.warn('Planet type not in the list, using purely rocky casse')
            planet_type = 'rocky'
            self._get_radius = lambda x: _rocky_model(x)
        self.planet_type = planet_type

    def __repr__(self):
        """
        Generates summary table of the computed parameters 
        when the object is printed.
        """
        if self.planet_type=='rocky':
            params_to_summarize = ['Mass', 'Radius', 'FeMF', 'CMF', 'xSi', 'xFe']
        elif self.planet_type=='water':
            params_to_summarize = ['Mass', 'Radius', 'WMF', 'CMF']
        elif self.planet_type=='envelope':
            params_to_summarize = ['Mass', 'Radius', 'AMF', 'CMF', 'Teq']
        
        # Calculate convergence (percentage of successful optimizer runs)
        # Using FeMF as the benchmark for a successful run
        if self.FeMF is not None:
            convergence_rate = (self._N / self.N_total) * 100
            convergence_str = f"Accepted samples {self._N:.0f} out of {self.N_total:.0f} ({convergence_rate:.2f}%)"
        else:
            return "N/A (Not yet run)"

        # Build the output string line by line
        out = [
            f'Inference for {self.planet_type}, N = {self.N}',
            f'{convergence_str}'
        ]
        
        # Table Header
        header = f"{'Param':<8} {'mean':>8} {'se_mean':>8} {'sd':>8} {'2.5%':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'97.5%':>8}"
        out.append(header)
        out.append("-" * len(header))
        for param in params_to_summarize:
            if hasattr(self, param):
                val = getattr(self, param)
                val = val[~np.isnan(val)]
                # Check if it's a populated numpy array
                mean = np.mean(val)
                sd = np.std(val)
                se_mean = sd / np.sqrt(len(val)) # Standard error of the mean
                q = np.percentile(val, [2.5, 25, 50, 75, 97.5])
                row = f"{param:<8} {mean:>8.2f} {se_mean:>8.2f} {sd:>8.2f} {q[0]:>8.2f} {q[1]:>8.2f} {q[2]:>8.2f} {q[3]:>8.2f} {q[4]:>8.2f}"
                out.append(row)
        return "\n".join(out)

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
        Points,_ = get_cached_data(self.planet_type)
        M_min,M_max = min(Points[1]), max(Points[1])
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
        
        try:
            pos = ( (self.Radius[pos] > get_R(np.asarray([np.repeat(xx[0],self._N),*args]).T)) &
                (self.Radius[pos] < get_R(np.asarray([np.repeat(xx[1],self._N),*args]).T)))
        except:
            raise Exception('The provided M-R pair is outside the model range. Try again or switch to a different planet type.')
        
        self._N = sum(pos)
        
        if self._N/self.N_total<0.1:
            warnings.warn('The number of accepted samples is less than 10% of the total samples. Consider switching a different planet type.')
        if self._N==0:
            raise Exception('Wrong planet type, no M-R pair in bounds')
        for item in  ['Mass','Radius','CMF','Teq','xSi','xFe']:
            setattr(self, item, getattr(self, item)[pos])

    def _run_MC(self,residual,args,bounds=[[0,1]],xi=0.3,tol=1e-8):
        res = []
        for i in range(self.N):
            star_ratios = None
            if self.stellar_prior:
                star_ratios = {item: float(self.host_star.dr_star_ratios[item][i]) for item in self.host_star.dr_star_ratios.keys()}
            parameters = [*args[i],star_ratios]
            
            results = minimize(residual,xi,args=parameters,bounds=bounds,tol=tol)
            if results.success:
                res = results.x
                if self.planet_type=='rocky': self.CMF[i] = res[0]
                if self.planet_type=='water': self.WMF[i] = res[0]
                if self.planet_type=='envelope': self.AMF[i] = res[0]

                if self.stellar_prior:
                    k = 0 if self.planet_type=='rocky' else 1
                    self.CMF[i], Xmgsi, self.xNi[i], self.xAl[i], self.xCa[i] = res[k:]
                    femf,simf,mgmf,camf,almf,nimf = chemistry(
                        self.CMF[i],xSi=self.xSi[i],xFe=self.xFe[i],
                        trace_core=self.xCore_trace,xNi=self.xNi[i],
                        xAl=self.xAl[i],xCa=self.xCa[i],xWu=0,xSiO2=0)
                    self.xSiO2[i],self.xWu[i] = (0, Xmgsi) if star_ratios['Mg/Si'] > mgmf / simf else (Xmgsi, 0)

        self.FeMF,self.SiMF,self.MgMF,self.CaMF,self.AlMF,self.NiMF = chemistry(
                    self.CMF,xSi=self.xSi,xFe=self.xFe,
                    trace_core=self.xCore_trace,xNi=self.xNi,
                    xAl=self.xAl,xCa=self.xCa,xWu=self.xWu,xSiO2=self.xSiO2)

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

