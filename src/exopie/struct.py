import numpy as np

class PlanetStruct:
    '''
    Initialize properties and prior distributions of a planet.
    '''
    def __init__(self, Mass, Radius, Teq, N, WMF=None, AMF=None, 
                 CMF=[0.325, 0.325], xSi=[0, 0.2], xFe=[0.0, 0.2], **kwargs):
        # Determine N safely using np.size to handle both lists and arrays
        self._N = max(np.size(Mass) if np.size(Mass) > 3 else int(N),
                     np.size(Radius) if np.size(Radius) > 3 else int(N))
        self.N_total = self._N

        # Initialize parameters
        self.Mass = self._set_parameter(mu=Mass[0], sigma=Mass[1:]) if Mass is not None and np.size(Mass) <= 3 else Mass
        self.Radius = self._set_parameter(mu=Radius[0], sigma=Radius[1:]) if Radius is not None and np.size(Radius) <= 3 else Radius
        self.Teq = self._set_parameter(mu=Teq[0], sigma=Teq[1:]) if Teq is not None and np.size(Teq) <= 3 else Teq
        
        self.CMF = self._set_parameter(a=CMF[0], b=CMF[1], distribution='uniform') if np.size(CMF) == 2 else np.array(CMF)
        self.xSi = self._set_parameter(a=xSi[0], b=xSi[1], distribution='uniform') if np.size(xSi) == 2 else np.array(xSi)
        self.xFe = self._set_parameter(a=xFe[0], b=xFe[1], distribution='uniform') if np.size(xFe) == 2 else np.array(xFe)
        
        self.WMF = np.array(WMF) if WMF is not None else None
        self.AMF = np.array(AMF) if AMF is not None else None
        
        # Initialize Kwargs 
        # Trace mineralogy
        self.xAl = kwargs.get('xAl', 0)
        self.xNi = kwargs.get('xNi', None)
        self.xCa = kwargs.get('xCa', 0)
        self.xWu = kwargs.get('xWu', 0.2)
        self.xSiO2 = kwargs.get('xSiO2', 0.0)
        self.xCore_trace = kwargs.get('xCore_trace', 0.02)
        
        self.FeMF = None
        self.stellar_prior = False

    def _set_parameter(self, mu=1, sigma=0.01, a=0, b=1, distribution='normal'):        
        if distribution == 'normal':
            sig = np.atleast_1d(sigma)
            # Unpack sigma: use the second value if available, else mirror the first
            sig_up = sig[0]
            sig_lw = sig[1] if len(sig) > 1 else sig[0]
            if len(sig) > 1:
                return np.random.choice(self._skewposterior(mu, sig_up, sig_lw, self._N), self._N)
            return np.random.normal(mu, sig_up, self._N)
        elif distribution == 'uniform':
            return np.random.uniform(a,b, self._N) # lower and upper bounds of uniform distibtuion
        else:
            raise ValueError(f"Unsupported distribution: '{distribution}'")
    
    def _skewposterior(self, mu, sigma_up, sigma_lw, N):
        UP = np.random.normal(0, abs(sigma_up), size=N)
        LW = np.random.normal(0, abs(sigma_lw), size=N)
        return mu + np.concatenate([UP[UP > 0], LW[LW < 0]])

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
    def Teq(self):
        return self._Teq
    @Teq.setter
    def Teq(self, value):
        self._Teq = value
