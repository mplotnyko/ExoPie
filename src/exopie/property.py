import numpy as np
from scipy.optimize import minimize
from exopie.tools import chemistry,_rocky_model, _water_model, _envelope_model, get_radius, get_cached_data
from exopie.struct import PlanetStruct
import warnings

class PlanetProperty(PlanetStruct):
    '''
    Class with all the methods and initialization of the interior data.
    '''
    def __init__(self, Mass, Radius, Teq, N, planet_type, CMF=[0.325,0.325],**kwargs):
        super().__init__(Mass, Radius, Teq, N, **kwargs) # initialize using the data class
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
        elif planet_type == 'star':
            self._get_radius = lambda x: _rocky_model(x)
        else:
            warnings.warn('Planet type not in the list, using purely rocky casse')
            planet_type = 'rocky'
            self._get_radius = lambda x: _rocky_model(x)
        self.planet_type = planet_type

    def __repr__(self):
        # Calls the summary function when the object is printed.
        if self.planet_type=='rocky':
            params_to_summarize = ['Mass', 'Radius', 'FeMF', 'CMF', 'xSi', 'xFe']
        elif self.planet_type=='water':
            params_to_summarize = ['Mass', 'Radius', 'WMF', 'CMF']
        elif self.planet_type=='envelope':
            params_to_summarize = ['Mass', 'Radius', 'AMF', 'CMF', 'Teq']
        elif self.planet_type=='star':
            params_to_summarize = ['FeMF', 'SiMF', 'MgMF', 'NiMF', 'CMF']
        else:
            params_to_summarize = ['FeMF', 'CMF']
        return self.summary(params_to_summarize)

    def _test(self):
        # check if parameters are accepted by the model.
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
        for i in range(self._N):
            star_ratios = None
            if self.stellar_prior:
                star_ratios = {k: self.host_star.dr_star_ratios[k][i] for k in self.ratio_keys}        
            parameters = [*args[i],star_ratios]
            results = minimize(residual,xi,args=parameters,bounds=bounds,tol=tol)
            if results.success:
                res = results.x
                if self.planet_type=='rocky': self.CMF[i] = res[0]
                if self.planet_type=='water': self.WMF[i] = res[0]
                if self.planet_type=='envelope': self.AMF[i] = res[0]

                if self.stellar_prior:
                    k = 0 if self.planet_type=='rocky' else 1
                    self.CMF[i] = res[k]
                    for item in ['xNi', 'xAl', 'xCa', 'xWu', 'xSiO2']:
                        getattr(self, item)[i] = self.host_star.chemistry_kwargs[item]
                    
                
            self.FeMF,self.SiMF,self.MgMF,self.CaMF,self.AlMF,self.NiMF = chemistry(
                    self.CMF,xSi=self.xSi,xFe=self.xFe,
                    trace_core=self.xCore_trace,xNi=self.xNi,
                    xAl=self.xAl,xCa=self.xCa,xWu=self.xWu,xSiO2=self.xSiO2)

    def get_outputs(self, parameters=['Mass', 'Radius'],truncate=False):
        '''
        Extracts and formats planet interior output.

        Parameters:
        -----------
        parameters: list
            List of supported parameters to extract.
            ['Mass', 'Radius', 'FeMF', 'SiMF', 'MgMF', 'CMF', 'WMF', 'AMF', 
            'Teq', Fe/Si', 'Fe/Mg', 'Mg/Si', 'xSi', 'xFe']
        truncate: bool
            Remove the asymptotic Fe/Mg values,
            that approach infinity as cmf -> 1.
        Returns:
        --------
        samples: 2D numpy array
            The formatted data chains of shape (nsamples, ndim), filtered for validity.
        labels: list of str
            The corresponding labels for the requested parameters.
        '''
        data = self.__dict__
        chains = []
        labels = []
        
        # Your custom label mapping
        label_map = {'Mass': 'M', 'Radius': 'R', 'Teq': r'T$_{eq}$',
                     'FeMF': 'Fe-MF','SiMF': 'Si-MF', 'MgMF': 'Mg-MF'}

        for item in parameters:
            chain = None
            # Specific ratio handaling
            if item == 'Fe/Mg':
                if 'FeMF' in data and 'MgMF' in data:
                    chain = data['FeMF'] / data['MgMF']
            elif item == 'Fe/Si':
                if 'FeMF' in data and 'SiMF' in data:
                    chain = data['FeMF'] / data['SiMF']
            elif item=='Mg/Si':
                if 'MgMF' in data and 'SiMF' in data:
                    chain = data['MgMF'] / data['SiMF']
                    
                
            # Standard parameters
            else:
                if f'_{item}' in data:
                    chain = data[f'_{item}']
                elif item in data:
                    chain = data[item]

            if chain is not None and len(np.atleast_1d(chain)) > 0:
                chains.append(np.ravel(chain))
                # Apply custom label if it exists, otherwise default to the item name
                labels.append(label_map.get(item, item))

        if not chains:
            raise ValueError("No valid data found for the requested parameters.")

        # Filter out infinite/extreme values
        pos = ~np.isnan(data['FeMF']) # use all values that have FeMF values
        if truncate:
            try:
                pos = (data['FeMF'] / data['MgMF']) < 15
            except KeyError:
                # If the data doesn't contain FeMF/MgMF, keep all samples
                pass

        samples = np.column_stack(chains)[pos]
        return samples, labels
    
    def summary(self,params_to_summarize):
        """
        Generates summary table of the computed parameters 
        when the object is printed.
        """
        
        # Calculate convergence (percentage of successful optimizer runs)
        # Using FeMF as the benchmark for a successful run
        if self.FeMF is not None:
            convergence_rate = (self._N / self.N_total) * 100
            convergence_str = f"Accepted samples {self._N:.0f} out of {self.N_total:.0f} ({convergence_rate:.2f}%)"
        else:
            return "N/A (Not yet run)"
        warning = ''
        if convergence_rate<10: warning = '\nThe number of accepted samples is less than 10% of the total samples. Consider switching a different planet type.'

        # Build the output string line by line
        out = [
            f'Inference for {self.planet_type} planet',
            f'{convergence_str}'+warning
        ]
        
        # Table Header
        header = f"{'Param':<8} {'mean':>8} {'se_mean':>8} {'sd':>8} {'2.5%':>8} {'25%':>8} {'50%':>8} {'75%':>8} {'97.5%':>8}"
        out.append(header)
        out.append("-" * len(header))
        for param in params_to_summarize:
            if hasattr(self, param):
                val = getattr(self, param)
                if val is not None:
                    val = val[~np.isnan(val)]
                    # Check if it's a populated numpy array
                    mean = np.mean(val)
                    sd = np.std(val)
                    se_mean = sd / np.sqrt(len(val)) # Standard error of the mean
                    q = np.percentile(val, [2.5, 25, 50, 75, 97.5])
                    row = f"{param:<8} {mean:>8.2f} {se_mean:>8.2f} {sd:>8.2f} {q[0]:>8.2f} {q[1]:>8.2f} {q[2]:>8.2f} {q[3]:>8.2f} {q[4]:>8.2f}"
                    out.append(row)
            
        return "\n".join(out)