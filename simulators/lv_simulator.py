import numpy as np    
from CPP import lvssa

class LVSim(object):
    def __init__(self, simulate_summary=True, scaling=5):
        self._simulate_summary = simulate_summary
        self.x_dim = 2
        self.n_params = 3
        self.T = 50
        self.scaling = scaling
        self.sum_dim = int((self.T/self.scaling)*self.x_dim) \
                        if simulate_summary else self.x_dim

    def simulate(self, params):
        sigma = 10
        times = np.arange(0,self.T,1)
        x = np.zeros((times.shape[0],self.x_dim))
        c = np.array(params)
        c[1] = c[1]/100
        
        success = 0
        while (success < 1):
            x_prev = np.array([100,100])
            ss = list()
            for i in range(1,len(times)):
                x_next = np.array(lvssa.LV(c, x_prev, times[i-1], times[i]))
                ss.append(x_next)      
                x_prev = x_next
            ss = np.array(ss)
            x0 = np.array([100,100]).reshape((1,self.x_dim))
            x = np.concatenate((x0, ss),axis=0) + np.random.randn(len(times),self.x_dim)*sigma
            if np.any(ss<0):# or np.any(ss>1000):
                success += 0
            else:
                success += 1

        if self._simulate_summary:
            return x[::self.scaling,:].flatten(order='C')
        else:
            return x    
            
    def __call__(self, parameter_set): 
        assert parameter_set.ndim==2, \
                "params array should be of size n-sim x dim(theta)"   
        if self._simulate_summary:
            sim_y = np.zeros((parameter_set.shape[0], self.sum_dim))
        else:
            sim_y = np.zeros((parameter_set.shape[0], self.T, self.sum_dim))
        print("Simulation started")
        for i in range(len(parameter_set)):
            sim_y[i,:] = self.simulate(parameter_set[i])
        print("Simulation Finished")
        return sim_y

