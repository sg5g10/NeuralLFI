import numpy as np
from CPP import pkyssa
class PKYSim(object):
    def __init__(self, simulate_summary=True, scaling=5):
        self._simulate_summary = simulate_summary
        # since we observe a 
        # linear combination of proteins thus x_dim = 1
        # but I used x_d=4 internally to 
        # resolve state-space size
        self.x_dim = 1 
        self.x_d = 4 
        self.n_params = 8
        self.T = 100
        self.scaling = scaling
        self.sum_dim = int((self.T/self.scaling)) \
                        if simulate_summary else self.x_dim
    def simulate(self, params):
        sigma = 2
        times = np.arange(0,int(self.T/2),.5)
        x = np.zeros((self.T,self.x_d))
        c = np.exp(params)
      
        success = 0
        while (success < 1):
            x_prev = np.array([8,8,8,5])
            ss = list()
            for i in range(1,self.T):
                x_next = np.array(pkyssa.PKY(c, x_prev, times[i-1], times[i]))
                ss.append(x_next)      
                x_prev = x_next
            ss = np.array(ss)
            x0 = np.array([8,8,8,5]).reshape((1,self.x_d))
            x = np.concatenate((x0, ss),axis=0) 
            y = (x[:,1]) + (2*x[:,2])
            if np.any(ss<0):
                success += 0
            else:
                success += 1

        if self._simulate_summary:
            return y[::self.scaling,0].reshape((self.sum_dim))
        else:
            return y[:,None]

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