import numpy as np

class OUSim(object):
    def __init__(self, simulate_summary=True, scaling=5):
        self._simulate_summary = simulate_summary
        self.x_dim = 1
        self.n_params = 3
        self.T = 100
        self.scaling = scaling
        self.sum_dim = int((self.T/self.scaling)) \
                        if simulate_summary else self.x_dim
   
    def simulate(self, params):
      """Simulates a OU diffusion process."""
      t_init = 0
      t_end  = 10
      dt     = float(t_end - t_init) / self.T
      ts = np.arange(t_init, t_end, dt)
      ys = np.zeros((self.T,1)) 
      ys[0] = 0.0
      success = 0
      while (success < 1):

        for i in range(1, ts.size):
            t = t_init + (i - 1) * dt
            y = ys[i - 1]
            ys[i] = y + ( (params[0]*(params[1] -y))* dt)  + (params[2] * np.random.normal(0.0, np.sqrt(dt)))

        if (np.any(ys>100)):
            success += 0
        else:
            success += 1        

      if self._simulate_summary:
          return ys[::self.scaling,0].reshape((self.sum_dim))
      else:
        return ys
    

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