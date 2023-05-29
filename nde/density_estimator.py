import jax
import flax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split
from jax import random
import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm


from nde.invertible_network import DeepConditionalDensityModel

class DensityEstimator(object):

    def __init__(self, simulator, priors, network_settings, summariser=None):


        self._simulator = simulator
        self._priors = priors
        self.theta_dim = self._simulator.n_params
        self.y_dim = self._simulator.sum_dim
        self.time_length = self._simulator.T

        self._network_settings = network_settings
        self._summariser = summariser
        self._stop_after_epochs = 10
        self._val_fraction = .3

    def get_train_val_dataset(self, n_sim=1000, batch_size=256):

        theta_sim_train = self._priors(n_sim)
        y_sim_train = self._simulator(theta_sim_train)

        theta_sim_val = self._priors(int(np.ceil(n_sim*self._val_fraction)))
        y_sim_val = self._simulator(theta_sim_val)        
        
        
        args_train = tuple((theta_sim_train, y_sim_train))
        data_set_train = tf.data.Dataset \
            .from_tensor_slices(args_train) \
            .shuffle(n_sim) \
            .batch(batch_size)
        
        args_val = tuple((theta_sim_val, y_sim_val))
        data_set_val = tf.data.Dataset \
            .from_tensor_slices(args_val) \
            .shuffle(int(np.ceil(n_sim*self._val_fraction))) \
            .batch(batch_size)
        return tfds.as_numpy(data_set_train), \
            tfds.as_numpy(data_set_val)

    def train(self, key=random.PRNGKey(1), n_sim=1000, batch_size=256, \
        init_lr=0.0005, num_epochs=2**31 - 1, num_warmup_epochs=1):

        steps_per_epoch=int(np.ceil(n_sim/ batch_size))
        steps_per_epoch_val=int(np.ceil((n_sim*self._val_fraction) / batch_size))
  
        # Init model
        model = DeepConditionalDensityModel(self.theta_dim, \
                                    random.PRNGKey(1),\
                                    self._network_settings['hidden_features'],\
                                    self._network_settings['n_blocks'],\
                                    summary_nw=self._summariser)


        # generate data from the simulator fro training
        train_ds, val_ds = self.get_train_val_dataset(n_sim, batch_size)

        # Init optimizer and learning rate schedule
        
        theta_init, y_init = np.random.randn(batch_size,self.theta_dim), \
                    np.random.randn(batch_size,self.time_length,self.y_dim)#
        rngs = {'params': jax.random.PRNGKey(0)}
        params = model.init(rngs, y_init, theta_init)
        
        opt = flax.optim.Adam(learning_rate=init_lr).create(params)
        del params
  
        def lr_warmup(step):
            return init_lr * jnp.minimum(1., step / (num_warmup_epochs * steps_per_epoch + 1e-8))

        @jax.vmap
        def mleloss(z, log_det_J):
            logpz = (0.5 * jnp.square(jnp.linalg.norm(z))) - log_det_J
            return logpz      

        @jax.jit
        def train_step(opt, batch):
            def loss_fn(params):
                theta, y = batch
                z, logdets = model.apply(params, y, theta)
                logpz = jnp.mean(mleloss(z, logdets))
                return logpz
            logs, grad = jax.value_and_grad(loss_fn)(opt.target)
            opt = opt.apply_gradient(grad, learning_rate=lr_warmup(opt.state.step))
            return logs, opt
        
        @jax.jit
        def eval_step(opt, batch):
            def loss_fn(params):
                theta, y = batch
                z, logdets = model.apply(params, y, theta)
                logpz = jnp.mean(mleloss(z, logdets))
                return logpz
            logs = loss_fn(opt.target)
            
            return logs
        
        losses = dict()
        val_loss = float("-Inf")
        best_val_loss = 0.
        epochs_since_last_improvement = 0
        best_param = []
        for ep in range(1, num_epochs + 1):
            losses[ep] = []
            
            with tqdm.tqdm(total=steps_per_epoch, desc='Training epoch {}'.format(ep)) as p_bar:
                # Loop through dataset
                for bi, batch in enumerate(train_ds):
                    args_bt = tuple(batch)                    
                    loss, opt = train_step(opt, args_bt)
                    # progress bar
                    losses[ep].append(loss)
                    p_bar.set_postfix_str("Epoch {0},Batch {1},Loss: {2:.3f},Running Loss: {3:.3f}"
                                            .format(ep, bi + 1, loss, np.mean(losses[ep])))
                    p_bar.update(1) 
            best_param.append(opt.target)
            
            loss_v = 0.
            for bi, batch in enumerate(val_ds):
                args_bv = tuple(batch)                    
                loss_v += eval_step(opt, args_bv)
                
            val_loss = loss_v /steps_per_epoch_val  

            if ep == 1 or val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_since_last_improvement = 0
            else:
                epochs_since_last_improvement += 1

             # If no validation improvement over many epochs, stop training.
            if epochs_since_last_improvement > self._stop_after_epochs - 1:
                best_params = best_param[-(1 + self._stop_after_epochs)]
                print("Density Estimator Converged")
                print("Total Epochs passed", ep)
                print("Params from epoch", (ep -(1 + self._stop_after_epochs) ))
                break
           
        return model, opt.target
    
    def posterior_samples(self, key, data, network, network_pars, n_samples=1000):

        return network.apply(network_pars, \
                y=data, \
                inverse=True, \
                sampling_key=key,\
                n_samples=n_samples)
