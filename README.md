# optimizers

- opimizer = Optimizer(param=your_weights_array, o_name='rms')
then in your iteation loop:
- optimizer.optimize( param=your_weights_array, dparam=your_gredient_array) 
  your_new_weights = optimizer.optimizer.get_param()
