# optimizers

- opimizer = Optimizer(param=your_weights_array, o_name='rms')
- 
- then in your iteation loop:
- 
- optimizer.optimize( param=your_weights_array, dparam=your_gredient_array) 
- your_(new)_weights = optimizer.get_param()
