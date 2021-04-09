import numpy as np

class Optimizer:
    def __init__(self, param=None, o_name='rms'):
        self.param = param
        self.dparam = None
        self._oname = o_name
        self._e = 1e-8
        self._eps = .01
        self._lambda = .01
        self._momentum = .4
        self._t = 0
        self._decay = 1e5
        self._eta = 150
        self._mem = np.zeros_like(param)
        self._v = np.zeros_like(param)
        self.param_t = np.array(
            [np.zeros_like(param), np.zeros_like(param)])

        self._b1, self._b2 = .9, .999
        self.optimizers = {'rms': self.rms, 'adam': self.adam,
                           'adamx': self.adamx, 'momento': self.momento, 'tempo': self.tempo}
        self._op = self.optimizers[self._oname]

    def set_momentum(self, m):
        self._momentum = m

    def get_param(self):
        return self.param

    def get_dparam(self):
        return self.dparam

    def optimize(self, param=None, dparam=None):
        self.param, self.dparam = param, dparam
        self._op()

    def rms(self):
        b1, eps, e = self._b1, self._eps, self._e
        self._mem = (b1 * self._mem + (1 - b1) * self.dparam*self.dparam)
        self.param = self.param - eps*self.dparam/(np.sqrt(self._mem) + e)

    def adam(self):
        self._t += 1
        b1, b2, e, eps = self._b1, self._b2, self._e, self._eps
        t = self._t
        self._mem = b1 * self._mem + (1 - b1) * self.dparam
        self._v = b2 * self._v + (1 - b2) * self.dparam*self.dparam

        mt = self._mem / (1-np.power(b1, t))
        vt = self._v / (1-np.power(b2, t))
        self.param = self.param - eps * mt / (np.sqrt(vt) + e)

    def tempo(self):
        self.param = self.param - self._eta*self.dparam
        self.param = self.param + self._momentum * \
            (self.param_t[0] - self.param_t[1])
        #np.diff(self.param_t, axis=0)[0]
        self.param_t[1] = self.param_t[0].copy()
        self.param_t[0] = self.param

    def adamx(self):
        b1, b2, eps, e = self._b1, self._b2, self._eps, self._e
        self._mem = b1 * self._mem + (1 - b1) * self.dparam
        self._v = np.maximum(b2 * self._v + e,  np.abs(self.dparam))
        q = self._mem / self._v
        self.param = self.param - eps * q

    def momento(self):
        b1, alpha = self._b1, self._eta
        self._v = self._b1 * self._v + (1-self._b1) * self.dparam
        self.param = self.param - self._eta * self._v
