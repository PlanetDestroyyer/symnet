import numpy as np

class RunningMeanStd:
    def __init__(self, shape=(), epsilon=1e-8):
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = epsilon

    def update(self, x):
        x = np.asarray(x)
        if x.ndim == len(self.mean.shape):
            x = x.reshape(1, *x.shape)
            
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        
        total = self.count + batch_count
        delta = batch_mean - self.mean
        
        self.mean += delta * batch_count / total
        self.var = (
            self.var * self.count + 
            batch_var * batch_count + 
            (delta ** 2) * self.count * batch_count / total
        ) / total
        self.count = total

    def normalize(self, x):
        return np.clip((x - self.mean) / (np.sqrt(self.var) + 1e-8), -10, 10)
