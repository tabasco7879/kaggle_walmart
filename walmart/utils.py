import numpy as np

def is_numeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def normalize(x):
    if len(x.shape)>1:
        norm=np.linalg.norm(x, axis=1)
        return x/norm[:,np.newaxis], norm
    else:
        norm=np.linalg.norm(x)
        return x/norm, norm