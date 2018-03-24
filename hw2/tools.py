import os
import random 
import time
import numpy as np
import pandas as pd

def rescaling(data):
    '''
    Arrange all feature btw [0, 1]
    '''
    if data.ndim > 2:
        raise('不支援 2 維以上的 array 喔喔喔喔喔喔喔')
    else:
        max, min = np.amax(data, axis = 0), np.amin(data, axis = 0)
        new_data = (data - min) / (max - min)
    return new_data, max, min