"""
some tools for diagnostics of numerical functions
"""
import numpy as np

def count(fn):
    """ simple counting decorator

    Counts the calls of a function as well as the elements of an input numpy array
    """
    def wrapper(x,*args, **kwargs):
        if isinstance(x,np.ndarray):
            wrapper.called+=x.shape[-1]
        else:
            wrapper.called+= len(x)
        return fn(x,*args, **kwargs)
    wrapper.called= 0
    wrapper.__name__= fn.__name__
    return wrapper


TIMING = True
TIMES ={}

def timer(func):
    """simple timing decorator"""
    if TIMING:
        @functools.wraps(func)
        def wrapper_timer(*args, **kwargs):
            start = time.time()    # 1
            value = func(*args, **kwargs)
            run_time = time.time() - start
            #print(f"%s finished in %1.2es"%(func.__name__,run_time))
            TIMES[func.__name__] = run_time
            return value
    else:
        wrapper_timer = func
    return wrapper_timer
