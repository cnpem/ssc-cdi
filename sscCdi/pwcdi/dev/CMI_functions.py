import numpy as np
import matplotlib.pyplot as plt

def create_filter_values(npoints=100,n=2):

    x = np.linspace(0,1,npoints)
    f1 = x
    f2 = 1-(x-1)**(2*n)
    f3 = x**(2*n)

    fig, ax = plt.subplots(dpi=150)
    ax.plot(x,f1,label='x')
    ax.plot(x,f2,label='1-(x-1)**(2*n)')
    ax.plot(x,f3,label='x**(2*n)')
    ax.legend()
    ax.set_title(f'n={n}')
    return f1, f2, f3

