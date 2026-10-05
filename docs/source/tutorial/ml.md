(ml)= 
# ML Engine

Conjugated Gradient Maximum-Likelihood (ML).

## Dictionary parameters

```python
ml = {
    'name': 'ML_python',          
    'batch': 256,                     
    'iterations':50,
    'metric': 'poisson',  
    'regularize': 0.0,    
}
```


| Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `name` | `AP` | (`str`) engine descriptor | `PIE`, `AP`, `RAAR`, `ML_python` |
| `batch` | 0 | (`int`) Number of data arrays to fit into the GPU. If 0 it runs automatically. | {math}`0,1,2,\cdots,N` |
| `iterations` | 50 | (`int`) Number of iterations | {math}`1,2,\cdots,N` |
| `metric` | `poisson` | (`str`) ML noise metric | `poisson`, `gauss`, `amplitude` | 
| `regularize` | {math}`0.0` | (`float`) Regularization; choose until background or noise is reduced; | {math}`0 \leq \cdot \leq 1` | 


### Advanced dictionary parameters

```python
ml = {
    'name': 'ML_python',          
    'batch': 256,                    
    'iterations':50,
    'metric': 'poisson',  
    'regularize': 0.0,  

    # Advanced
    'beta_cap': 10e6,        # Limits the learning rate to reduce the chance of bad object and probe updates                        
    'force_restart': True,   # True: Restart the conjugate gradient each 100 iterations, can help to avoid stagnation                         
    'verbose': True,         # False: Do not print errors during optimization                        
    'precond': True,         # True: Enforce unitary scaling between probe and object gradients to help convergence                        
    'refine_solution': False # True: perform a Newton-Raphson refinement of the cost function for possible better results at cost of time
}
```


| Advanced Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `beta_cap` | {math}`10e6` | (`float`) Limits the learning rate to reduce the chance of bad object and probe update | {math}`\geq 0` | 
| `force_restart` | `True` | (`bool`) True: Restart the conjugate gradient each 100 iterations, can help to avoid stagnation | `True` or `False` | 
| `verbose`| `True` | (`bool`) False: Do not print errors during optimization | `True` or `False` | 
| `precond` | `True` | (`bool`) True: Enforce unitary scaling between probe and object gradients to help convergence | `True` or `False` |  
| `refine_solution` | `False` | (`bool`) True: perform a Newton-Raphson refinement of the cost function for possible better results at cost of time | `True` or `False` |  


## ML implementation details