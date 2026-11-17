(ml)= 
# ML Engine

Conjugated Gradient Maximum-Likelihood (ML) {cite}`thibault2012`.

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
    'probe_update_start': 1  # Starting iteration for probe updates, also applied if 'force_restart' is active
}
```

| Advanced Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `beta_cap` | {math}`10e^{6}` | (`float`) Limits the learning rate to reduce the chance of bad object and probe update | {math}`\geq 0` | 
| `force_restart` | `True` | (`bool`) True: Restart the conjugate gradient each 100 iterations, can help to avoid stagnation | `True` or `False` | 
| `verbose`| `True` | (`bool`) False: Do not print errors during optimization | `True` or `False` | 
| `precond` | `True` | (`bool`) True: Enforce unitary scaling between probe and object gradients to help convergence | `True` or `False` |  
| `refine_solution` | `False` | (`bool`) True: perform a Newton-Raphson refinement of the cost function for possible better results at cost of time | `True` or `False` |  
|`probe_update_start` | {math}`1` | (`int`) Starting iteration for probe updates, also applied if `force_restart` is active |{math}`\geq 1` |


## ML implementation details

The object {math}`O(r)` and probe {math}`P(r)`  are obtained from the minimization of a cost function {math}`L(\bar{I}(O, P), I)`, where {math}`\bar{I}(r)` and {math}`I(r)` are the simulated and measured diffraction patterns. This is performed with a conjugated gradient descent method (CG) {cite}`thibault2012`, where the updates in a iteration {math}`i`  is given by: 

{math}`O^i(r) = O^{i-1}(r)+ t^{i}\Delta O^{i}(r)`

{math}`P^i(r) = P^{i-1}(r)+ t^{i}\Delta P^{i}(r)`

where {math}`\Delta O^{i}` and  {math}`\Delta P^{i}` are {math}`L(O, P)` minimization directions and  {math}`t` is the optimal step.  The minimization direction is given by the CG standard way:

{math}`\Delta O^i(r) = -\frac{\partial L^i(r)}{\partial O^i(r)}  + \beta^i\Delta O^{i-1}(r)`

{math}`\Delta P^i(r) = -\frac{\partial L^i(r)}{\partial P^i(r)}  + \beta^i\Delta P^{i-1}(r)`

where {math}`\beta^i` is a scalar given by:

{math}`\beta^i =\frac{g_O^i(g_O^i)^{\dagger} + s^2g_P^i(g_P^i)^{\dagger} - g_O^{i-1}(g_O^i)^{\dagger} - s^2g_P^{i-1}(g_P^i)^{\dagger}}{g_O^{i-1}(g_O^{i-1})^{\dagger} + s^2g_P^{i-1}(g_P^{i -1})^{\dagger}}`

with {math}`s^2 = \frac{|g_O^i|²}{|g_P^i|²}` and {math}`g_O =\frac{\partial L^i(r)}{\partial O^i(r)}`  and {math}`g_P =\frac{\partial L^i(r)}{\partial P^i(r)}` .

To obtain the optimal step {math}`t_{min}` is necessary to explicitly optimize {math}`L(t)` obtained with {math}`L(O^{i-1}+ t\Delta O^{i}, P^{i-1}+ t\Delta P^{i})`. Them, we have two options to  minimize {math}`L(t)`

i) Determine the truncated  {math}`L(t)` to obtain {math}`t_{min}  = - L_{2_{nd}order}''(0)/2L_{2_{nd}order}'(0)`

ii) Minimize {math}`L(t)` with Newton-Raphson.

The implementation uses the first option by standard (the truncation is much faster than the refinement and provide good solutions), then, the Newton-Raphson can be utilized to refine the solution (set `refine_solution` : `True`).

The available cost functions are the following 

- Poissonian log-likelihood:
{math}`L(\bar{I}, I) = \sum\limits_{j q}I_{jq}log(\bar{I_{jq}}) - \bar{I_{jq}}`

- Gaussian:
{math}`L(\bar{I}, I) = \sum\limits_{j q}\frac{w_{jq}}{2}\left(I_{jq} - \bar{I_{jq}}\right)^2`

- Amplitude:
{math}`L(\bar{I}, I) = \sum\limits_{j q}2w_{jq}\left(\sqrt{I_{jq}} - \sqrt{\bar{I_{jq}}}\right)^2`

where {math}`j` and {math}`q` are the index of all position and all pixels respectively, and {math}`w_{jq} = \frac{1}{1 + I_{jq}}` .

There are some details to this implementation to add some features that may increase the chance to obtain a good reconstruction. It is common in CG procedures to force the restart of the gradients to ensure that the minimization is not stuck in some local minima. This can be controle with the `force_restart` parameter, in case of `True`, the gradients are going to be restarted each 100 iterations and each time that the gradients pair ({math}`\Delta O`, {math}`g_O`) and ({math}`\Delta P`, {math}`g_P`)  are perpendicular at the same time. The forced restart also, will stop the probe update for the amount of iterations passed to `probe_update_start` parameter, this can help to avoid any kind of correlation between probe and object that is stagnating the convergence in  local minima. 

The optimization is preconditioned by the use of a scaling between the probe and object gradients. This helps the convergence significantly, specially in the case where the initial probe and object are random or constant, such that it is set as the default procedure. 

The regulatization is performed by a Laplacian regularizer and the cost function has the form:

{math}`L_R (O, P) = L(O, P) + \mu\int| \nabla O(r)|^2dr`

The parameter {math}`\mu` (or `regularize`) control the regularization strenght (values closer to 0: closer to standard reconstruction, values to large: oversmooth reconstruction) is not bounded by data conditions (flux, number of pixels, etc), such that its value can vary a lot between datasets. The use of regularization is a adviced when many reconstructions will be performed, as the ML as the rendency to overfit noise if iterations numbers are too large.  



