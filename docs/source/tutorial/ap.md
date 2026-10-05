(ap)= 
# AP Engine

The Alternated Projections (AP) {cite}`griffin1984`.

## Dictionary parameters

```python
ap = {
    'name': 'AP',          
    'batch': 0,                     
    'iterations':50,
    'regularization_object': 1e-3, # float (min: 0, max: 1); 
    'regularization_probe': 1e-3,  # float (min: 0, max: 1);
    'momentum_obj': 0.0,           # float; 
    'momentum_probe': 0.0,         # float; 
    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations
}
```


| Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `name` | `AP` | (`str`) engine descriptor | `PIE`, `AP`, `RAAR`, `ML_python` |
| `batch` | 0 | (`int`) Number of data arrays to fit into the GPU. If 0 it runs automatically. | {math}`0,1,2,\cdots,N` |
| `iterations` | 50 | (`int`) Number of iterations | {math}`1,2,\cdots,N` |
| `regularization_object` | {math}`0.0` | (`float`) Object update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `regularization_probe` | {math}`0.0` | (`float`) Probe update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `momentum_obj`| {math}`0.0` | (`float`) Object momentum parameter | {math}`\geq 0` | 
| `momentum_probe` | {math}`0.0` | (`float`) Probe momentum parameter | {math}`\geq 0` | 
| `position_correction` | 0 (no correction) | (`int`) Performs position correction every N iterations. | {math}`0, 1,2,\cdots,N` |


### Advanced dictionary parameters

```python
ap = {
    'name': 'AP',          
    'batch': 0,                    
    'iterations':50,
    'regularization_object': 1e-3, # float (min: 0, max: 1); 
    'regularization_probe': 1e-3   # float (min: 0, max: 1);
    'momentum_obj': 0.0,           # float; 
    'momentum_probe': 0.0,         # float;  
    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations

    # Advanced
    'step_object': 1.0,            # float (min: 0, max: 1); 
    'step_probe': 1.0,             # float (min: 0, max: 1); 
}
```


| Advanced Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `step_object` | {math}`1.0` | (`float`) Object gradient descent step | {math}`\geq 0` | 
| `step_probe` | {math}`1.0` | (`float`) Probe gradient descent step | {math}`\geq 0` | 



## AP implementation details

The engine `AP` {cite}`griffin1984` is implemented with gradient descent with `rPIE` preconditioning {cite}`konijnenberg2017` and heavy-ball momentum {cite}`nesterov2013,beck2017,polyak1964` for object and probe update.  
The iterations are described as: Set

```{math}
\psi_{j}^{n} = P^{n}O^{n}_{j}, \quad \Psi_{j}^{n} = \mathcal{F}\{\psi_{j}^{n}\}, 
\quad \text{and} \quad 
\Delta \psi_{j} = \mathcal{F}^{-1}\{ \Psi_{j}^{n} \frac{\sqrt{I}}{|\Psi_{j}^{n}|} - \Psi_{j}^{n} \}. 
```

Partial Gradient descent `rPIE` update for object {math}`\bar{O}^{n+1}` and probe {math}`\bar{P}^{n+1}`,

```{math}
\bar{O}^{n+1} = O^{n} + \alpha_{o} \Delta O, 
\quad 
\Delta O = \sum\limits_{j} \frac{ {P^{n}}^{*} \Delta \psi_{j}}{ (1-\gamma_{o})|P^{n}|^{2} + \gamma_{o} |P^{n}|^{2}_{max}} 
```

```{math} 
\bar{P}^{n+1} = P^{n} + \alpha_{p} \Delta P, 
\quad 
\Delta P = \sum\limits_{j} \frac{ {O^{n}}^{*} \Delta \psi_{j}}{ (1-\gamma_{p})|O^{n}|^{2} + \gamma_{p} |O^{n}|^{2}_{max}}, 
```

where {math}`\alpha_{o}` and  {math}`\alpha_{p}` are the respective object and probe steps. The parameters {math}`0\leq \gamma_{o} \leq 1`, and {math}`0\leq \gamma_{p} \leq 1` are the respective regularizations. The heavy-ball momentum and velocity for object and probe are given by

```{math}
v_{o}^{n+1} = \mu_{o}v_{o}^{n} + \Delta O, 
\quad 
v_{p}^{n+1} = \mu_{p}v_{p}^{n} + \Delta P
```

```{math}
O^{n+1} = O^{n} + \alpha_{o}v_{o}^{n+1}, 
\quad 
P^{n+1} = P^{n} + \alpha_{p}v_{p}^{n+1}, 
```

where {math}`\mu_{o}` and  {math}`\mu_{p}` are the respective object and probe momentum parameter. If the probe is updated with no momentum, then {math}`P^{n+1} = \bar{P}^{n+1}`.

The parameters relation to input dictionary is:


| Parameter | Dictionary entry | Description | Reference values |
| :--- | :--- | :--- | :--- |
| {math}`\alpha_o`, {math}`\alpha_p` | `step_object`, `step_probe` | Gradient descent step | {math}`\alpha \geq 0` |
| {math}`\gamma_o`, {math}`\gamma_p` | `regularization_object`, `regularization_probe` | Object/probe update preconditioner parameter | {math}`0 \leq \gamma \leq 1` |
| {math}`\mu_o`, {math}`\mu_p` | `momentum_obj`, `momentum_probe` | Momentum parameter | {math}`\mu \geq 0` |


