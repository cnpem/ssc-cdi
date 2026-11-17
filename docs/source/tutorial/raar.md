(raar)= 
# RAAR Engine

Relaxed Average Alternating Reflections (RAAR) {cite}`luke2005,marchesini2016`.

## Dictionary parameters

```python
raar = {
    'name': 'RAAR',          
    'batch': 0,                     
    'iterations':50,
    'beta': 0.9.
    'regularization_object': 1e-3, # float (min: 0, max: 1); 
    'regularization_probe': 1e-3,  # float (min: 0, max: 1); 
    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations
}
```

| Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `name` | `AP` | (`str`) engine descriptor | `PIE`, `AP`, `RAAR`, `ML_python` |
| `batch` | 0 | (`int`) Number of data arrays to fit into the GPU. If 0 it runs automatically. | {math}`0,1,2,\cdots,N` |
| `iterations` | 50 | (`int`) Number of iterations | {math}`1,2,\cdots,N` |
| `beta` | {math}`0.0` | (`float`) RAAR {math}`\beta` parameter | {math}`0 \leq \cdot \leq 1` | 
| `regularization_object` | {math}`0.0` | (`float`) Object update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `regularization_probe` | {math}`0.0` | (`float`) Probe update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `position_correction` | 0 (no correction) | (`int`) Performs position correction every N iterations. | {math}`0, 1,2,\cdots,N` |


### Advanced dictionary parameters

```python
raar = {
    'name': 'RAAR',          
    'batch': 0,                    
    'iterations':50,
    'beta': 0.9.
    'regularization_object': 1e-3, # float (min: 0, max: 1); 
    'regularization_probe': 1e-3   # float (min: 0, max: 1); 
    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations

    # Advanced
    'step_object': 1.0,            # float (min: 0, max: 1); 
    'step_probe': 1.0,             # float (min: 0, max: 1); 
    'momentum_obj': 0.0,           # float; 
    'momentum_probe': 0.0,         # float;
    'probe_cycles': 1,             # int; 
}
```

| Advanced Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `step_object` | {math}`1.0` | (`float`) Object descent step | {math}`\geq 0` | 
| `step_probe` | {math}`1.0` | (`float`) Probe descent step | {math}`\geq 0` | 
| `momentum_obj`| {math}`0.0` | (`float`) Object momentum parameter | {math}`\geq 0` | 
| `momentum_probe` | {math}`0.0` | (`float`) Probe momentum parameter | {math}`\geq 0` |  
| `probe_cycles` | {math}`1` | (`int`) Probe cycles update | {math}`1,2,\cdots,N` |  


## RAAR implementation details

The engine `RAAR` {cite}`luke2005,marchesini2016` is implemented with regularization {cite}`maiden2017` and heavy-ball momentum {cite}`maiden2017,nesterov2013,beck2017` for object and probe update.  
The iterations are described as: Define,

```{math}
\mathbf{P}_{A}\,\psi_{j}^{n} = \mathcal{F}^{-1}\{ \mathcal{F}\{\psi_{j}^{n}\} \frac{\sqrt{I}}{|\mathcal{F}\{\psi_{j}^{n}\}|}\},
\quad \text{ and } \quad 
\mathbf{P}_{S}\,\psi_{j}^{n} =  P^{n}O^{n}_{j} 
```
```{math}
\mathbf{R}_{S}\,\psi_{j}^{n} = 2 \mathbf{P}_{S}\,\psi_{j}^{n} - \psi_{j}^{n}, 
\quad 
\mathbf{R}_{A}\,\psi_{j}^{n} = 2 \mathbf{P}_{A}\,\psi_{j}^{n} - \psi_{j}^{n} 
```

```{math}
T_{RAAR}\,\psi_{j}^{n} = \beta (\psi_{j}^{n} + \mathbf{P}_{A}\mathbf{R}_{S}\,\psi_{j}^{n}) + (1-2\beta)P^{n}O^{n}_{j}.
```

Set

```{math}
\psi_{j}^{n+1} = T_{RAAR}\,\psi_{j}^{n}.
```

The object and probe update is

```{math}
\bar{O}^{n+1} = O^{n} + \sum\limits_{j} \frac{ {P^{n}}^{*} \psi_{j}^{n+1}}{ (1-\gamma_{o})|P^{n}|^{2} + \gamma_{o} |P^{n}|^{2}_{max}},
```

```{math}
\bar{P}^{n+1} = P^{n} + \sum\limits_{j} \frac{ {O^{n}}^{*} \psi_{j}^{n+1}}{ (1-\gamma_{p})|O^{n}|^{2} + \gamma_{p} |O^{n}|^{2}_{max}}.
```

Now set,

```{math}
\Delta O = \bar{O}^{n+1} - O^{n}, 
\quad \text{ and } \quad
\Delta P = \bar{P}^{n+1} - P^{n}
```

The parameters {math}`0\leq \gamma_{o} \leq 1`, and {math}`0\leq \gamma_{p} \leq 1` are the respective regularizations. The heavy-ball momentum and velocity for object and probe are given by.

```{math}
v_{o}^{n+1} = \mu_{o}v_{o}^{n} + \Delta O, 
\quad 
v_{p}^{n+1} = \mu_{p}v_{p}^{n} + \Delta P
```

```{math} 
O^{n+1} = O^{n} + v_{o}^{n+1}, 
\quad 
P^{n+1} = P^{n} + v_{p}^{n+1}, 
```

where {math}`\mu_{o}` and {math}`\mu_{p}` are the respective object and probe momentum parameter. We can also add step parameters {math}`\alpha_{o}` and  {math}`\alpha_{p}`, like `AP` engine, such that

```{math}
O^{n+1} = O^{n} + \alpha_{o}v_{o}^{n+1}, 
\quad 
P^{n+1} = P^{n} + \alpha_{p}v_{p}^{n+1}.
```

The original update for `RAAR` {cite}`luke2005` is {math}`T_{RAAR} = \frac{\beta}{2}(\mathbf{R}_{S}\mathbf{R}_{A} + \mathbf{I}) + (1-\beta)\mathbf{P}_{A}`, and can be rewritten as 
{math}`T_{RAAR} = \beta(\mathbf{I} + \mathbf{P}_{S}\mathbf{R}_{A}) + (1-2\beta)\mathbf{P}_{A}`. The `ssc-cdi` changes the order of the wavefront update to {math}`T_{RAAR} = \frac{\beta}{2}(\mathbf{R}_{A}\mathbf{R}_{S} + \mathbf{I}) + (1-\beta)\mathbf{P}_{S}`, implementing the `RAAR` as an extension of the `DM` (Difference Maps) engine {cite}`thibault2009`.

The parameters relation to input dictionary is:

| Parameter | Dictionary entry | Description | Reference values |
| :--- | :--- | :--- | :--- |
| {math}`\alpha_o`, {math}`\alpha_p` | `step_object`, `step_probe` | Descent step | {math}`\alpha \geq 0` |
| {math}`\gamma_o`, {math}`\gamma_p` | `regularization_object`, `regularization_probe` | Object/probe update preconditioner parameter | {math}`0 \leq \gamma \leq 1` |
| {math}`\mu_o`, {math}`\mu_p` | `momentum_obj`, `momentum_probe` | Momentum parameter | {math}`\mu \geq 0` |