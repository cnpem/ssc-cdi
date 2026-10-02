(rpie)= 
# rPIE Engine

## Dictionary parameters

```python
pie = {
    'name': 'PIE',          
    'iterations':50,
    'regularization_object': 1e-3, # float (min: 0, max: 1); 
    'regularization_probe': 1e-3,  # float (min: 0, max: 1); 
    'step_object': 1.0,            # float (min: 0, max: 1); 
    'step_probe': 1.0,             # float (min: 0, max: 1);
    'position_correction': 0,      # 0: no correction. N: performs correction every N iterations
}
```

| Dictionary Parameters | Default Value | Description | Options | 
| :--- | :--- | :--- | :--- |
| `name` | `AP` | (`str`) engine descriptor | `PIE`, `AP`, `RAAR`, `ML_python` |
| `iterations` | 50 | (`int`) Number of iterations | {math}`1,2,\cdots,N` |
| `regularization_object` | {math}`0.0` | (`float`) Object update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `regularization_probe` | {math}`0.0` | (`float`) Probe update preconditioner parameter | {math}`0 \leq \cdot \leq 1` | 
| `step_object` | {math}`1.0` | (`float`) Object gradient descent step | {math}`\geq 0` | 
| `step_probe` | {math}`1.0` | (`float`) Probe gradient descent step | {math}`\geq 0` |
| `position_correction` | 0 (no correction) | (`int`) Performs position correction every N iterations. | {math}`0, 1,2,\cdots,N` |


## rPIE implementation details

The parameters `step` and `reg` on `rPIE` are parameters of the gradient descent preconditioned object and probe update. The object and probe update of `rPIE` is given by,

```{math}
O^{n+1}({\bf x}) = O^{n}({\bf x}) + \alpha_{o} \frac{ P^{n*}( \psi^{n+1} - \psi^{n} ) }{ \gamma_{o} \, |P^{n}|_{max}^{2} + ( 1 - \gamma_{o} ) | P^{n} |^2}
```

```{math}
P^{n+1}({\bf x}) = P^{n}({\bf x}) + \alpha_{p} \frac{ O^{n*}( \psi^{n+1} - \psi^{n} ) }{ \gamma_{p} \, |O^{n}|_{max}^{2} + ( 1 - \gamma_{p} ) | O^{n} |^2 }
```

The parameters relation to input dictionary is:


| Parameter | Dict entry | Description | Reference values |
| :--- | :--- | :--- | :--- |
| {math}`\alpha_o`, {math}`\alpha_p` | `step_object`, `step_probe` | Gradient descent step | {math}`\alpha \geq 0` |
| {math}`\gamma_o`, {math}`\gamma_p` | `regularization_object`, `regularization_probe` | Object/probe update preconditioner parameter | {math}`0 \leq \gamma \leq 1` |
