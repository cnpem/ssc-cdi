(errors)= 
# Errors 


The errors are calculated as bellow, where {math}`I_{ j,{\bf q} }` and {math}`\bar{I}_{j,{\bf q} }` are the measured and predicted intensities of the jth diffraction pattern at the scattering vector {math}`{\bf q}`.


Normalized mean-square error (NMSE) of the intensity:


```{math}
\mathcal{L}_{\text{NMSE}} [ \{ \bar{I}_{j',{\bf q'} } \}_{j',{\bf q'} }] = \frac{\sum_{j,{\bf q}} | I_{ j,{\bf q} } - \bar{I}_{j,{\bf q} } |^{2} }{\sum_{j,{\bf q}} I_{ j,{\bf q} }^{2}}
```

R-factor of the amplitude (R):

```{math}
\mathcal{L}_{\text{R}} [ \{ \bar{I}_{j',{\bf q'} } \}_{j',{\bf q'} }] = \frac{\sum_{j,{\bf q}} | \sqrt{I_{ j,{\bf q} }} - \sqrt{\bar{I}_{j,{\bf q} }} | }{\sum_{j,{\bf q}} \sqrt{I_{ j,{\bf q} }}}
```

Generalized Kullback-Leibler divergence (NRGKLD) of the intensity:

```{math}
\mathcal{L}_{\text{NRGKLD}} [ \{ \bar{I}_{j',{\bf q'} } \}_{j',{\bf q'} }; \epsilon] = \frac{\sum_{j,{\bf q}} \{ ( \bar{I}_{j,{\bf q} }^{+} - I_{ j,{\bf q} }^{+}) -  I_{ j,{\bf q} }^{+} \, \ln{( \frac{\bar{I}_{j,{\bf q} }^{+} }{ I_{ j,{\bf q} }^{+} + \delta(I_{ j,{\bf q} }) } )} \} }{\sum_{j,{\bf q}} I_{ j,{\bf q} }}
```


## Previous version error computation

