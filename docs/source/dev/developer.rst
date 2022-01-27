#TODO Developer (CUDA)
=================================

This is the developers documentation. Some example of equations:

- :math:`f = a + ib` is the image, :math:`a` is the attenuation and :math:`b` is the phase;
- :math:`m \equiv \min(0, \Re(\text{sign}(f)))`;
- :math:`\mathscr{F}`: 2D Fourier transform;
- :math:`\mathscr{F}[k](w) = \exp[-i(wx^2 + wy^2)/\eta]`;
- :math:`\mathcal{F}(e^{-f}) = \mathscr{F}^{-1}\left(\hat{k}\cdot\mathscr{F}(e^{-f})\right)`: forward Fresnel convolution operator.

This documentation is divided into one page for each source code file:

.. toctree::

    inc
    src