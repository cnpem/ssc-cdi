# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################


import cupy as cp
import numpy as np
import matplotlib.pyplot as plt

from .engines_common import calculate_errors

def calculate_gradients_obj_probe(obj, probe, positions, data):
    gradient_o = cp.zeros_like(obj, dtype=complex)
    gradient_p = cp.zeros_like(probe, dtype=complex)

    u, v = cp.linspace(0, probe.shape[1] - 1, probe.shape[1]), cp.linspace(0, probe.shape[0] - 1, probe.shape[0])
    U, V = cp.meshgrid(u, v)

    shapey, shapex = probe.shape
    wavefronts = cp.zeros_like(data,dtype=complex)

    for i in range(positions.shape[0]):
        mask = cp.ones_like(probe).astype(cp.complex64)
        mask[data[i] < 0] = 0
        obj_roi = obj[positions[i, 0]:positions[i, 0] + shapey, positions[i, 1]:positions[i, 1] + shapex]

        total_intensity = cp.abs(cp.fft.fftshift(cp.fft.fft2(obj_roi * probe)))**2
        total_intensity += 1e-6

        psi = cp.fft.fftshift(cp.fft.fft2(obj_roi * probe))
        wavefronts[i] = psi
        factor1 = mask * (1 - data[i] / total_intensity)
        Fqsi = psi * factor1
        term2 = cp.fft.ifft2(cp.fft.ifftshift(Fqsi))

        gradient_o[positions[i, 0]:positions[i, 0] + shapey, positions[i, 1]:positions[i, 1] + shapex] += 2 * cp.conj(probe) * term2
        gradient_p += 2 * cp.conj(obj_roi) * term2

    return gradient_o, gradient_p, wavefronts

def total_variation_regularization(obj, weight=1e-3, epsilon=1e-8):
    grad_x = cp.roll(obj, -1, axis=1) - obj
    grad_y = cp.roll(obj, -1, axis=0) - obj
    grad_magnitude = cp.sqrt(cp.abs(grad_x)**2 + cp.abs(grad_y)**2 + epsilon)
    tv_reg = cp.sum(grad_magnitude)
    grad_tv_x = grad_x / grad_magnitude
    grad_tv_y = grad_y / grad_magnitude
    div_x = grad_tv_x - cp.roll(grad_tv_x, 1, axis=1)
    div_y = grad_tv_y - cp.roll(grad_tv_y, 1, axis=0)
    grad_tv = div_x + div_y
    return weight * tv_reg, weight * grad_tv

def update_parameters(obj, probe, gradient_o, gradient_p, direction_o, direction_p, beta_o, beta_p, step_o, step_p, optimizer):
    if optimizer == 'gradient_descent':
        obj -= step_o * gradient_o
        probe -= step_p * gradient_p
        direction_o = None
        direction_p = None
    elif optimizer == 'conjugate_gradient':
        direction_o = gradient_o + beta_o * direction_o
        direction_p = gradient_p + beta_p * direction_p
        obj -= step_o * direction_o
        probe -= step_p * direction_p
    return obj,probe,direction_o, direction_p

def ML_cupy(data, positions, initial_obj, initial_probe, algo_inputs):
    """
    Performs the ML (Maximum Likelihood) algorithm using CuPy for accelerated computation.

    Args:
        data (array-like): The measured data.
        positions (array-like): The positions of the measurements.
        initial_obj (array-like): The initial guess for the object.
        initial_probe (array-like): The initial guess for the probe.
        algo_inputs (dict): A dictionary containing algorithm inputs including:
            - iterations (int): The number of iterations.
            - step_object (float): The step size for updating the object.
            - step_probe (float): The step size for updating the probe.
            - optimizer (str): The optimizer to use for updating the parameters.
            - probe_support_array (array-like): The probe support array.

    Returns:
        tuple: A tuple containing the updated object, updated probe, and the error array.

    Raises:
        ValueError: If the error increases by a factor of 1000.

    """
    data = cp.array(data) 
    positions = cp.array(positions)
    initial_obj = cp.array(initial_obj)
    initial_probe = cp.array(initial_probe)



    iterations = algo_inputs['iterations']
    step_o = algo_inputs['step_object']
    step_p = algo_inputs['step_probe']
    optimizer = algo_inputs['optimizer']
    probe_support = algo_inputs['probe_support_array']

    if algo_inputs["fresnel_regime"] == True:
        pass
    else:
        algo_inputs['source_distance'] = None

    if probe_support is not None:
        probe_support = cp.array(probe_support)
    
    # display_interval=None

    obj = initial_obj.copy()
    probe = initial_probe.copy()

    shapey, shapex = probe.shape

    gradient_o_prev, gradient_p_prev = None, None
    direction_o, direction_p = None, None

    error = cp.zeros((iterations,1))

    for i in range(iterations):
        print('Iteration:', i,end='\r')
        # if i % display_interval == 0:
        #     fig, ax = plt.subplots(1, 5, figsize=(20, 7))
        #     ax[0].imshow(cp.asnumpy(cp.abs(obj)))
        #     ax[1].imshow(cp.asnumpy(cp.angle(obj)))
        #     ax[2].imshow(cp.asnumpy(cp.abs(probe)))
        #     ax[3].imshow(cp.asnumpy(cp.angle(probe)))
        #     # ax[4].plot(cp.asnumpy(error_obj_list), 'o-', label='obj')
        #     ax[4].grid()
        #     plt.show()
 
        gradient_o, gradient_p, wavefronts = calculate_gradients_obj_probe(obj, probe, positions, data)

        # tv_reg, tv_gradient = total_variation_regularization(obj, weight=weight_tv) #TODO: test TV regularization

        beta_o, beta_p = 0, 0
        if optimizer == 'conjugate_gradient':
            if gradient_o_prev is not None and gradient_p_prev is not None:
                beta_o = cp.sum(cp.conj(gradient_o) * (gradient_o - gradient_o_prev)) / cp.sum(cp.conj(gradient_o_prev) * gradient_o_prev)
                beta_p = cp.sum(cp.conj(gradient_p) * (gradient_p - gradient_p_prev)) / cp.sum(cp.conj(gradient_p_prev) * gradient_p_prev)
            else:
                direction_o = gradient_o # + tv_gradient
                direction_p = gradient_p # + tv_gradient

        obj,probe, direction_o, direction_p = update_parameters(obj, probe, gradient_o, gradient_p, direction_o, direction_p, beta_o, beta_p, step_o, step_p, optimizer)
        
        if probe_support is not None:
            probe = probe*probe_support[0]

        gradient_o_prev, gradient_p_prev = gradient_o, gradient_p

        error[i] = calculate_errors(data,wavefronts,algo_inputs) 
        if i>0 and error[i] > 1000*error[i-1]:
            fig, ax = plt.subplots(figsize=(10, 7)) 
            ax.plot(error[:i].get(), 'o-', label='error')
            ax.grid()
            raise ValueError('Error increased by a factor of 1000. Exiting...')


    return obj.get(), probe.get(), error.get()
