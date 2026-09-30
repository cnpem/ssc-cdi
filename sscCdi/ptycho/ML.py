# Academic License Agreement:
#
# This license agreement sets forth the terms and conditions under which the Brazilian Center for Research in Energy and #Materials (CNPEM) (hereafter "LICENSOR")
#  will grant you (hereafter "LICENSEE") a royalty-free, non-exclusive license for #academic, non-commercial purposes only (hereafter "LICENSE") 
# to use the ssc-cdi computer software program and associated documentation furnished hereunder (hereafter "PROGRAM"). 
#
# For the complete LICENSE description see LICENSE file available within the root directory of this project.
##################################################################################################################################################################
import numpy as np
from datetime import datetime


def ML_LSQ_cupy(data, positions, initial_obj, initial_probe, algo_inputs, GPU = [0]):
    import gc
    try:
        import cupy as cpnp
        cpnp.cuda.runtime.setDevice(GPU[0])
    except:
        print("Could not import cupy, reverting to numpy!")
        cpnp = np
    """
    Performs the ML (Maximum Likelihood) algorithm using CuPy for accelerated computation.  
    *WARNING:* This function is a test engine and is still in development.
    Args:
        data (array-like): The measured data.
        positions (array-like): The positions of the measurements.
        initial_obj (array-like): The initial guess for the object.
        initial_probe (array-like): The initial guess for the probe.
        algo_inputs (dict): A dictionary containing algorithm inputs including:
            - iterations (int): The number of iterations.
            - probe_update_start (int, default 0)
            - beta_cap (float or None)
            - restart_if_not_descent (bool)
            - verbose (bool)
            - precond (bool)
            - probe_support_array (array-like): The probe support array.
    Returns:
        tuple: A tuple containing the updated object, updated probe, and the error array.
    """
    mempool = cpnp.get_default_memory_pool()
    pinned_mempool = cpnp.get_default_pinned_memory_pool()
    ## Quick dot product

    def cdot(a, b):
        return cpnp.real(cpnp.vdot(a, b))

    def gather_patches(arr2d, pos_batch, yy, xx):
        py = pos_batch[:, 0][:, None, None]
        px = pos_batch[:, 1][:, None, None]
        return arr2d[py + yy, px + xx]

    def scatter_add_patches(dest2d, pos_batch, patch_batch, yy, xx, obj_w):
        """
        Scatter-add complex patch_batch into complex dest2d using cpnp.add.at
        by splitting real and imaginary parts.
        """
        py = pos_batch[:, 0][:, None, None]
        px = pos_batch[:, 1][:, None, None]

        lin_idx = ((py + yy) * obj_w + (px + xx)).ravel()

        dest_flat = dest2d.ravel()
        patch_flat = patch_batch.ravel()

        cpnp.add.at(dest_flat.real, lin_idx, patch_flat.real)
        cpnp.add.at(dest_flat.imag, lin_idx, patch_flat.imag)

    ## Inputs
    data = cpnp.asarray(data, dtype=cpnp.float32)
    positions = cpnp.asarray(positions, dtype=cpnp.int32)
    obj = cpnp.asarray(initial_obj, dtype=cpnp.complex64).copy()
    probe = cpnp.asarray(initial_probe, dtype=cpnp.complex64).copy()
    if probe.shape[0] > 1:
        raise ValueError(f"ML-LSQ implementation only allows 1 probe mode, current probe has {probe.shape[0]} modes")
    ObjH, ObjW = obj.shape
    N, H, W = probe.shape
    npos = int(positions.shape[0])

    iterations = int(algo_inputs.get("iterations", 100))
    batch_size = int(algo_inputs.get("batch_size", 64))
    verbose = bool(algo_inputs.get("verbose", False))
    metric = str(algo_inputs.get("ML_metric", "amplitude"))
    obj_reg = float(algo_inputs.get("regularization_object", 1))
    probe_reg = float(algo_inputs.get("regularization_probe", 1))

    if metric not in ["amplitude"]:
        raise ValueError("'amplitude' are the available metrics")
    probe_support = algo_inputs.get("probe_support_array", None)
    optimization = str(algo_inputs.get("ML_optimization", "polynomial"))
    if probe_support is not None:
        probe_support = cpnp.asarray(probe_support, dtype=cpnp.float32)
    data_f = cpnp.fft.ifftshift(data, axes=(-2, -1))
    data_f = cpnp.maximum(data_f, 0.0)

    if verbose:
        # Warmup (useful for consistency when benchmarking)
        _ = cpnp.fft.fft2(cpnp.zeros((H, W), dtype=cpnp.complex64))
        # for m in range(15):
        #     print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")    
        #     print("*WARNING:* This function is a test engine and is still in development.")

    # Weighting: w = mask/(1 + data)
    if(metric == 'amplitude'):
        weights = cpnp.where(data >= 0, 1.0 / (2*(1.0 + data)), 0.0).astype(cpnp.float32)
        #weights = cpnp.where(data >= 0, 1, 0).astype(cpnp.int32)
    elif(metric == "poisson"):
        weights = cpnp.where(data >= 0, 1, 0).astype(cpnp.int32)
    weights_f = cpnp.fft.ifftshift(weights, axes=(-2, -1))

    del(data)
    mempool.free_all_blocks()

    yy = cpnp.arange(H, dtype=cpnp.int32)[None, :, None]
    xx = cpnp.arange(W, dtype=cpnp.int32)[None, None, :]


    gradient_o = cpnp.zeros_like(obj, dtype=cpnp.complex64)
    gradient_p = cpnp.zeros_like(probe, dtype=cpnp.complex64)
    denom_o = cpnp.zeros_like(obj, dtype=cpnp.complex64)
    denom_p = cpnp.zeros_like(probe, dtype=cpnp.float64)
    update_o = cpnp.zeros_like(obj, dtype=cpnp.complex64)

    llk =  cpnp.empty(iterations, dtype=cpnp.float32)
    mse =  cpnp.empty(iterations, dtype=cpnp.float32)
    r_factor =  cpnp.empty(iterations, dtype=cpnp.float32)
    r_factor_norm = cpnp.sum(cpnp.sqrt(data_f))
    mse_norm = H*W

    for it in range(iterations):

        llk_temp = cpnp.array(0.0, dtype=cpnp.float64)  
        r_factor_temp = cpnp.array(0.0, dtype=cpnp.float64)
        mse_temp = cpnp.array(0.0, dtype=cpnp.float64)

        for start in range(0, npos, batch_size):
            
            gradient_o.fill(0.0)
            gradient_p.fill(0.0)
            update_o.fill(0.0)
            denom_o.fill(0.0)
            denom_p.fill(0.0)

            probe_conj = cpnp.conj(probe)

            end = min(start + batch_size, npos)
            pos_batch = positions[start:end].copy()
            n_batch = data_f[start:end]
            w_batch = weights_f[start:end]
            obj_rois = gather_patches(obj, pos_batch, yy, xx)

            # Model
            exit_batch = obj_rois * probe[0]
            psi_batch = cpnp.fft.fft2(exit_batch, axes=(-2, -1))
            intensity = cpnp.maximum(cpnp.abs(psi_batch)** 2, 0.0)
            exit_batch_new = cpnp.fft.ifft2((cpnp.sqrt(n_batch)/(cpnp.sqrt(intensity) + 1e-9)) * psi_batch, axes=(-2, -1))
            xi_batch =  exit_batch - exit_batch_new 
            #print(f"xi: {cpnp.amin(xi_batch)}")

            # Gradients batchs
            grad_o_patch = 2.0 * xi_batch * probe_conj[0]
            grad_p_patch = 2.0 * xi_batch * cpnp.conj(obj_rois)
            #print(f"Grads: {cpnp.amin(grad_o_patch)}, {cpnp.amin(grad_p_patch)}")
            scatter_add_patches(gradient_o, pos_batch, grad_o_patch, yy, xx, ObjW)
            gradient_p += cpnp.sum(grad_p_patch, axis=0)

            #Gradients normalization
            scatter_add_patches(denom_o, pos_batch, cpnp.ones_like(grad_o_patch) * cpnp.abs(probe[0])**2, yy, xx, ObjW)
            denom_p += cpnp.sum(cpnp.abs(obj_rois)**2, axis = 0)
            #print(f"Grads denom: {cpnp.amin(denom_o)}, {cpnp.amin(denom_p)}")

            #Gradients
            gradient_o = gradient_o/(cpnp.real(denom_o) + obj_reg)
            gradient_p = gradient_p/(denom_p + probe_reg)
            #print(f"Grads final: {cpnp.amin(gradient_o)}, {cpnp.amin(gradient_p)}")

            gradient_o_rois = gather_patches(gradient_o, pos_batch, yy, xx)
            ##################################################################
            #This is the orthogonal probe and object gradients approximation
            #The code solves the linear system to obtain the steps, maybe can become an option on the future
            # #Step batchs
            # step_o_patch = cpnp.sum(cpnp.real(xi_batch * cpnp.conj(gradient_o_rois * probe[0])), axis = (1,2))
            # step_p_patch =  cpnp.sum(cpnp.real(xi_batch * cpnp.conj(gradient_p * obj_rois)), axis = (1,2))
            # #print(f"Steps: {cpnp.amin(step_o_patch)}, {cpnp.amin(step_p_patch)}")

            # #Steps normalization
            # denom_step_o = cpnp.sum(cpnp.abs(gradient_o_rois * probe[0])**2, axis = (1,2))
            # denom_step_p = cpnp.sum(cpnp.abs(gradient_p * obj_rois)**2, axis = (1,2))
            # #print(f"Steps denom: {cpnp.amin(denom_step_o)}, {cpnp.amin(denom_step_p)}")

            # #Steps 
            # step_o = cpnp.maximum(step_o_patch/(denom_step_o + 1e-6), 0.0)
            # step_p = cpnp.maximum(step_p_patch/(denom_step_p + 1e-6), 0.0)
            # #print(f"Steps final: {cpnp.amin(step_o)}, {cpnp.amin(step_p)}")
            ########################################################################

            # Generalized step calculation

            A1 = cpnp.sum(cpnp.abs(gradient_o_rois*probe[0])**2 + 1e-9 , axis = (1,2))
            A2 = cpnp.sum(cpnp.abs(gradient_p*obj_rois)**2 + 1e-9, axis = (1,2))
            A3 = cpnp.sum((gradient_o_rois*probe[0])*cpnp.conj(gradient_p*obj_rois), axis = (1,2))
            C1 = cpnp.sum(cpnp.real(xi_batch * cpnp.conj(gradient_o_rois * probe[0])), axis = (1,2))
            C2 = cpnp.sum(cpnp.real(xi_batch * cpnp.conj(gradient_p * obj_rois)), axis = (1,2))
            D = A1*A2 - cpnp.abs(A3)**2

            step_o = (C1*A2 - C2*A3)/(D + 1e-9)
            step_p = (C2*A1 - C1*cpnp.conj(A3))/(D + 1e-9)
            step_o = cpnp.maximum(step_o, 0.0)
            step_p = cpnp.maximum(step_p, 0.0)
            #Updates Object and Probe
            scatter_add_patches(update_o, pos_batch,step_o[:, None, None]*gradient_o_rois*cpnp.abs(probe[0])**2, yy, xx, ObjW)

            obj -= update_o/(cpnp.real(denom_o) + 1e-9)
            probe -= cpnp.sum(step_p[:, None, None]*gradient_p*cpnp.abs(obj_rois)**2, axis = 0)/(denom_p + 1e-9)
            probe = probe*probe_support

            # error metrics
            llk_temp -= cpnp.sum(w_batch * (n_batch*cpnp.log(intensity) - intensity))
            r_factor_temp += cpnp.sum(w_batch*cpnp.abs(cpnp.sqrt(intensity) - cpnp.sqrt(n_batch))) 
            mse_temp += cpnp.sum(w_batch*(intensity - n_batch)**2)
        r_factor[it] = r_factor_temp
        llk[it] = llk_temp
        mse[it] = mse_temp

        ## Save prev grads
        if verbose and it%10 == 0:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S") + f":{now.microsecond:06d}"
            error1 = float((r_factor_temp/r_factor_norm).get())
            error2 = float(llk_temp.get())
            error3 = float((mse_temp/mse_norm).get())
            print(f"%[{timestamp}] : iter {it}/{iterations}, r_factor = {error1:.5e}, llk = {error2:.5e}, mse = {error3:.5e}")

    # Final sync and pool cleanup
    cpnp.cuda.Stream.null.synchronize()

    result_obj = obj.get()
    result_probe = probe.get()
    result_positions = positions.get()
    r_factor = (r_factor/r_factor_norm).get()
    llk = llk.get()
    mse = (mse/(H*W)).get()

    # Delete GPU arrays
    del obj, probe
    del gradient_o, gradient_p, denom_o, denom_p
    del data_f, positions, weights_f, yy, xx
    del step_o, step_p     
    del probe_conj
    del obj_rois, exit_batch, psi_batch, intensity, xi_batch, grad_o_patch, grad_p_patch

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()

    return result_obj, result_probe, r_factor, llk, mse, result_positions



def ML_CG_cupy(data_cpu, positions, initial_obj, initial_probe, algo_inputs, GPU = [0]):
    import gc
    import numpy as np
    try:
        import cupy as cpnp
        import cupyx.scipy as scipy 
        #cpnp = cpnp.get_array_module(wavefront) # make code agnostic to cupy and numpy
        cpnp.cuda.runtime.setDevice(GPU[0])
        mempool = cpnp.get_default_memory_pool()
        pinned_mempool = cpnp.get_default_pinned_memory_pool()

        # total memory in MB
        free_memory, total_memory = cpnp.cuda.Device().mem_info
        free_memory = free_memory / (1024**2)
        total_memory = total_memory / (1024**2)
    except:
        print("Could not import cupy, reverting to numpy!")
        cpnp = np
        import scipy

    """
    Performs the ML (Maximum Likelihood) algorithm using CuPy for accelerated computation.  
    *WARNING:* This function is a test engine and is still in development.
    Args:
        data (array-like): The measured data.
        positions (array-like): The positions of the measurements.
        initial_obj (array-like): The initial guess for the object.
        initial_probe (array-like): The initial guess for the probe.
        algo_inputs (dict): A dictionary containing algorithm inputs including:
            - iterations (int): The number of iterations.
            - probe_update_start (int, default 0)
            - beta_cap (float or None)
            - restart_if_not_descent (bool)
            - verbose (bool)
            - precond (bool)
            - probe_support_array (array-like): The probe support array.
    Returns:
        tuple: A tuple containing the updated object, updated probe, and the error array.
    """


    # Quick dot product
    def cdot(a, b):
        return cpnp.real(cpnp.vdot(a, b))

    def probe_orthogonalize(probe):
        probe_ortho = probe.copy()
        for i in range(1, probe.shape[0], 1):
            v = probe_ortho[i]
            for j in range(0, i, 1):
                u = probe_ortho[j]
                probe_ortho[i] -= u*cpnp.vdot(u, v)/cpnp.vdot(u,u)
        return probe_ortho


    def gather_patches(arr2d, pos_batch, yy, xx):
        py = pos_batch[:, 0][:, None, None]
        px = pos_batch[:, 1][:, None, None]
        return arr2d[py + yy, px + xx]

    def scatter_add_patches(dest2d, pos_batch, patch_batch, yy, xx, obj_w):
        """
        Scatter-add complex patch_batch into complex dest2d using cpnp.add.at
        by splitting real and imaginary parts.
        """
        py = pos_batch[:, 0][:, None, None]
        px = pos_batch[:, 1][:, None, None]

        lin_idx = ((py + yy) * obj_w + (px + xx)).ravel()

        dest_flat = dest2d.ravel()
        patch_flat = patch_batch.ravel()

        cpnp.add.at(dest_flat.real, lin_idx, patch_flat.real)
        cpnp.add.at(dest_flat.imag, lin_idx, patch_flat.imag)

    def load_data_GPU(data_cpu, metric):

        data = cpnp.asarray(data_cpu, dtype=cpnp.float32)
        data_f = cpnp.fft.ifftshift(data, axes=(-2, -1))
        data_f = cpnp.maximum(data_f, 0.0)
    
        if(metric == 'gauss' or metric == "amplitude"):
            weights = cpnp.where(data >= 0, 1.0 / (2*(1.0 + data)), 0.0).astype(cpnp.float32)
        elif(metric == "poisson"):
            weights = cpnp.where(data >= 0, 1, 0).astype(cpnp.int32)
        weights_f = cpnp.fft.ifftshift(weights, axes=(-2, -1))
        
        #del(data)
        #mempool.free_all_blocks()

        return data_f, weights_f


    ## Inputs
    iterations = int(algo_inputs.get("iterations", 100))
    batch_size = int(algo_inputs.get("batch_size", 64))
    probe_update_start = int(algo_inputs.get("probe_update_start", 1))
    beta_cap = algo_inputs.get("beta_cap", 0.99)
    force_restart = bool(algo_inputs.get("force_restart", True))
    verbose = bool(algo_inputs.get("verbose", True))
    precond = bool(algo_inputs.get("precond", True))
    metric = str(algo_inputs.get("metric", "poisson"))
    if metric not in ["gauss", "poisson", "amplitude"]:
        raise ValueError("'gauss', 'poisson'  and 'amplitude' are the available metrics")
    probe_support = algo_inputs.get("probe_support_array", None)
    refine_solution = bool(algo_inputs.get("refine_solution", False))
    if probe_support is not None:
        probe_support = cpnp.asarray(probe_support, dtype=cpnp.float32)
    regularize = float(algo_inputs.get("regularize", 0))
    
    # mask data to avoid doing it many times
    #data_cpu = data_cpu*np.where(data_cpu >= 0, 1, 0)

    #Important constants from data 
    N_counts = np.sum(data_cpu*np.where(data_cpu >= 0, 1, 0))
    N_counts_2 = np.sum((data_cpu*np.where(data_cpu >= 0, 1, 0))**2)
    N_counts_root = np.sum(np.sqrt(data_cpu*np.where(data_cpu >= 0, 1, 0)))

    #This is temporary, the correct threshold value must be determined from profiling (to do)
    if(data_cpu.nbytes/ (1024**2) < 0.25 * free_memory):
        print("Can load all data to GPU, procedding!")
        data_f, weights_f = load_data_GPU(data_cpu, metric)
        del(data_cpu)
        data_load_batch = False
    else:
        data_load_batch = True

    positions = cpnp.asarray(positions, dtype=cpnp.int32)
    obj = cpnp.asarray(initial_obj, dtype=cpnp.complex64).copy()
    probe = cpnp.asarray(initial_probe, dtype=cpnp.complex64).copy()

    ObjH, ObjW = obj.shape
    N_modes, H, W = probe.shape
    npos = int(positions.shape[0])

    #This is intended to make the regularization value invariant to the data
    #Those values do not work, but it is a 1-to-1 of the paper description
    # if regularize:
    #     K_reg = (8.0 * (ObjH*ObjW)**2) / (npos *  N_valid * N_counts)
    #     scale_regularizer = 0.01 /  (K_reg + 1e-9)
    #     print(K_reg, scale_regularizer)
    

    # Warmup (useful for consistency when benchmarking and the code is slow if you not use it)
    _ = cpnp.fft.fft2(cpnp.zeros((H, W), dtype=cpnp.complex64))

    yy = cpnp.arange(H, dtype=cpnp.int32)[None, :, None]
    xx = cpnp.arange(W, dtype=cpnp.int32)[None, None, :]

    #gradients
    gradient_o = cpnp.zeros_like(obj, dtype=cpnp.complex64)
    gradient_p = cpnp.zeros_like(probe, dtype=cpnp.complex64)
    hO = cpnp.zeros_like(obj, dtype=cpnp.complex64)
    hP = cpnp.zeros_like(probe, dtype=cpnp.complex64)
    gradient_o_prev = None
    gradient_p_prev = None

    # error metrics
    llk =  cpnp.empty(iterations, dtype=cpnp.float32)
    mse =  cpnp.empty(iterations, dtype=cpnp.float32)
    r_factor =  cpnp.empty(iterations, dtype=cpnp.float32)
    mse_norm = N_counts_2
    r_factor_norm = N_counts_root
    llk_norm = N_counts

    #usefull
    k_probe_start = 0

    for it in range(iterations):
        gradient_o.fill(0)
        gradient_p.fill(0)
        
        # This is brute force restart of the CG 
        # It at best help the minimization and at worst do nothing 
        if(it%100 == 0 and force_restart == True):
            hO.fill(0)
            hP.fill(0)
            gradient_o_prev = None
            gradient_p_prev = None
            k_probe_start = 0

        if(N_modes > 1):
            probe = probe_orthogonalize(probe)

        llk_temp = cpnp.array(0.0, dtype=cpnp.float64)  
        r_factor_temp = cpnp.array(0.0, dtype=cpnp.float64)
        mse_temp = cpnp.array(0.0, dtype=cpnp.float64)
        probe_conj = cpnp.conj(probe)

        for start in range(0, npos, batch_size):

            end = min(start + batch_size, npos)
            pos_batch = positions[start:end].copy()
            obj_rois = gather_patches(obj, pos_batch, yy, xx)

            if(data_load_batch):
                n_batch, w_batch = load_data_GPU(data_cpu[start:end], metric)
            else:    
                n_batch = data_f[start:end]
                w_batch = weights_f[start:end]


            # Model
            intensity = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
            psi_batch = cpnp.empty((N_modes, n_batch.shape[0], H, W), dtype = cpnp.complex64)
            for mode in range(N_modes):
                exit_batch = obj_rois * probe[mode]
                psi_batch[mode] = cpnp.fft.fft2(exit_batch, axes=(-2, -1))
                intensity += cpnp.abs(psi_batch[mode]) ** 2

            for mode in range(N_modes):
                if(metric == "gauss"):
                    xi_batch = cpnp.fft.ifft2(w_batch * (intensity - n_batch) * psi_batch[mode], axes=(-2, -1))
                elif(metric == "poisson"):
                    xi_batch = cpnp.fft.ifft2(w_batch * ((n_batch/(intensity + 1e-6)) - 1) * psi_batch[mode], axes = (-2, -1))
                elif(metric == "amplitude"):
                    xi_batch = cpnp.fft.ifft2(w_batch * 2.0 * (1 - cpnp.sqrt(n_batch/(intensity + 1e-6))) * psi_batch[mode], axes = (-2, -1))
                # Gradients
                grad_o_patch = xi_batch * probe_conj[mode]
                scatter_add_patches(gradient_o, pos_batch, grad_o_patch, yy, xx, ObjW)
                gradient_p[mode] += cpnp.sum(xi_batch * cpnp.conj(obj_rois), axis=0)

            # error metrics 
            # generalized Kullback-Leibler divergence
            llk_temp += cpnp.sum(w_batch * ((n_batch   - intensity) - (n_batch + 1e-9)*cpnp.log((intensity + 1e-9)/((n_batch + 1e-9) + cpnp.where(n_batch == 0, 1, 0)))))
            # r_factor
            r_factor_temp += cpnp.sum(w_batch*cpnp.abs(cpnp.sqrt(intensity) - cpnp.sqrt(n_batch))) 
            #normalized mean  square error
            mse_temp += cpnp.sum(w_batch*(intensity - n_batch)**2)
            
        r_factor[it] = r_factor_temp
        llk[it] = llk_temp
        mse[it] = mse_temp

        #regularization
        if (regularize > 0):
            gradient_o = gradient_o + regularize*scipy.ndimage.laplace(obj)

        ## Update probe only if a given number of iters have been calculated
        if (k_probe_start < probe_update_start):
            gradient_p.fill(0)
            k_probe_start += 1

        ## Enforce probe support to probe gradient
        if probe_support is not None and it >= probe_update_start:
            gradient_p = gradient_p * probe_support

        ## Preconditioning coefficient
        if precond:
            # s^2 = ||gradient_o||^2 / ||gradient_p||^2
            denom = cdot(gradient_p,gradient_p) + 1e-9
            s_sq = cdot(gradient_o,gradient_o) / denom
        else:
            s_sq = cpnp.array(1.0, dtype=cpnp.float32)

        ## Beta for Polak–Ribière (with optional preconditioning)
        if gradient_o_prev is None:
            beta = cpnp.array(0.0, dtype=cpnp.float32)
        else:
            beta_num = (s_sq * (cdot(gradient_p,gradient_p) - cdot(gradient_p, gradient_p_prev))
                      + (cdot(gradient_o,gradient_o) - cdot(gradient_o, gradient_o_prev)))
            beta_den = (s_sq * cdot(gradient_p_prev,gradient_p_prev) + cdot(gradient_o_prev,gradient_o_prev) + 1e-6)
            beta = cpnp.maximum(beta_num / beta_den, 0.0)
            #beta = beta_num / beta_den
            if beta_cap is not None:
                beta = cpnp.minimum(beta, beta_cap)

        ## Update directions
        hO = beta * hO - gradient_o
        hP = beta * hP - (s_sq * gradient_p)

        ## Optional descent restart
        if force_restart:
            dot = cpnp.real(cpnp.vdot(gradient_o.ravel(), hO.ravel()) + cpnp.vdot(gradient_p.ravel(), hP.ravel()))
            if dot.get() > 0:
                beta = cpnp.array(0.0, dtype=cpnp.float32)
                hO = -gradient_o
                hP = -(s_sq * gradient_p)

        ## Apply probe support to direction too
        if probe_support is not None and it >= probe_update_start:
            hP = hP * probe_support

        ## Last step: find gamma using eqs. from Appendix B and considering as a 2nd degree polynomial
        B1 = cpnp.array(0.0, dtype=cpnp.float64)
        B2 = cpnp.array(0.0, dtype=cpnp.float64)
        for start in range(0, npos, batch_size):
            end = min(start + batch_size, npos)
            pos_batch = positions[start:end].copy()
            
            if(data_load_batch):
                n_batch, w_batch = load_data_GPU(data_cpu[start:end], metric)
            else:    
                n_batch = data_f[start:end]
                w_batch = weights_f[start:end]

            obj_rois = gather_patches(obj, pos_batch, yy, xx)
            hO_rois = gather_patches(hO, pos_batch, yy, xx)

            I0 = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
            A0 = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
            A1 = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
            A2 = cpnp.zeros_like(n_batch, dtype = cpnp.float32)

            for mode in range(N_modes):

                psi_batch = cpnp.fft.fft2(obj_rois * probe[mode], axes=(-2, -1))
                a_batch = cpnp.fft.fft2(probe[mode]* hO_rois + hP[mode] * obj_rois,axes=(-2, -1))
                b_batch = cpnp.fft.fft2(hP[mode] * hO_rois, axes=(-2, -1))

                I0 += cpnp.abs(psi_batch) ** 2
                A1 += 2.0 * cpnp.real(psi_batch * cpnp.conj(a_batch))
                A2 += 2.0 * cpnp.real(psi_batch * cpnp.conj(b_batch)) + cpnp.abs(a_batch) ** 2

            A0 = I0 - n_batch

            # this will create the effective coeficient of the L error parametrization
            if(metric == 'gauss'):
                B1 += cpnp.sum(w_batch * (2.0 * A0 * A1))
                B2 += cpnp.sum(w_batch * (A1**2 + 2.0 * A0 * A2))
            elif(metric == "poisson"):
                B1 += cpnp.sum(w_batch * A1 * (-1 + n_batch/(I0 + 1e-6)))
                B2 += cpnp.sum(w_batch * ((n_batch/(I0 + 1e-6)) * (A2 - (A1**2)/(2*I0 + 1e-6)) - A2))
            elif(metric == "amplitude"):
                B1 += cpnp.sum(w_batch * A1 * (1 - cpnp.sqrt(cpnp.maximum(n_batch / (I0 + 1e-6), 0.0))))
                B2 += cpnp.sum(w_batch * (A2 - cpnp.sqrt(cpnp.maximum(n_batch / (I0 + 1e-6), 0.0)) * (A2 - ((A1**2) / (4*I0 + 1e-6)))))

        gamma  = cpnp.where(cpnp.abs(B2) > 1e-12, -0.5*B1/B2, 0.0) 
        
        #Optional : Newton-Raphson step
        if refine_solution:
            gamma_trial = gamma.copy()
            L0 = cpnp.inf
            for n_nrint in range(3):
                L = 0.0
                dL = 0.0
                d2L = 0.0
                for start in range(0, npos, batch_size):
                    end = min(start + batch_size, npos)
                    pos_batch = positions[start:end].copy()

                    if(data_load_batch):
                        n_batch, w_batch = load_data_GPU(data_cpu[start:end], metric)
                    else:    
                        n_batch = data_f[start:end]
                        w_batch = weights_f[start:end]

                    obj_rois = gather_patches(obj, pos_batch, yy, xx)
                    hO_rois = gather_patches(hO, pos_batch, yy, xx)
                    
                    I = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
                    dI = cpnp.zeros_like(n_batch, dtype = cpnp.float32)
                    d2I = cpnp.zeros_like(n_batch, dtype = cpnp.float32)

                    for mode in range(N_modes):
                        psi_batch = cpnp.fft.fft2(obj_rois * probe[mode], axes=(-2, -1))
                        a_batch = cpnp.fft.fft2(probe[mode]* hO_rois + hP[mode] * obj_rois,axes=(-2, -1))
                        b_batch = cpnp.fft.fft2(hP[mode] * hO_rois, axes=(-2, -1))

                        I0 = cpnp.abs(psi_batch) ** 2
                        A1 = 2.0 * cpnp.real(psi_batch * cpnp.conj(a_batch))
                        A2 = 2.0 * cpnp.real(psi_batch * cpnp.conj(b_batch)) + cpnp.abs(a_batch) ** 2
                        A3 = 2.0 * cpnp.real(b_batch * cpnp.conj(a_batch))
                        A4 = cpnp.abs(b_batch) ** 2
                        I += I0 + gamma_trial * A1 + (gamma_trial**2) * A2 + (gamma_trial**3) * A3 + (gamma_trial**4) * A4 
                        dI += A1 + (2.0*gamma_trial) * A2 + (3.0*gamma_trial**2) * A3 + (4.0*gamma_trial**3) * A4
                        d2I += (2.0) * A2 + (6.0*gamma_trial) * A3 + (12.0*gamma_trial**2) * A4
                    
                    # this will create the effective coeficient of the L error parametrization
                    if(metric == 'gauss'):
                        r = I - n_batch
                        L += cpnp.sum(w_batch * r**2)
                        dL += cpnp.sum(w_batch * r *dI)
                        d2L += cpnp.sum(w_batch*(r*d2I + dI**2))
                    elif(metric == "poisson"):
                        e = 1e-9
                        I = cpnp.maximum(I, e)
                        L -= cpnp.sum(w_batch * (n_batch*cpnp.log(I) - I))
                        dL -= cpnp.sum(w_batch*((n_batch/I) - 1)*dI)
                        d2L -= cpnp.sum(w_batch*((-1 + n_batch/I)*d2I - (n_batch/I**2)*dI**2))
                    elif(metric == 'amplitude'):
                        e = 1e-9
                        I = cpnp.maximum(I, e)
                        L += cpnp.sum(w_batch *(cpnp.sqrt(I) - cpnp.sqrt(n_batch))**2)
                        dL += cpnp.sum(2 * w_batch * (1 - cpnp.sqrt(n_batch / I)) * dI)
                        d2L += cpnp.sum(2 * w_batch * (cpnp.sqrt(n_batch / I) * (((dI**2)/(2*I)) - d2I) + d2I))
                   

                if(L0 > L and d2L > 0):
                    L0 = L
                    if(n_nrint > 0):
                        gamma  = gamma_trial.copy()
                    gamma_trial = gamma_trial - dL/(d2L + 1e-9)
                else:
                    break


        gamma = float(gamma.get())

        ## Update object and probe
        obj += gamma * hO
        probe += gamma * hP
        ## Use probe support after update too
        if probe_support is not None and it >= probe_update_start:
            probe = probe * probe_support
        ## Save prev grads
        gradient_o_prev = gradient_o.copy()
        gradient_p_prev = gradient_p.copy()
        if verbose and it%10 == 0:
            now = datetime.now()
            timestamp = now.strftime("%H:%M:%S") + f":{now.microsecond:06d}"
            error1 = float((r_factor_temp/r_factor_norm).get())
            error2 = float((llk_temp/llk_norm).get())
            error3 = float((mse_temp/mse_norm).get())
            print(f"%[{timestamp}] : iter {it}/{iterations}, r_factor = {error1:.5e}, llk = {error2:.5e}, mse = {error3:.5e}")

    # Final sync and pool cleanup
    cpnp.cuda.Stream.null.synchronize()

    result_obj = obj.get()
    result_probe = probe.get()
    result_positions = positions.get()
    r_factor = (r_factor/r_factor_norm).get()
    llk = (llk/llk_norm).get()
    mse = (mse/(mse_norm)).get()
    
    # Delete GPU arrays
    del obj, probe
    del gradient_o, gradient_p, hO, hP
    if(data_load_batch == False):
        del data_f, weights_f
    del  positions, yy, xx
    del gradient_o_prev, gradient_p_prev
    try:
        del B1, B2, I0, A1, A2
    except:
        del I, dI, d2I, I0, A1, A2, A3, A4
    del probe_conj, gamma
    del obj_rois, exit_batch, psi_batch, intensity, xi_batch, grad_o_patch

    mempool.free_all_blocks()
    pinned_mempool.free_all_blocks()
    gc.collect()

    return result_obj, result_probe, r_factor, llk, mse, result_positions
