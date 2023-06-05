from sys import argv
import numpy as np
import time, json

""" Sirius Scientific Computing Imports """
import sscCdi

#+++++++++++++++++++++++++++++++++++++++++++++++++
#
# MAIN APPLICATION (for SIRIUS CATERETE beamline)
#
# +++++++++++++++++++++++++++++++++++++++++++++++++

np.random.seed(1)  # define seed for generation of the same random values

if __name__ == '__main__':

    t0 = time.time()

    input_dict = json.load(open(argv[1]))  # open input_dict file containing desired inputs

    print("\nCreating folders...")
    input_dict = sscCdi.carnauba.cnb_ptycho_processing.define_paths(input_dict)

    t1 = time.time()
    print('Starting restoration...')
    restoration_dict_list, restored_data_info_list = sscCdi.caterete.cat_restoration.restoration_CAT(input_dict) # restoration of all frames; restored DPs saved at output temporary folder
    t2 = time.time()

    input_dict, object,probe, probe_positions = sscCdi.caterete.cat_ptycho_processing.cat_ptychography(input_dict,restoration_dict_list,restored_data_info_list)
    t3 = time.time()

    print('Finished reconstruction!\n')

    """ ===================== Save and preview data ===================== """
    object, probe = sscCdi.misc.save_volume_from_parts(input_dict)

    print('\nSaving Object of shape: ',object.shape)
    sscCdi.misc.save_variable(input_dict, object,name='object')

    print('\nSaving Probe of shape: ',probe.shape)
    sscCdi.misc.save_variable(input_dict,probe,name='probe')

    sscCdi.misc.save_json_logfile(input_dict) 
    sscCdi.misc.delete_temporary_folders(input_dict)

    t4 = time.time()
    time_elapsed_restauration = t2 - t1
    time_elapsed_ptycho = t3 - t2
    print('\n')
    print(f'Restoration time:     {time_elapsed_restauration:.2f} seconds = {(time_elapsed_restauration) / 60:.2f} minutes ({100*(time_elapsed_restauration)/(t6 - t0):.0f}%)')
    print(f'Ptychography time:     {time_elapsed_ptycho:.2f} seconds = {(time_elapsed_ptycho) / 60:.2f} minutes ({100*(time_elapsed_ptycho)/(t6 - t0):.0f}%)')
    print(f'Save time:             {t4 - t3:.2f} seconds = {(t4 - t3) / 60:.2f} minutes ({100*(t4 - t3)/(t4 - t0):.0f}%)')
    print(f'Total time:            {t4 - t0:.2f} seconds = {(t4 - t0) / 60:.2f} minutes')
