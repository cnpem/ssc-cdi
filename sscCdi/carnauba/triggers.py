import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import medfilt

def ParseTriggers(trigg):

    state = False
    highedge = []
    lowedge = []

    for k in range(trigg.size):
        if trigg[k] == True and state == False:
            highedge.append(k)
            state = True
        elif trigg[k] == False and state == True:
            lowedge.append(k)
            state = False

    return highedge, lowedge

def GetBeamlineParams(jason):

    name = jason["Ibira_Path"] + jason["Proposal"] + '/proc/' + jason["BeamlineParameters_Filename"]

    data = h5py.File(name,'r')
    highedge,lowedge = ParseTriggers(np.asarray(data['CRIO/Triggers']) > 0.5) #returns lists

    encoder_posx = np.asarray(data['/CRIO/Encoder Piezo Horizontal']) - 0.06 * np.asarray(data['/CRIO/Capacitive Sensor Rz HFM'])

    encoder_posy = np.asarray(data['/CRIO/Encoder Piezo Vertical']) - 0.15 * np.asarray(data['/CRIO/Capacitive Sensor Rx VFM'])
    
    #encoder_posx = scipy.signal.medfilt(encoder_posx)
    #encoder_posy = scipy.signal.medfilt(encoder_posy)
    
    measured_I0 = np.asarray(data['/CRIO/I0'])
    measured_I1 = np.asarray(data['/CRIO/I1'])
    measured_Photodiode_HFM = np.asarray(data['/CRIO/Photodiode HFM'])
    energy = data['beamline_parameters/4CM Energy'][()]

    data.close()

    state = False
    posx = []
    posy = []
    I0 = []
    I1 = []
    Photodiode_HFM = []

    for k in range(len(highedge)):
        s = np.s_[highedge[k]:lowedge[k]]
        posx.append( encoder_posx[s][0::30] )
        posy.append( encoder_posy[s][0::30] )
        I0.append( measured_I0[s].mean() )
        I1.append( measured_I1[s].mean() )
        Photodiode_HFM.append( measured_Photodiode_HFM[s].sum() )

    posx = np.asarray(posx).astype(np.float32)
    posy = np.asarray(posy).astype(np.float32)

    posx -= posx.mean()
    posy -= posy.mean()

    beamline_params = {}
    beamline_params['I0'] = np.asarray(I0).astype(np.float32)
    beamline_params['I1'] = np.asarray(I1).astype(np.float32)
    beamline_params['posx'] = posx
    beamline_params['posy'] = posy
    beamline_params['energy'] = energy
    beamline_params['Photodiode_HFM'] = np.asarray(Photodiode_HFM).astype(np.float32)

    np.save('/ibira/lnls/beamlines/carnauba/apps/jupyter/posx.npy', posx)
    np.save('/ibira/lnls/beamlines/carnauba/apps/jupyter/posy.npy', posy)

    print("\nGot beamline parameters successfully!")
    return beamline_params

