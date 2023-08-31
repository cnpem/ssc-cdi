import numpy as np
from skimage.io import imsave, imread

folderpath = '/ibira/lnls/labs/tepui/proposals/20210062/yuri/Caterete/yuri-ssc-cdi/'
input_filename = 'difpadssum.npy'

data = np.load(folderpath+input_filename)

# Select real or imaginary part of data, if array is complex. Tiff only accepts real data
data_real = np.real(data)
data_imag = np.imag(data)

print('Data shape: ',data.shape)

if 0: # IF OUTPUT IS RAW

    filename = folderpath + input_filename[:-4] + '.raw'
    print('Output filename: ', filename)

    data.tofile(filename)

else: # IF OUTPUT IS TIFF

    filename = folderpath + input_filename[:-4] + '_real.tif'
    print('Output filename: ', filename)
    imsave(filename,data_real)

    # filename = folderpath + input_filename[:-4] + '_imag.tif'
    # print('Output filename: ', filename)
    # imsave(filename,data_imag)
