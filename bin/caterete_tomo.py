import sys, json

""" Sirius Scientific Computing Imports """
from sscCdi.tomo.tomo_processing import tomo_sort, tomo_crop, tomo_unwrap, tomo_equalize, tomo_alignment, tomo_recon, tomo_equalize3D
from sscCdi.caterete.cat_tomo_processing import read_data

dic = json.load(open(sys.argv[1])) # input dictionary

if dic["processing_steps"]["read"]: 
    object, angles, dic = read_data(dic["sinogram_path"],dic["recon_method"])

if dic["processing_steps"]["sort"]: 
    dic = tomo_sort(dic,object, angles)

if dic["processing_steps"]["crop"]:  
    dic = tomo_crop(dic)

if dic["processing_steps"]["unwrap"]:
    dic = tomo_unwrap(dic)

if dic["processing_steps"]["equalize2D"]:
    dic = tomo_equalize(dic)

if dic["processing_steps"]["alignment"]: 
    dic = tomo_alignment(dic)

if dic["processing_steps"]["tomography"]: 
    dic = tomo_recon(dic)

if dic["processing_steps"]["equalize3D"]:
    dic = tomo_equalize3D(dic)

# save dic with metadata
output_dict = json.dumps(dic,indent=3,sort_keys=True)
jsonFile = open(dic["output_dict_path"], "w")
jsonFile.write(output_dict)
jsonFile.close()