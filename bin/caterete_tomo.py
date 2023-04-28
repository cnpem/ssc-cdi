import sys, json

""" Sirius Scientific Computing Imports """
from sscCdi.tomo.tomo_processing import tomo_sort, tomo_crop, tomo_unwrap, tomo_equalize, tomo_alignment, tomo_recon, tomo_equalize3D
from sscCdi.caterete.cat_tomo_processing import read_data

dic = json.load(open(sys.argv[1])) # input dictionary

if dic["processing_steps"]["read"]: 
    object, angles = read_data(dic["sinogram_path"],dic["recon_method"])

if dic["processing_steps"]["sort"]: 
    tomo_sort(dic,object, angles)

if dic["processing_steps"]["crop"]:  
    tomo_crop(dic)

if dic["processing_steps"]["unwrap"]:
    tomo_unwrap(dic)

if dic["processing_steps"]["equalize2D"]:
    tomo_equalize(dic)

if dic["processing_steps"]["alignment"]: 
    tomo_alignment(dic)

if dic["processing_steps"]["tomography"]: 
    tomo_recon(dic)

if dic["processing_steps"]["equalize3D"]:
    tomo_equalize3D(dic)
