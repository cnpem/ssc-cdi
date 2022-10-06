Simply run:

	python3 runme_SciCompty.py
    
    python3 runme_SciCompty.py --enablegui yes  (default behaviour)
    python3 runme_SciCompty.py --enablegui no


Simulated data are in the synthdata folder.
The entire script processing should take less than 20s, 10s of pre-processing and 10s of actual processing.

Requirements are:
- numpy
- pytorch (> 1.10)
- scikit-image (> 0.17.2)
- cv2
- tifffile
- matplotlib
- h5py

I'm currently using Anaconda Python 3.8.12/3.9.7

My typical setup string is:

	conda install numpy scipy scikit-image scikit-learn matplotlib h5py joblib
	conda install -c conda-forge tifffile silx
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
	pip install opencv-contrib-python

