# constrained-harmonic-resynthesis4timbre-clustering
This project builds on Constrained harmonic resynthesis model which is developed along with the Violin Etudes paper.

Constrained harmonic resynthesis model that can be used in labeling single-instrument, monophonic music performance datasets.

This package is pased on MTG/sms_tools. Following the instruction in their github page: 
to use the tools, after downloading the whole package, you need to compile some C functions. For that you should go to the directory sms_tools/models/utilFunctions_C and type:

$ python compileModule.py build_ext --inplace 

## key files
* `constrained_harmonic_resynthesis.py`: analyse `*.wav`, generate `*.npz` containing harmonic magnitudes (hmag) and resynthesise audio based on `*.npz`.
* `timbre_clustering.ipynb`: analyse hmag (timbre) data and do KMeans clustering to generate `*.tb.csv`.
