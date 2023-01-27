## Overview 
This repository contain the code needed to reproduce the results described in
the paper on MEG sensor array optimization by Zhdanov, Nurminen, Iivanainen
and Taulu expected to be published somewhen around 2023. 

The code is written in Python and uses Anaconda for installation.

## Instructions for use
The instructions assume that you have Anaconda installed.

### Installation
- Create the conda environment. From the root folder of this repository
(the one that contains `environment.yml`) run `conda env create`
- Activate the environment by `conda activate megsim`
- Install the package by `pip install -e .` from the root folder 

### Running
- Run the optimizations for the 2D and 3D imaging volumes. From the `opt`
subfolder run `python opt_run_2D.py /path/to/output/folder` (for the 2D 
imaging volume) or `python opt_run_3D.py /path/to/output/folder` (for the 3D 
imaging volume). The optimization process will run for several days and save
the results to the  `/path/to/output/folder`.  
- Reproduce the paper's figures by running the scripts from the `make_figures`
subfolder. The scripts receive the path to the optimization results as a
command-line parameter. For example, if you have your optimization results in
the `/path/to/output/folder`, you can view the sensor geometry progression by
running `python sens_geom_opt.py /path/to/output/folder`  

Note, that you don't need to wait for the optimization process to finish to
start looking at the results. The optimization saves intermediate results to
the output folder as it progresses; the scripts from the `make_figures`
subfolder can read and visualize partial results.  

For more information see the comments in the files.

## Notes

- The plots that require human anatomical data get the data from the MNE
  sample data set. If the dataset is not present on the computer, MNE
  functions will try to download it automatically. The location to which
  dataset is downloaded can be controlled by MNE_DATASETS_SAMPLE_PATH
  environment variable.

- For spherical coordinates, the code uses the convention of Taulu and
  Kajola 2005, Hill 1954, etc. That is, the angles are (theta, phi), where:  
    - theta: polar (colatitudinal) coordinate; must be between 0 and pi.  
    - phi: azimuthal (longitudinal) coordinate; must be between 0 and 2*pi.  
    This is different from the scipy convention.
