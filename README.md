## Instructions for use (for conda)

- Create and activate a suitable Python environment
    - Easiest way is to download `environment.yml` from the package root and run `conda env create`
    - Activate the environment by `conda activate megsim`
- Clone repository and cd into the directory
- Install package by `pip install -e .`
- You should now be able to run plotting scripts etc.
- Package can be updated simply by `git pull`

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
