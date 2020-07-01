## Instructions for use (for conda)

- Create and activate a suitable Python environment
    - Easiest way is to download `environment.yml` from the package root and run `conda env create`
    - Activate the environment by `conda activate megsim`
- Clone repository and cd into the directory
- Install package by `python setup.py develop`
- You should now be able to run plotting scripts etc.
- Package can be updated simply by `git pull`

## Notes

- For spherical coordinates, the code uses the convention of Taulu and Kajola 2005, Hill 1954, etc. That is, the angles are (theta, phi), where:  
    - theta: polar (colatitudinal) coordinate; must be between 0 and pi.  
    - phi: azimuthal (longitudinal) coordinate; must be between 0 and 2*pi.  
    This is different from the scipy convention.
