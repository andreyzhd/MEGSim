## Instructions for use (for conda)

- Activate a suitable Python environment (should have numpy, matplotlib and such)
- Clone repository and cd into the directory
- Install package by `python setup.py develop`
- You should now be able to run plotting scripts etc.

## Notes

- For spherical coordinates, the code uses the convention of Taulu and Kajola 2005, Hill 1954, etc. That is, the angles are (theta, phi), where:  
    - theta: polar (colatitudinal) coordinate; must be between 0 and pi.  
    - phi: azimuthal (longitudinal) coordinate; must be between 0 and 2*pi.  
    This is different from the scipy convention.
