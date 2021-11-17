"""Miscellaneous utilities for MEG/EEG field modelling """

from .compare_fields import compare_fields
from .fieldcomp import dipfld_sph, biot_savart, magdipfld
from .spharm import sph_X_fixed, sph_v
from .volume_slicer import VolumeSlicer
