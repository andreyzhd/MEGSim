#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File related util functions for megsim.

"""
from pathlib import Path
import subprocess
import platform
import os
import tempfile
import numpy as np


def _named_tempfile(suffix=None):
    """Return a name for a temporary file.
    Does not open the file. Cross-platform. Replaces tempfile.NamedTemporaryFile
    which behaves strangely on Windows.
    """
    if suffix is None:
        suffix = ''
    elif suffix[0] != '.':
        raise ValueError('Invalid suffix, must start with dot')
    return os.path.join(tempfile.gettempdir(), os.urandom(24).hex() + suffix)


def _montage_figs(fignames, montage_fn, ncols_max=None):
    """Montages a bunch of figures into montage_fname.

    fignames is a list of figure filenames.
    montage_fn is the resulting montage name.
    ncols_max defines max number of columns for the montage.
    """
    if ncols_max is None:
        ncols_max = 4
    # educated guess for the location of the montage binary
    if platform.system() == 'Linux':
        MONTAGE_CMD = '/usr/bin/montage'
    else:
        MONTAGE_CMD = 'C:/Program Files/ImageMagick-7.0.10-Q16/montage.exe'
    if not Path(MONTAGE_CMD).exists():
        raise RuntimeError('montage binary not found, cannot montage files')
    # set montage geometry
    nfigs = len(fignames)
    geom_cols = ncols_max
    geom_rows = int(np.ceil(nfigs / geom_cols))  # figure out how many rows we need
    geom_str = f'{geom_cols}x{geom_rows}'
    MONTAGE_ARGS = ['-geometry', '+0+0', '-tile', geom_str]
    # compose a list of arguments
    theargs = [MONTAGE_CMD] + MONTAGE_ARGS + fignames + [montage_fn]
    print('running montage command %s' % ' '.join(theargs))
    subprocess.call(theargs)  # use call() to wait for completion

