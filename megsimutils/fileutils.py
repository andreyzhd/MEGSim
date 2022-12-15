#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
File related util functions for megsim.

"""
import sys
import pickle
import pathlib
from math import isclose, inf

from pathlib import Path
import subprocess
import platform
import os
import tempfile
import numpy as np
from megsimutils.utils import subsample

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

def read_opt_res(inp_path, max_n_samp=inf):
    """Read the results of optimization run"""
    interm_res = []

    # Read the starting timestamp, etc
    fl = open('%s/start.pkl' % inp_path, 'rb')
    params, t_start, v0, sens_array = pickle.load(fl)
    fl.close()

    interm_res.append((v0, sens_array.comp_fitness(v0), False, t_start))

    # Try to read the final result
    try:
        fl = open('%s/final.pkl' % inp_path, 'rb')
        opt_res, final_tstamp = pickle.load(fl)
        fl.close()
    except:
        opt_res = None
        print('Could not find the final result, using the last intermediate result instead')

    # Read the intermediate results
    file_list = sorted(pathlib.Path(inp_path).glob('iter*.pkl'))
    sys.setrecursionlimit(min(len(file_list), max_n_samp) + 1000)
    indx = subsample(len(file_list) + (opt_res is not None) + 1, max_n_samp)

    if opt_res is None:
        file_indx = (i - 1 for i in indx[1:])
    else:
        file_indx = (i - 1 for i in indx[1:-1])

    for i in file_indx:
        fname = file_list[i]
        print('Reading %s ...' % fname)
        fl = open(fname, 'rb')
        v, f, accept, tstamp = pickle.load(fl)
        fl.close()
        assert isclose(sens_array.comp_fitness(v), f, rel_tol=1e-4)
        interm_res.append((v, f, accept, tstamp))

    assert len(interm_res) > 1  # should have at least one intermediate result

    if opt_res is not None:
        interm_res.append((opt_res.x, sens_array.comp_fitness(opt_res.x), True, final_tstamp))

    return params, sens_array, interm_res, opt_res, indx
