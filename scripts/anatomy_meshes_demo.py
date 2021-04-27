#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 16:26:57 2021

@author: andrey
"""
# On Wed, Apr 21, 2021 at 4:10 PM Jussi Nurminen <jnu@iki.fi> wrote:
#     Actually, it's trivial to read the meshes from mne example files yourself. Here's the code:

import pathlib
import mne
from mne.io.constants import FIFF


data_path = pathlib.Path(mne.datasets.sample.data_path())
bem_file = data_path / 'subjects/sample/bem/sample-5120-5120-5120-bem-sol.fif'
bs = mne.read_bem_surfaces(bem_file)

# head
head = [s for s in bs if s['id'] == FIFF.FIFFV_BEM_SURF_ID_HEAD][0]
assert head['coord_frame'] == FIFF.FIFFV_COORD_MRI
head_pts, head_tris = head['rr'], head['tris']

# brain
brain = [s for s in bs if s['id'] == FIFF.FIFFV_BEM_SURF_ID_BRAIN][0]
assert brain['coord_frame'] == FIFF.FIFFV_COORD_MRI
brain_pts, brain_tris = brain['rr'], brain['tris']



#%% ...and here's some code for plotting the actual cortical surface, with folds and all:

import pathlib
import mne

from megsimutils.viz import _mlab_trimesh


data_path = pathlib.Path(mne.datasets.sample.data_path())
subjects_dir = data_path / 'subjects'
subject = 'sample'
# source spacing; normally 'oct6', 'oct4' for sparse source space
SRC_SPACING = 'oct6'

# create the volume source space
src_cort = mne.setup_source_space(
    subject, spacing=SRC_SPACING, subjects_dir=subjects_dir, add_dist=False
)

# visualize
# src_cort is indexed by hemisphere (0=left, 1=right)
# separate meshes for left & right hemi
_mlab_trimesh(src_cort[0]['rr'], src_cort[0]['tris'])

