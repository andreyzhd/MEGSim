#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Util functions for megsim.

"""
import IPython


def _ipython_setup(enable_reload=False):
    """Performs some IPython magic if we are running in IPython"""
    try:
        __IPYTHON__
    except NameError:
        return
    from IPython import get_ipython

    ip = get_ipython()
    ip.magic("gui qt5")  # needed for mayavi plots
    #ip.magic("matplotlib qt")  # do mpl plots in separate windows
    if enable_reload:
        ip.magic("reload_ext autoreload")  # these will enable module autoreloading
        ip.magic("autoreload 2")
