# -*- coding: utf-8 -*-
"""海冰厚度规则网格"""
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap
from paths import paths

grids = xr.open_dataset(paths['grids_path'])

# 定义北极规则网格
def north_regular_grid():
    grid_x_n, grid_y_n = np.meshgrid(
        np.arange(grids['x'].min().values, grids['x'].max().values+1, 25000),
        np.arange(grids['y'].min().values, grids['y'].max().values+1, 25000)
    )
    return grid_x_n, grid_y_n
