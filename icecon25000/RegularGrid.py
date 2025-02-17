# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import os
from pyproj import pyproj
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap

plt.rcParams['font.sans-serif'] = ['SimHei']

grids = xr.open_dataset(r"D:\zmh\icecon25000_data\RegularGrid\Grid_n25000.nc")
grids_s = xr.open_dataset(r"D:\zmh\icecon25000_data\RegularGrid\Grid_s25000.nc")

# 定义北极规则网格
def north_regular_grid():
    grid_x_n, grid_y_n = np.meshgrid(
        np.arange(grids['x'].values.min(), grids['x'].values.max(), 25000),
        np.arange(grids['y'].values.min(), grids['y'].values.max(), 25000)
    )
    return grid_x_n, grid_y_n

# 定义南极规则网格
def south_regular_grid():
    grid_x_s, grid_y_s = np.meshgrid(
        np.arange(grids_s['x'].values.min(), grids_s['x'].values.max(), 25000),
        np.arange(grids_s['y'].values.min(), grids_s['y'].values.max(), 25000)
    )
    return grid_x_s, grid_y_s

def generate_snapshot(grid_data, grid_x, grid_y, hemisphere, output_file, datesetname, snapshot_folder):
    """
    生成快视图.
    北极, 投影经度中心=-45°
    南极, 投影经度中心=0° 
    """
    try:
        if hemisphere == 'north':
            proj_crs = ccrs.NorthPolarStereo(central_longitude=-45)
            extent = [-180, 180, 60, 90]
            # 北极投影参数
            proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)
        else:  
            proj_crs = ccrs.SouthPolarStereo(central_longitude=0)
            extent = [-180, 180, -90, -60]
            # 南极投影参数
            proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=0)

        colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
        cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)

        plt.figure(figsize=(10, 10))
        ax = plt.axes(projection=proj_crs)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        grid_data = grid_data.astype(float)
        grid_data = np.ma.masked_invalid(grid_data)  # 掩膜无效数据

        # 转换 grid_x 和 grid_y 为 longitude 和 latitude 使用 pyproj
        lon, lat = proj(grid_x, grid_y, inverse=True)

        mappable = ax.pcolormesh(
            lon, lat, grid_data,
            transform=ccrs.PlateCarree(), cmap=cmap, shading='auto'
        )

        # 添加 land, coastlines, gridlines
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                          linewidth=1, color='gray', alpha=0.5, linestyle='--')
        gl.right_labels = False
        gl.top_labels = False

        cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('海冰密集度 (%)')

        title_date = os.path.basename(output_file).split('_')[-1].replace('.nc', '')
        plt.title(f"{datesetname} 海冰密集度 ({title_date})", fontsize=15)

        # 保存快视图
        snapshot_path = os.path.join(snapshot_folder, os.path.basename(output_file).replace('.nc', '.png'))
        plt.savefig(snapshot_path, dpi=100, bbox_inches='tight')
        plt.close()
        # print(f"Snapshot saved: {snapshot_path}")

    except Exception as e:
        print(f"Error generating snapshot: {e}")