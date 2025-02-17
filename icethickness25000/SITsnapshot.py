# -*- coding: utf-8 -*-
"""海冰厚度快视图生成模块"""
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import pyproj
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False   #  添加这行代码，使用 ASCII 负号

def generate_SIT_snapshot(
        data: xr.Dataset, grid_x: np.ndarray, grid_y: np.ndarray,
        output_filepath: str, dataset_name: str, snapshot_folder: str,
        show_figure: bool = False,
) -> None:
    """生成海冰厚度快视图。"""

    try:
        # 定义北极投影和范围
        proj_crs = ccrs.NorthPolarStereo(central_longitude=-45)
        extent = [-180, 180, 60, 90]

        # 北极投影参数
        proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)

        # 创建海冰厚度的颜色映射
        colors = ["#FFFFFF", "#D4E6F1", "#85C1E9", "#3498DB", "#2874A6", "#1B4F72"]
        cmap = LinearSegmentedColormap.from_list("ice_thickness", colors, N=6)

        # 创建图形和坐标轴
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj_crs})
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 提取 'asit' 数据为 NumPy 数组
        sit_data = data[dataset_name].values 

        # 屏蔽无效值
        sit_data = sit_data.astype(float)
        masked_data = np.ma.masked_invalid(sit_data)

        # 将投影坐标转换为经度和纬度
        lon, lat = proj(grid_x, grid_y, inverse=True)

        # 绘制海冰厚度数据
        mappable = ax.pcolormesh(
            lon,
            lat,
            masked_data,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            shading='auto',
        )

        # 添加陆地、海岸线和网格线
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.coastlines(resolution='50m', color='black', linewidth=1) 
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=1,
            color='gray',
            alpha=0.5,
            linestyle='--',
        )
        gl.right_labels = False
        gl.top_labels = False

        # 添加colorbar
        cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8)
        cbar.set_label('海冰厚度 (m)', fontsize=12)

        # 设置图形标题
        title_date = os.path.basename(output_filepath).split('_')[-1].replace('.nc', '') 
        ax.set_title(f'{dataset_name}海冰厚度{title_date}', fontsize=15)

        # 保存快视图
        snapshot_path = os.path.join(
            snapshot_folder, os.path.basename(output_filepath).replace('.nc', '.png')
        )
        plt.savefig(snapshot_path, dpi=100, bbox_inches='tight')
        if show_figure:
            plt.show()
        plt.close(fig)

    except Exception as e:
        print(f"生成快视图时出错: {e}")