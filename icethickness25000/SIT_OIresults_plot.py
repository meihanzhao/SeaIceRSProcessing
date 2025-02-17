# -*- coding: utf-8 -*-
"""
北极海冰厚度融合产品绘图
说明：
    该脚本自动扫描文件夹 "D:\zmh\icethickness\SIT_OI_merge_R25km" 下的所有 NetCDF 文件，
    并利用最优插值融合后的数据文件绘制北极海冰厚度图。绘图使用 Cartopy 投影及 Matplotlib，
    并将生成的图像保存至指定的输出目录中。
"""
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from pyproj import Proj
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LATITUDE_FORMATTER

plt.rcParams['font.sans-serif'] = ['SimHei']

def create_north_polar_plot(data_file, output_dir):
    """
    绘制基于最优插值融合完成的海冰厚度（SIT）北极产品。
    输入：
        data_file - 海冰厚度的NetCDF文件，生成文件路径为 "D:\\zmh\\icethickness\\SIT_OI_merge_R25km\\*.nc"
        output_dir - 输出图像保存目录
    """
    # 自定义色带，建议根据海冰厚度范围调整色标
    # 创建海冰厚度的颜色映射
    # colors = ["#C7E9F1", "#D4E6F1", "#85C1E9", "#3498DB", "#2874A6", "#1B4F72"]
    # cmap = LinearSegmentedColormap.from_list("ice_thickness", colors, N=10)
    colors = ["#C7E9F1", "#A2D2F2", "#75B8EA", "#538DD5", "#2A65B3", "#0E3D99"]
    cmap = LinearSegmentedColormap.from_list("ice_thickness", colors, N=8)

    
    # 北极投影设置
    proj_crs = ccrs.NorthPolarStereo(central_longitude=-45)
    proj = Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)
    
    # 打开NetCDF文件并提取数据
    ds = xr.open_dataset(data_file)
    # 提取日期信息，文件名格式为 "Arctic_SIT_OI_R25km_YYYYMMDD.nc"
    date_str = os.path.basename(data_file).split('_')[4].split('.')[0]
    
    # 海冰厚度变量名称为 "SIT"
    sit_data = ds['SIT'].astype(float)
    sit_data = np.ma.masked_invalid(sit_data)
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj_crs})
    extent = [-180, 180, 58.5, 90]
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # 获取经纬度坐标
    lon_grid, lat_grid = np.meshgrid(ds.x.values, ds.y.values)
    lon, lat = proj(lon_grid, lat_grid, inverse=True)
    
    # 绘制海冰厚度
    # 此处vmin=0, vmax=5，根据实际数据范围调整（单位：米）
    mappable = ax.pcolormesh(lon, lat, sit_data, transform=ccrs.PlateCarree(),
                             cmap=cmap, shading='auto', vmin=0, vmax=4)
    
    ax.set_title(f'海冰厚度融合产品 ({date_str})', fontproperties='SimHei')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.coastlines(resolution='50m', color='black', linewidth=1)
    
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    
    cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8, ax=ax)
    cbar.set_label('海冰厚度 (m)')
    
    # 保存图像
    output_filename = os.path.join(output_dir, f"海冰厚度融合产品_25000_{date_str}.png")
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    ds.close()

def main():
    # 指定海冰厚度融合结果文件所在目录
    data_dir = r"D:\zmh\icethickness\SIT_OI_merge_R25km"
    # 指定图像输出目录
    output_dir = os.path.join(data_dir, "snapshot")
    os.makedirs(output_dir, exist_ok=True)
    
    # 处理目录下所有.nc文件
    for filename in sorted(os.listdir(data_dir)):
        if filename.endswith(".nc") and "SIT_OI_R25km" in filename:
            data_file = os.path.join(data_dir, filename)
            print(f"生成图像: {data_file}")
            create_north_polar_plot(data_file, output_dir)
    print("北极海冰厚度产品绘图完成。")

if __name__ == "__main__":
    main()
