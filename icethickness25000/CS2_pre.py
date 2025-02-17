# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from RegularGrid import north_regular_grid
from SITsnapshot import generate_SIT_snapshot
from paths import paths
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# 文件路径
CS2_base_folder_path = paths['CS2_base_folder_path']
output_folder_north = os.path.join(CS2_base_folder_path, '0_pre_com', 'north')
failure_folder = os.path.join(CS2_base_folder_path, '0_pre_com', 'failures')
snapshot_folder = os.path.join(CS2_base_folder_path, "0_pre_com", "snapshots")
for folder in [output_folder_north, failure_folder, snapshot_folder]:
    os.makedirs(folder, exist_ok=True)

# 定义北半球CRS
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)  # 北极投影 (x, y)

grid_x_n, grid_y_n = north_regular_grid()  # 北极规则网格


def process_north():
    """
    处理北半球的海冰厚度数据。
    """

    total_files = 0  # 总文件数
    successful_files = 0  # 成功处理的文件数
    failed_files = 0  # 处理失败的文件数

    # 遍历每个年份的数据
    all_files = []  # 存储所有文件的列表
    for year in range(2004, 2024):  # 遍历年份
        year_folder_path = os.path.join(CS2_base_folder_path, str(year))  # 年份文件夹路径
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue

        for filename in os.listdir(year_folder_path):  # 遍历文件夹中的文件
            if filename.endswith('.nc'):  # 仅处理.nc文件
                file_path = os.path.join(year_folder_path, filename)  # 文件路径
                all_files.append(file_path)  # 添加文件路径到列表

    # 遍历所有文件
    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        for file_path in all_files: 
            try:
                # 读取NetCDF文件
                sit_data = xr.open_dataset(file_path)
                sit_data = sit_data[['analysis_sea_ice_thickness', 'lon', 'lat']].drop('time')  # 仅保留海冰厚度、经度和纬度数据
                sit_data['analysis_sea_ice_thickness'] = sit_data['analysis_sea_ice_thickness'].squeeze()  # 海冰厚度数据

                # 投影转换
                mask = sit_data['lat'] >= 60
                sit_data = sit_data.where(mask, drop=True)
                transformer = Transformer.from_crs(crs_wgs84, crs_psn, always_xy=True)
                x, y = transformer.transform(sit_data['lon'].values, sit_data['lat'].values)

                # 插值到规则网格
                sit_data_interpolated = griddata(
                    (x.flatten(), y.flatten()),
                    sit_data['analysis_sea_ice_thickness'].values.flatten(),
                    (grid_x_n, grid_y_n)
                )

                # 保存插值数据
                da = xr.DataArray(
                    sit_data_interpolated,
                    coords={'y': grid_y_n[:, 0], 'x': grid_x_n[0, :]},  # 为数据添加坐标
                    dims=['y', 'x']
                )

                da_dataset = da.to_dataset(name='CS2_sit')
                output_filename = os.path.join(
                    output_folder_north,  
                    f'CS2_north_SIT_25000_{os.path.basename(file_path)[-36:-28]}.nc'  # 生成文件名
                )
                # 生成快视图
                generate_SIT_snapshot(da_dataset, grid_x_n, grid_y_n, output_filename, 'CS2_sit', snapshot_folder)

                # 保存为NetCDF文件
                da_dataset.to_netcdf(output_filename)
                successful_files += 1
            except Exception as e:
                failed_files += 1
                failure_filename = os.path.join(failure_folder, os.path.basename(file_path))  

                shutil.copy(file_path, failure_filename)  
                print(f"Error processing file {file_path}: {e}")
            finally:
                total_files += 1
                pbar.update(1)

    # 输出处理结果
    print(f"Total files processed: {total_files}")
    print(f"Successful files: {successful_files}")
    print(f"Failed files: {failed_files}")


if __name__ == '__main__':
    process_north()
    print("Success!")