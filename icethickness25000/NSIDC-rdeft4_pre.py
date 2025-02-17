# -- coding: utf-8 --
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
rdeft4_base_folder_path = paths['rdeft4_base_folder_path']
output_folder_north = os.path.join(rdeft4_base_folder_path, '0_pre_com', 'north')
failure_folder = os.path.join(rdeft4_base_folder_path, '0_pre_com', 'failures')
snapshot_folder = os.path.join(rdeft4_base_folder_path, "0_pre_com", "snapshots")
for folder in [output_folder_north, failure_folder, snapshot_folder]:
    os.makedirs(folder, exist_ok=True)

# 定义北半球CRS
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)  # 北极投影 (x, y)
grid_x_n, grid_y_n = north_regular_grid()  # 北极规则网格

def process_north():

    """
    处理北半球的海冰厚度数据，使用手动 NumPy进行裁剪。

    """

    total_files = 0
    successful_files = 0
    failed_files = 0
    all_files = []

    for year in range(2004, 2024):
        year_folder_path = os.path.join(rdeft4_base_folder_path, str(year))
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue

        for filename in os.listdir(year_folder_path):
            if filename.endswith('.nc'):
                file_path = os.path.join(year_folder_path, filename)
                all_files.append(file_path)

    with tqdm(total=len(all_files), desc="Processing files") as pbar:
        for file_path in all_files:  # 仍然只处理前两个文件用于测试

            try:
                sit_data = xr.open_dataset(file_path)
                sit_data = sit_data[['sea_ice_thickness','lon', 'lat']]
                # 1. 提取 NumPy arrays
                lat_values = sit_data['lat'].values
                lon_values = sit_data['lon'].values
                sit_values = sit_data['sea_ice_thickness'].values
                # print(f"原始数据 shapes - lat: {lat_values.shape}, lon: {lon_values.shape}, sit: {sit_values.shape}")

                # 将海冰厚度裁剪到最小值 0
                sit_values = np.clip(sit_values, a_min=0, a_max=None) # 将 sit_values 中小于 0 的值设置为 0

                # 2. 创建纬度掩码，使用 NumPy
                mask_lat_np = lat_values >= 60

                # 3. 应用掩码
                filtered_lat = lat_values[mask_lat_np]
                filtered_lon = lon_values[mask_lat_np]
                filtered_sit = sit_values[mask_lat_np]

                # 4. 在过滤后显式删除 NaN值
                not_nan_mask_filtered = np.isfinite(filtered_lon) & np.isfinite(filtered_lat) & np.isfinite(filtered_sit)
                filtered_lon_no_nan = filtered_lon[not_nan_mask_filtered]
                filtered_lat_no_nan = filtered_lat[not_nan_mask_filtered]
                filtered_sit_no_nan = filtered_sit[not_nan_mask_filtered]

                if filtered_sit_no_nan.size == 0:
                    failed_files += 1
                    failure_filename = os.path.join(failure_folder, os.path.basename(file_path))
                    shutil.copy(file_path, failure_filename)

                else:
                    # 5. 如果有有效数据，则继续进行投影和插值
                    transformer = Transformer.from_crs(crs_wgs84, crs_psn, always_xy=True)
                    x, y = transformer.transform(filtered_lon_no_nan, filtered_lat_no_nan)

                    sit_data_interpolated = griddata(
                        (x.flatten(), y.flatten()),
                        filtered_sit_no_nan.flatten(),
                        (grid_x_n, grid_y_n),
                    )

                    da = xr.DataArray(
                        sit_data_interpolated,
                        coords={'y': grid_y_n[:, 0], 'x': grid_x_n[0, :]},
                        dims=['y', 'x']
                    )

                    da_dataset = da.to_dataset(name='rdeft4_sit')
                    output_filename = os.path.join(
                        output_folder_north,
                        f'rdeft4_north_SIT_25000_{os.path.basename(file_path)[-11:-3]}.nc'
                    )

                    generate_SIT_snapshot(da_dataset, grid_x_n, grid_y_n, output_filename, 'rdeft4_sit', snapshot_folder)
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

    print(f"Total files processed: {total_files}")
    print(f"Successful files: {successful_files}")
    print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    process_north()
    print("Success!")