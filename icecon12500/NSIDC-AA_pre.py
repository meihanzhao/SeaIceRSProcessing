# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import h5py
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from RegularGrid import north_regular_grid, south_regular_grid,generate_snapshot

# 文件路径
base_folder_path = r"D:\zmh\icecon12500_data\NSIDC-AA"

# 定义南北极的CRS
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)    # 北极（x,y）
crs_pss = CRS.from_epsg(3976)    # 南极（x,y）

# 快视图
snapshot_folder = os.path.join(base_folder_path, "0_pre_com", "snapshots")
os.makedirs(snapshot_folder, exist_ok=True)

def process_hemisphere(hemisphere: str):
    """
    处理指定半球的海冰浓度数据。
    此函数将HDF文件中的海冰浓度数据读取、筛选、投影转换、
    插值到规则网格并保存为NetCDF文件，同时生成图像快视图。
    """
    if hemisphere == 'north':
        crs_ps = crs_psn  # 极地投影坐标系
        grid_x_hem, grid_y_hem = north_regular_grid()  # 北半球规则网格
        data_field = 'SI_12km_NH_ICECON_DAY'  # 数据名称（HDF5 file）
        lat_condition = lambda lat: lat >= 60  # 纬度筛选条件
        output_folder = os.path.join(base_folder_path, "0_pre_com", "north")
        failure_folder = os.path.join(base_folder_path, "0_pre_com", "failures")
        epsg_crs = crs_psn
        ice_concentration_path = 'HDFEOS/GRIDS/NpPolarGrid12km/Data Fields/SI_12km_NH_ICECON_DAY'
        lat_path = 'HDFEOS/GRIDS/NpPolarGrid12km/lat'
        lon_path = 'HDFEOS/GRIDS/NpPolarGrid12km/lon'
    elif hemisphere == 'south':
        crs_ps = crs_pss
        grid_x_hem, grid_y_hem = south_regular_grid() # 南半球规则网格
        data_field = 'SI_12km_SH_ICECON_DAY'
        lat_condition = lambda lat: lat <= -60
        output_folder = os.path.join(base_folder_path, "0_pre_com", "south")
        failure_folder = os.path.join(base_folder_path, "0_pre_com", "failures")
        epsg_crs = crs_pss
        ice_concentration_path = 'HDFEOS/GRIDS/SpPolarGrid12km/Data Fields/SI_12km_SH_ICECON_DAY'
        lat_path = 'HDFEOS/GRIDS/SpPolarGrid12km/lat'
        lon_path = 'HDFEOS/GRIDS/SpPolarGrid12km/lon'
    else:
        raise ValueError("Invalid hemisphere specified. Choose 'north' or 'south'.")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failure_folder, exist_ok=True)

    total_files = 0
    successful_files = 0
    failed_files = 0

    # 遍历所有文件，从2004年到2023年
    all_files = []
    for year in range(2004, 2024):
        year_folder_path = os.path.join(base_folder_path, str(year))
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue

        for filename in os.listdir(year_folder_path):
            if filename.endswith('.he5'):  #  选取 HDF-EOS5 文件
                file_path = os.path.join(year_folder_path, filename)
                all_files.append(file_path)

    # 遍历所有文件
    with tqdm(total=len(all_files), desc=f"Processing {hemisphere} hemisphere", unit="file") as pbar:
        for file_path in all_files:
            total_files += 1

            # 构建输出文件路径与名字
            output_file = os.path.join(
                output_folder,
                f'NSIDC-AA_{hemisphere}_icecon_{12500}_{os.path.basename(file_path)[-12:].replace(".he5", ".nc")}'
            )

            # 检查输出文件是否已存在，若存在则跳过
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping.")
                successful_files += 1
                pbar.update(1)
                continue

            # --- 步骤1：从 HDF5 文件中读取数据 ---
            try:
                with h5py.File(file_path, 'r') as f:
                    ice_concentration = f[ice_concentration_path][:]
                    lat = f[lat_path][:]
                    lon = f[lon_path][:]
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1
                pbar.update(1)
                continue

            # --- 步骤2：创建 xarray 数据集并筛选数据 ---
            try:
                # 创建 xarray 数据集
                ds = xr.Dataset(
                    {
                        'ice_concentration': (['y', 'x'], ice_concentration),
                    },
                    coords={
                        'y': np.arange(ice_concentration.shape[0]),  
                        'x': np.arange(ice_concentration.shape[1]),
                        'lat': (['y', 'x'], lat),
                        'lon': (['y', 'x'], lon),
                    }
                )

                # 筛选纬度在指定范围内的数据
                mask = lat_condition(ds["lat"].values)
                filtered_ds = ds.where(mask, drop=False)

                # 筛选海冰浓度在 0-100% 之间的数据
                mask = (filtered_ds['ice_concentration'] >= 0) & (filtered_ds['ice_concentration'] <= 100)
                filtered_ds['ice_concentration'] = filtered_ds['ice_concentration'].where(mask, drop=False)
            except Exception as e:
                print(f"Error creating dataset for {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1
                pbar.update(1)
                continue

            # --- 步骤3：投影转换 ---
            try:
                # Create a transformer for coordinate transformation
                transformer = Transformer.from_crs(crs_wgs84, epsg_crs, always_xy=True)
                lon_vals = filtered_ds['lon'].values
                lat_vals = filtered_ds['lat'].values

                # Transform coordinates from WGS84 to Polar Stereographic
                x, y = transformer.transform(lon_vals, lat_vals)
            except Exception as e:
                print(f"Error with projection transformation for {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1
                pbar.update(1)
                continue

            # --- 步骤4：插值（重采样） ---
            try:
                x_flat = x.flatten()
                y_flat = y.flatten()
                z_flat = filtered_ds['ice_concentration'].values.flatten()

                # 采用最近邻插值法将数据插值到定义的12.5 km网格上
                grid_data = griddata((x_flat, y_flat), z_flat, (grid_x_hem, grid_y_hem), method="nearest")
            except Exception as e:
                print(f"Error during interpolation for {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1
                pbar.update(1)
                continue

            # --- 步骤5：保存为 NetCDF 文件 ---
            try:
                # 为插值数据创建 xarray DataArray
                da = xr.DataArray(
                    grid_data,
                    coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])],
                    dims=['y', 'x']
                )

                # 转换 DataArray 为 Dataset
                da_dataset = da.to_dataset(name=f'NSIDC-AA_{hemisphere}_icecon')

                # 构建输出文件路径与名字
                output_file = os.path.join(
                    output_folder,
                    f'NSIDC-AA_{hemisphere}_icecon_{12500}_{os.path.basename(file_path)[-12:].replace(".he5", ".nc")}'
                )

                # 保存 dataset 为 NetCDF 文件
                da_dataset.to_netcdf(output_file)
                generate_snapshot(grid_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'NSIDC-AA', snapshot_folder)
                # print(f"Successfully saved NetCDF: {output_file}")
                successful_files += 1
            except Exception as e:
                print(f"Error saving NetCDF for file {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1
            
            pbar.update(1)   

    print(f"Total files processed: {total_files}")
    print(f"Successfully processed files: {successful_files}")
    print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    # --- 步骤3：运行北极和南极的整个处理流程 ---
    process_hemisphere('north')
    process_hemisphere('south')