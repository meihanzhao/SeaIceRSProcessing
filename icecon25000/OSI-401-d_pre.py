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
base_folder_path = r"D:\zmh\icecon25000_data\OSI-401-d"
snapshot_folder = os.path.join(base_folder_path, "0_pre_com_25", "snapshots")
os.makedirs(snapshot_folder, exist_ok=True)

def process_hemisphere(hemisphere: str, start_year: int, end_year: int):
    """
    处理指定半球的海冰浓度数据。
    此函数将HDF文件中的海冰浓度数据读取、筛选、投影转换、
    插值到规则网格并保存为NetCDF文件，同时生成图像快视图。
    """
    # Determine output folders
    if hemisphere == 'north':
        output_folder = os.path.join(base_folder_path, "0_pre_com_25", "north")
        failure_folder = os.path.join(base_folder_path, "0_pre_com_25", "failures","north")
        mask_condition = lambda lat: lat >= 60
        grid_x_hem, grid_y_hem = north_regular_grid()
        region_identifier = "_nh_"
        ice_concentration_field = 'OSI-401-d'
    elif hemisphere == 'south':
        output_folder = os.path.join(base_folder_path, "0_pre_com_25", "south")
        
        failure_folder = os.path.join(base_folder_path, "0_pre_com_25", "failures","south")
        mask_condition = lambda lat: lat <= -60
        grid_x_hem, grid_y_hem = south_regular_grid()
        region_identifier = "_sh_"
        ice_concentration_field = 'OSI-401-d'
    else:
        raise ValueError("Invalid hemisphere specified. Choose 'north' or 'south'.")

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failure_folder, exist_ok=True)

    total_files = 0
    successful_files = 0
    failed_files = 0

    # 遍历每个月的数据
    for year in range(start_year, end_year + 1):
        for month in range(1, 13):
            year_month_folder = os.path.join(base_folder_path, str(year), f"{month:02d}")
            if not os.path.exists(year_month_folder):
                print(f"Month folder {year_month_folder} does not exist, skipping.")
                continue

            # 查找名称中带有polstere的nc文件
            all_files = [
                os.path.join(year_month_folder, f)
                for f in os.listdir(year_month_folder)
                if f.endswith('.nc') and "polstere" in f.lower() and region_identifier in f
            ]

            # 遍历所有文件
            with tqdm(total=len(all_files), desc=f"Processing {hemisphere} hemisphere {year}-{month:02d}", unit="file") as pbar:
                for sic_file in all_files:
                    total_files += 1

                    # 定义输出文件夹及文件名
                    output_filename = os.path.join(
                        output_folder,
                        f'OSI-401-d_{hemisphere}_icecon_25000_{os.path.basename(sic_file)[-15:-7]}.nc'
                    )

                    # 跳过已存在的文件
                    if os.path.exists(output_filename):
                        print(f"Output file {output_filename} already exists, skipping.")
                        successful_files += 1  
                        pbar.update(1)  
                        continue

                    try:
                        # --- 步骤1：读取数据 ---
                        ds = xr.open_dataset(sic_file, engine='netcdf4')
                        
                        # 删掉不需要的变量
                        ds = ds.drop_vars([
                            'time_bnds', 'Polar_Stereographic_Grid', 'confidence_level', 'status_flag', 
                            'total_uncertainty', 'smearing_uncertainty', 'algorithm_uncertainty', 
                            'ice_conc_unfiltered', 'masks'
                        ], errors='ignore')
                        
                        ds_sic = ds.drop('time', errors='ignore')
                        ds_sic['ice_conc'] = ds_sic['ice_conc'].squeeze()
                        ds_sic = ds_sic.rename({'yc': 'y', 'xc': 'x'})
                        ds_sic['x'] = ds_sic['x'] * 1000
                        ds_sic['y'] = ds_sic['y'] * 1000

                        mask = mask_condition(ds_sic['lat'])
                        filtered_ds = ds_sic.where(mask, drop=False)

                        # --- 步骤2：插值（重采样） ---
                        x = filtered_ds['x'].values
                        y = filtered_ds['y'].values
                        x_2d, y_2d = np.meshgrid(x, y)
                        filtered_ds = filtered_ds.reset_index(["x", "y"]) 

                        filtered_ds = filtered_ds.assign_coords({ "x_2d": (("y", "x"), x_2d), "y_2d": (("y", "x"), y_2d)})
                        filtered_ds = filtered_ds.drop_vars(["x", "y"]) 
                        filtered_ds = filtered_ds.rename({"x_2d": "x", "y_2d": "y"})

                        # 提取 2D x, y, z 数据
                        x_2d = filtered_ds['x'].values  # 2D x array
                        y_2d = filtered_ds['y'].values  # 2D y array
                        z_2d = filtered_ds['ice_conc'].values  

                        # 展开 2D x, y, z 数据
                        x_flat = x_2d.flatten()
                        y_flat = y_2d.flatten()
                        z_flat = z_2d.flatten()

                        # 插值到规则网格
                        grid_data = griddata((x_flat, y_flat), z_flat, (grid_x_hem, grid_y_hem), method="nearest")

                        # --- 步骤3：保存为 NetCDF 文件 ---
                        da = xr.DataArray(
                            grid_data,
                            coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])],
                            dims=['y', 'x']
                        )

                        # 保存为 NetCDF
                        output_filename = os.path.join(
                            output_folder, 
                            f'OSI-401-d_{hemisphere}_icecon_25000_{os.path.basename(sic_file)[-15:-7]}.nc'
                        )
                        da.to_dataset(name=f'{ice_concentration_field}_{hemisphere}_icecon').to_netcdf(output_filename)
                        generate_snapshot(grid_data, grid_x_hem, grid_y_hem, hemisphere, output_filename, 'OSI-401-d', snapshot_folder)

                        # print(f"Successfully saved NetCDF: {output_filename}")
                        successful_files += 1

                    except Exception as e:
                        print(f"Error processing file {sic_file}: {e}")
                        shutil.copy(sic_file, failure_folder)
                        failed_files += 1

                    pbar.update(1)

    print(f"Total files processed: {total_files}")
    print(f"Successfully processed files: {successful_files}")
    print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    # 执行处理北极和南极的函数（例如，2004-2023年）
    process_hemisphere('north', 2004, 2024)
    process_hemisphere('south', 2004, 2024)