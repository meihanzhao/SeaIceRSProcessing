# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from scipy.spatial import ConvexHull, Delaunay
from RegularGrid import north_regular_grid, south_regular_grid,generate_snapshot

# 文件路径 
base_folder_path = r"D:\zmh\icecon12500_data\UB-AMSR2"
output_folder_north = os.path.join(base_folder_path, '0_pre_com', 'north') 
output_folder_south = os.path.join(base_folder_path, '0_pre_com', 'south')  
failure_folder = os.path.join(base_folder_path, '0_pre_com', 'failures')
snapshot_folder = os.path.join(base_folder_path, "0_pre_com", "snapshots")  
os.makedirs(output_folder_north, exist_ok=True)  
os.makedirs(output_folder_south, exist_ok=True)  
os.makedirs(failure_folder, exist_ok=True)  
os.makedirs(snapshot_folder, exist_ok=True)

# 定义南北半球 CRS 
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)    # 北极投影 (x, y)
crs_pss = CRS.from_epsg(3976)    # P南极投影 (x, y)

# 定义南北半球规则网格
grid_x_n, grid_y_n = north_regular_grid()  # 北极规则网格
grid_x_s, grid_y_s = south_regular_grid()  # 南极规则网格

def process_hemisphere(hemisphere: str):
    """
    处理指定半球的海冰浓度数据,定义参数。
    此函数将NetCDF文件中的海冰浓度数据读取、筛选、投影转换、
    插值到规则网格并保存为NetCDF文件。
    """
    if hemisphere == 'north':
        crs_ps = crs_psn  
        grid_x_hem = grid_x_n  
        grid_y_hem = grid_y_n  
        data_path = os.path.join(base_folder_path, 'north')  
        output_folder = output_folder_north  
        epsg_crs = crs_psn  
    elif hemisphere == 'south':
        crs_ps = crs_pss  
        grid_x_hem = grid_x_s  
        grid_y_hem = grid_y_s  
        data_path = os.path.join(base_folder_path, 'south') 
        output_folder = output_folder_south  
        epsg_crs = crs_pss  
    else:
        raise ValueError("Invalid hemisphere specified. Choose 'north' or 'south'.")
    
    total_files = 0  # 总文件数
    successful_files = 0  # 成功处理的文件数
    failed_files = 0  # 处理失败的文件数
    
    # 遍历每个年份的数据
    all_files = []  # 存储所有文件的列表
    for year in range(2004, 2024):  # 遍历年份
        year_folder_path = os.path.join(data_path, str(year))  # 年份文件夹路径
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue
        
        for filename in os.listdir(year_folder_path):  # 遍历文件夹中的文件
            if filename.endswith('.nc'):  # 仅处理.nc文件
                file_path = os.path.join(year_folder_path, filename)  # 文件路径
                all_files.append(file_path)  # 添加文件路径到列表
    
    # 遍历所有文件
    with tqdm(total=len(all_files), desc=f"Processing {hemisphere} hemisphere", unit="file") as pbar:
        for file_path in all_files:
            total_files += 1  # 增加总文件数计数器
            try:
                # --- 步骤1：打开并读取NetCDF文件 ---
                ds_sic = xr.open_dataset(file_path, engine='netcdf4')  
                x = ds_sic['x'].values  # 提取 x-coordinates
                y = ds_sic['y'].values  # 提取 y-coordinates
                x_grid, y_grid = np.meshgrid(x, y)  # 创建网格

                # --- 步骤2：坐标转换（投影） ---
                transformer = Transformer.from_crs(crs_ps, crs_wgs84, always_xy=True)  # 创建坐标转换器
                lon, lat = transformer.transform(x_grid, y_grid)  # 转换坐标

                ds_sic = ds_sic.assign_coords(
                    {"lon": (("y", "x"), lon),
                     "lat": (("y", "x"), lat)})

                # 筛选北极或南极数据
                mask = ds_sic['lat'] >= 60 if hemisphere == 'north' else ds_sic['lat'] <= -60  
                ds_sic = ds_sic.where(mask, drop=False)  

                # --- 步骤3：插值到目标网格 ---
                x_flat = x_grid.flatten()  
                y_flat = y_grid.flatten()  
                z_flat = ds_sic['z'].values.flatten()  
                
                grid_data = griddata((x_flat, y_flat), z_flat, (grid_x_hem, grid_y_hem), method="nearest")  # 插值到规则网格
                
                da = xr.DataArray(
                    grid_data, 
                    coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])],  
                    dims=['y', 'x']  
                )
                
                da_dataset = da.to_dataset(name=f'UB-AMSR2_{hemisphere}_icecon')  
                output_file = os.path.join(
                    output_folder,
                    f'UB-AMSR2_{hemisphere}_icecon_12500_{os.path.basename(file_path)[-16:-8]}.nc'  # 输出文件名
                )
                generate_snapshot(grid_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'UB-AMSR2', snapshot_folder)

                # --- 步骤4：保存为 NetCDF 文件 ---
                da_dataset.to_netcdf(output_file)  
                # print(f"Successfully saved NetCDF: {output_file}")
                successful_files += 1  # 增加成功处理的文件计数器
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")  
                shutil.copy(file_path, failure_folder)  
                failed_files += 1  # 增加处理失败的文件计数器
            
            pbar.update(1)  
    
    # 打印处理结果
    print(f"Total files processed for {hemisphere}: {total_files}")  
    print(f"Successfully processed files for {hemisphere}: {successful_files}")  
    print(f"Failed files for {hemisphere}: {failed_files}")  

if __name__ == "__main__":
    process_hemisphere('north')  # 处理北半球
    process_hemisphere('south')  # 处理南半球