# -*- coding: utf-8 -*-
import xarray as xr
import numpy as np
from pathlib import Path
from scipy.interpolate import griddata
import os
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from scipy.spatial import ConvexHull, Delaunay
from RegularGrid import north_regular_grid, south_regular_grid,generate_snapshot

# 文件路径
base_folder_path  = r"D:\zmh\icecon12500_data\OISST"   # OISST数据的根目录
output_path_north = os.path.join(base_folder_path, "north")  # 北极daily NC文件的输出目录
output_path_south = os.path.join(base_folder_path, "south")  # 南极daily NC文件的输出目录
final_output_dir_north = os.path.join(base_folder_path, "0_pre_com", "north")
final_output_dir_south = os.path.join(base_folder_path, "0_pre_com", "south")
failure_folder = os.path.join(base_folder_path, "0_pre_com", "failures")
snapshot_folder = os.path.join(base_folder_path, "0_pre_com", "snapshots")

# 检查输出文件夹是否存在
os.makedirs(output_path_north, exist_ok=True)
os.makedirs(output_path_south, exist_ok=True)
os.makedirs(final_output_dir_north, exist_ok=True)
os.makedirs(final_output_dir_south, exist_ok=True)
os.makedirs(failure_folder, exist_ok=True)
os.makedirs(snapshot_folder, exist_ok=True)

# 数据时间范围
start_year, end_year = 2003, 2024

# --- 步骤1：定义南北极CRS和网格 ---
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)    # 北极投影坐标 (x, y)
crs_pss = CRS.from_epsg(3976)    # 南极投影坐标 (x, y)
grid_x_n, grid_y_n = north_regular_grid()  # 北极规则网格
grid_x_s, grid_y_s = south_regular_grid()  # 南极规则网格

# --- 步骤1：数据切分 ---
def process_oisst_data(input_path, output_path_north, output_path_south, start_year, end_year):
    """
    数据切分为南极和北极
    """

    for year in range(start_year, end_year + 1):
        for month in range(1, 13):  # 遍历每个月的数据
            input_dir = os.path.join(input_path, str(year), f"{month:02}") # 输入目录
            
            if not os.path.exists(input_dir):
                continue
            
            # 遍历每个文件
            for filename in os.listdir(input_dir):
                if filename.endswith(".nc"):  # 仅处理.nc文件
                    file_path = os.path.join(input_dir, filename)
                    
                    try:
                        ds = xr.open_dataset(file_path, engine='netcdf4')
                        ds = ds.drop_vars(['sst', 'anom', 'err', 'zlev'], errors='ignore')
                        ds = ds.drop(['time'], errors='ignore')
                        north_ds = ds.sel(lat=ds.lat >= 60)
                        south_ds = ds.sel(lat=ds.lat <= -60)
            
                        if 'ice' in north_ds: # 检查是否存在ice变量
                            north_ds['ice'] = north_ds['ice'].squeeze()
                        if 'ice' in south_ds:
                            south_ds['ice'] = south_ds['ice'].squeeze()
                        
                        # 从文件名中提取日期（例如，20220101）
                        date = filename.split(".")[1]  
                        north_filename = f"OISST_north_{date}.nc"
                        south_filename = f"OISST_south_{date}.nc"
                        
                        # 保存数据
                        if north_ds and len(north_ds.dims) > 0:
                            north_output_path = os.path.join(output_path_north, north_filename)
                            north_ds.to_netcdf(north_output_path)
                            # print(f"Saved: {north_output_path}")
                        if south_ds and len(south_ds.dims) > 0:
                            south_output_path = os.path.join(output_path_south, south_filename)
                            south_ds.to_netcdf(south_output_path)
                            # print(f"Saved: {south_output_path}")
                    
                    except Exception as e:
                        print(f"Failed to process file: {filename}, Error: {e}")

# --- 步骤2：定义凸包掩膜函数 --
def in_hull(points, hull):
    """
    检查点是否位于凸包内。
    输入：
    points - 待检查的坐标点
    hull - 凸包对象
    输出：
    布尔数组，表示哪些点在凸包内。
    """
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(points) >= 0 

# --- 步骤3：定义查找.nc文件的函数 ---
def find_nc_files(directory):
    """
    递归搜索指定目录下的所有.nc文件。
    输入：
    directory - 要搜索的目录
    输出：
    包含.nc文件路径的列表
    """
    nc_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".nc"):
                nc_files.append(os.path.join(root, file))
    return nc_files

# --- 步骤4：处理每日NetCDF文件 ---
def process_daily_nc(input_dir: str, hemisphere: str, crs_ps: CRS, grid_x_hem: np.ndarray, grid_y_hem: np.ndarray, output_dir: str):
    """
    处理每日NetCDF文件，包括数据筛选、投影转换、插值和输出。
    输入：
    input_dir - 输入目录，包含每日NetCDF文件
    hemisphere - 半球名称（north 或 south）
    crs_ps - 极地立体投影坐标系
    grid_x_hem - x方向的规则网格
    grid_y_hem - y方向的规则网格
    output_dir - 输出目录，用于存储处理后的NetCDF文件
    """
    all_files = find_nc_files(input_dir)  

    if not all_files:
        print(f"No .nc files found in {input_dir} for {hemisphere} hemisphere.")
        return

    for file in tqdm(all_files, desc=f"Processing {hemisphere} hemisphere", unit="file"):
        try:
            ds_sic = xr.open_dataset(file)

            # 按照纬度筛选数据
            if hemisphere == 'north':
                mask = ds_sic["lat"] >= 60
            else:  
                mask = ds_sic["lat"] <= -60
            filtered_ds = ds_sic.where(mask, drop=False)

            # 筛选海冰浓度在 0-1 之间的数据，并转换为百分比
            filtered_ds['ice'] = filtered_ds['ice'].where(
                (filtered_ds['ice'] >= 0) & (filtered_ds['ice'] <= 1), drop=False
            )
            filtered_ds['ice'] *= 100  

            # 投影转换
            transformer = Transformer.from_crs(crs_wgs84, crs_ps, always_xy=True)
            lon = filtered_ds['lon'].values
            lat = filtered_ds['lat'].values
            lon_2d, lat_2d = np.meshgrid(lon, lat)
            x, y = transformer.transform(lon_2d, lat_2d)

            # 添加新的坐标
            filtered_ds = filtered_ds.assign_coords({
                "x": (("lat", "lon"), x),
                "y": (("lat", "lon"), y),
            })

            # 凸包掩膜
            points = np.column_stack((x.flatten(), y.flatten()))
            hull = ConvexHull(points)
            grid_points = np.column_stack((grid_x_hem.ravel(), grid_y_hem.ravel()))
            mask = in_hull(grid_points, hull)

            # 插值到规则网格
            z_flat = filtered_ds['ice'].values.flatten()
            interpolated_data = griddata(
                (x.flatten(), y.flatten()), z_flat, (grid_x_hem, grid_y_hem), method="nearest"
            )

            masked_data = np.full(interpolated_data.shape, np.nan)
            masked_data[mask.reshape(grid_x_hem.shape)] = interpolated_data[mask.reshape(grid_x_hem.shape)]

            # 创建最终的 xarray 数据集
            da = xr.DataArray(masked_data, coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])], dims=['y', 'x'])
            final_ds = da.to_dataset(name=f'OISST_{hemisphere}_icecon')

            # 保存为 NetCDF
            output_file = os.path.join(output_dir, f"OSTIA_{hemisphere}_icecon_12500_{file[-11:-3]}.nc")
            final_ds.to_netcdf(output_file)
            generate_snapshot(masked_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'OISST', snapshot_folder)
            # print(f"Successfully saved final NetCDF: {output_file}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            shutil.copy(file, failure_folder) 

if __name__ == "__main__":
    # --- 步骤5：执行处理 ---
    process_oisst_data(base_folder_path, output_path_north, output_path_south, start_year, end_year) # 数据切分
    process_daily_nc(output_path_north, 'north', crs_psn, grid_x_n, grid_y_n, final_output_dir_north) # 处理北极每日数据
    process_daily_nc(output_path_south, 'south', crs_pss, grid_x_s, grid_y_s, final_output_dir_south) # 处理南极每日数据
