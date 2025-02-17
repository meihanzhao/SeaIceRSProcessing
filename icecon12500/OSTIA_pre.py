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
base_folder_path = r"D:\zmh\icecon12500_data\OSTIA"
north_input_dir = os.path.join(base_folder_path, "north")
south_input_dir = os.path.join(base_folder_path, "south")
north_output_dir = os.path.join(base_folder_path, "north_daily")
south_output_dir = os.path.join(base_folder_path, "south_daily")
final_output_dir_north = os.path.join(base_folder_path, "0_pre_com", "north")
final_output_dir_south = os.path.join(base_folder_path, "0_pre_com", "south")
failure_folder = os.path.join(base_folder_path, "0_pre_com", "failures")
snapshot_folder = os.path.join(base_folder_path, "0_pre_com", "snapshots")

# 检查输出文件夹是否存在
os.makedirs(north_output_dir, exist_ok=True)
os.makedirs(south_output_dir, exist_ok=True)
os.makedirs(final_output_dir_north, exist_ok=True)
os.makedirs(final_output_dir_south, exist_ok=True)
os.makedirs(failure_folder, exist_ok=True)
os.makedirs(snapshot_folder, exist_ok=True)

# 定义南北半球 CRS 
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)    # 北极 (x, y)
crs_pss = CRS.from_epsg(3976)    # 南极 (x, y)

# 定义南北半球规则网格
grid_x_n, grid_y_n = north_regular_grid() # 北极规则网格
grid_x_s, grid_y_s = south_regular_grid() # 南极规则网格

def in_hull(points, hull):
    """
    凸包掩膜，用于检查点是否位于凸包内。
    输入：
    points - 待检查的坐标点
    hull - 凸包对象
    输出：
    布尔数组，表示哪些点在凸包内。
    """
    delaunay = Delaunay(hull.points[hull.vertices])
    return delaunay.find_simplex(points) >= 0  # 点在凸包内，返回True

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

# --- 步骤1：将大时间维度的.nc文件拆分为每日.nc文件 ---
def split_daily_nc(input_dir: str, output_dir: str, hemisphere: str):
    """
    将大时间维度的NetCDF文件拆分为按天存储的NetCDF文件。
    输入：
    input_dir - 输入目录，包含大时间维度的.nc文件
    output_dir - 输出目录，用于存储每日.nc文件
    hemisphere - 半球名称（north 或 south）
    """
    all_files = find_nc_files(input_dir)  
    
    # print(f"Files found for {hemisphere} hemisphere: {all_files}")

    if not all_files:
        print(f"No .nc files found in {input_dir} for {hemisphere} hemisphere.")
        return

    for file in tqdm(all_files, desc=f"Splitting {hemisphere} hemisphere", unit="file"):
        try:
            ds = xr.open_dataset(file)
            time_dim = ds.sizes['time']

            # 分割每日数据
            for i in range(time_dim):
                ds_sic = ds.isel(time=i)
                date = str(ds_sic.time.values)[:10].replace('-', '')  # 格式YYYYMMDD
                output_file = os.path.join(output_dir, f"OSTIA_{hemisphere}_{date}.nc")
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                ds_sic.to_netcdf(output_file)
                # print(f"Saved daily NetCDF: {output_file}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            shutil.copy(file, failure_folder)  # 失败文件移动到失败文件夹

# --- 步骤2：处理每日NetCDF文件 ---
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
    # print(f"Files found for processing in {hemisphere} hemisphere: {all_files}")

    if not all_files:
        print(f"No .nc files found in {input_dir} for {hemisphere} hemisphere.")
        return

    for file in tqdm(all_files, desc=f"Processing {hemisphere} hemisphere", unit="file"):
        try:
            # 读取daily数据
            ds = xr.open_dataset(file)
            ds_sic = ds.drop('time') 

            # 筛选北极或南极数据
            if hemisphere == 'north':
                mask = ds_sic["latitude"] >= 60
            else: 
                mask = ds_sic["latitude"] <= -60
            filtered_ds = ds_sic.where(mask, drop=False)

            # 筛选海冰浓度在 0-100% 之间的数据
            filtered_ds['sea_ice_fraction'] = filtered_ds['sea_ice_fraction'].where(
                (filtered_ds['sea_ice_fraction'] >= 0) & (filtered_ds['sea_ice_fraction'] <= 1), drop=False
            )
            filtered_ds['sea_ice_fraction'] *= 100  

            # 投影转换
            transformer = Transformer.from_crs(crs_wgs84, crs_ps, always_xy=True)
            lon = filtered_ds['longitude'].values
            lat = filtered_ds['latitude'].values
            longitude_2d, latitude_2d = np.meshgrid(lon, lat)
            x, y = transformer.transform(longitude_2d, latitude_2d)

            filtered_ds = filtered_ds.assign_coords({
                "x": (("latitude", "longitude"), x),
                "y": (("latitude", "longitude"), y),
            })

            # 凸包掩膜
            points = np.column_stack((x.flatten(), y.flatten()))
            hull = ConvexHull(points)
            grid_points = np.column_stack((grid_x_hem.ravel(), grid_y_hem.ravel()))
            mask = in_hull(grid_points, hull)

            # 插值到规则网格
            z_flat = filtered_ds['sea_ice_fraction'].values.flatten()
            interpolated_data = griddata(
                (x.flatten(), y.flatten()), z_flat, (grid_x_hem, grid_y_hem), method="nearest"
            )

            # 掩膜无效数据
            masked_data = np.full(interpolated_data.shape, np.nan)
            masked_data[mask.reshape(grid_x_hem.shape)] = interpolated_data[mask.reshape(grid_x_hem.shape)]

            # 创建 xarray 数据集
            da = xr.DataArray(masked_data, coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])], dims=['y', 'x'])
            final_ds = da.to_dataset(name=f'OSTIA_{hemisphere}_icecon')

            # 保存为 NetCDF
            output_file = os.path.join(output_dir, f"OSTIA_{hemisphere}_icecon_12500_{file[-11:-3]}.nc")
            final_ds.to_netcdf(output_file)
            generate_snapshot(masked_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'OSTIA', snapshot_folder)
            # print(f"Successfully saved final NetCDF: {output_file}")

        except Exception as e:
            print(f"Error processing file {file}: {e}")
            shutil.copy(file, failure_folder)  

if __name__ == "__main__":
    # --- 步骤3：运行北极和南极的整个处理流程 ---
    split_daily_nc(north_input_dir, north_output_dir, 'north') # 拆分北极每日数据
    process_daily_nc(north_output_dir, 'north', crs_psn, grid_x_n, grid_y_n, final_output_dir_north) # 处理北极每日数据

    split_daily_nc(south_input_dir, south_output_dir, 'south') # 拆分南极每日数据
    process_daily_nc(south_output_dir, 'south', crs_pss, grid_x_s, grid_y_s, final_output_dir_south) # 处理南极每日数据

    print("Success!")