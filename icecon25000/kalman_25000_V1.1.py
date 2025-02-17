# -*- coding: utf-8 -*-
import xarray as xr
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import LinearSegmentedColormap
from pyproj import CRS, Transformer, Proj, pyproj
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams['font.sans-serif'] = ['SimHei']

# --- 卡尔曼滤波函数 ---
def kalman_filter_update(z_current, F, H, Q, R, x_prev, P_prev):
    """
    对单个时间步执行卡尔曼滤波更新
    输入：
    z_current - 当前时刻的观测值 (二维数组)
    F - 状态转移矩阵
    H - 观测矩阵
    Q - 过程噪声协方差
    R - 观测噪声协方差
    x_prev - 上一时刻的状态估计 (二维数组)
    P_prev - 上一时刻的协方差估计 (二维数组)
    输出：
    x_current - 当前时刻的状态估计 (二维数组)
    P_current - 当前时刻的协方差估计 (二维数组)
    """
    # 预测步
    x_pred = F * x_prev
    P_pred = F * P_prev * F + Q

    # 更新步
    y = np.where(np.isnan(z_current), 0, z_current - (H * x_pred))
    S = H * P_pred * H + R
    K = np.where(np.isnan(z_current), 0, P_pred * H / S)
    x_current = np.where(np.isnan(z_current), x_pred, x_pred + K * y)
    P_current = np.where(np.isnan(z_current), P_pred, (1 - K * H) * P_pred)

    return x_current, P_current

def estimate_Q_R_single(data_prev, data_current, default_Q=30, default_R=10):
    """估算单个时间步的过程噪声Q和测量噪声R"""
    ice_conc_diff = data_current - data_prev
    process_variances = np.nanvar(ice_conc_diff)
    Q = process_variances 
    measurement_variances = np.nanvar(data_current)
    R = measurement_variances 
    return Q, R

# --- 读取数据函数 ---
def read_data_single(file_path, var_name_pattern):
    """读取单个 NetCDF 文件，返回指定变量的数据和从文件名中提取的时间"""
    ds = xr.open_dataset(file_path)
    var_name = next((name for name in ds.data_vars if var_name_pattern in name), None)
    if var_name is None:
        ds.close()
        raise ValueError(f"No variable found in file matching pattern '{var_name_pattern}'")
    data = ds[var_name].values
    # 从文件名中提取时间
    filename = os.path.basename(file_path)
    try:
        time_str = filename.split('_')[-1].split('.')[0]
        time = pd.to_datetime(time_str, format='%Y%m%d')
    except (ValueError, IndexError):
        ds.close()
        raise ValueError(f"Could not extract time from filename: {filename}")
    ds.close()
    return data, time, var_name

# --- 数据融合函数 ---
def calculate_combined_uncertainty(estimated_values, uncertainty_list):
    """
    结合多个数据集的估计值和不确定性
    使用加权平均法结合多个数据源，生成融合的海冰密集度数据
    """
    estimates_arrays = np.stack(estimated_values, axis=0)
    uncer_arrays = np.stack(uncertainty_list, axis=0)
    inverse_uncer_arrays = 1 / uncer_arrays + 1e-10
    sum_inverse_uncer_arrays = np.where(
        np.isnan(inverse_uncer_arrays).all(axis=0),
        np.nan,
        np.nansum(inverse_uncer_arrays, axis=0)
    )
    combined_uncertainty = 1 / sum_inverse_uncer_arrays
    weights = np.where(
        np.isnan(sum_inverse_uncer_arrays),
        np.nan,
        inverse_uncer_arrays / sum_inverse_uncer_arrays
    )
    weighted_estimates = weights * estimates_arrays
    combined_values = np.where(
        np.isnan(weighted_estimates).all(axis=0),
        np.nan,
        np.nansum(weighted_estimates, axis=0)
    )
    return combined_values, combined_uncertainty

# --- 主程序 ---
if __name__ == "__main__":
    base_data_dir = r"D:\zmh\icecon25000_data"
    hemispheres = ["south"]
    dataset_configs = {
        "NISE": {"path": "NISE/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "NSIDC-Bootstrap": {"path": "NSIDC-Bootstrap/0_pre_com", "var_pattern": "NSIDC-Bootstrap", "Q": 30, "R": 10},
        "FY3C": {"path": "FY/FY-3C/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "FY3D": {"path": "FY/FY-3D/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
    }

    for hemisphere in hemispheres:
        print(f"Processing {hemisphere} hemisphere")
        output_base_dir = os.path.join(base_data_dir, "ICECON_merge_25000_OISST_North") 

        # 获取所有数据集的时间列表
        all_times = set()
        for dataset_name, config in dataset_configs.items():
            folder_path = os.path.join(base_data_dir, config["path"], hemisphere)
            files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc') and hemisphere in f])
            for file in files:
                try:
                    _, time, _ = read_data_single(file, config["var_pattern"])
                    all_times.add(time)
                except ValueError as e:
                    print(f"Skipping file {file} due to error: {e}")
        sorted_times = sorted(list(all_times))

        # 初始化每个数据集的上一时刻状态和协方差
        x_prev_dict = {}
        P_prev_dict = {}

        # 循环处理每个时间点
        for t_index, current_time in enumerate(sorted_times):
            print(f"Processing time: {current_time}")

            # 存储当前时刻所有数据集的估计值和不确定性，用于最终融合
            estimated_values = []
            uncertainty_list = []

            # 循环处理每个数据集
            for dataset_name, config in dataset_configs.items():
                if hemisphere == "south" and dataset_name == "OISST":
                    print(f"  Skipping dataset: {dataset_name} for south hemisphere")
                    continue
                print(f"  Processing dataset: {dataset_name}")
                folder_path = os.path.join(base_data_dir, config["path"], hemisphere)

                # 根据半球调整变量名称和参数
                adjusted_var_pattern = config["var_pattern"]
                if dataset_name == 'NSIDC-Bootstrap':
                    if hemisphere == 'south':
                        adjusted_var_pattern = "NSIDC-BS_south_icecon"
                        # config["Q"] = 8
                    else:
                        adjusted_var_pattern = "NSIDC-BS_north_icecon"
                        # config["Q"] = 30
                        
                current_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.nc') and hemisphere in f and pd.to_datetime(f.split('_')[-1].split('.')[0], format='%Y%m%d') == current_time]

                if not current_files:
                    print(f"    No data available for {dataset_name} at {current_time}, skipping...")
                    continue
                
                current_file = current_files[0] # 每个时间点每个数据集只有一个文件
                
                try:
                    current_data, _, var_name = read_data_single(current_file, adjusted_var_pattern)
                except ValueError as e:
                    print(f"    Error reading data for {dataset_name} at {current_time}: {e}")
                    continue

                # 获取x和y的坐标信息（所有数据集的坐标相同）
                if t_index == 0:
                    sample_ds = xr.open_dataset(current_file)
                    x = sample_ds.x.values
                    y = sample_ds.y.values
                    sample_ds.close()

                # 初始化或更新状态和协方差
                if dataset_name not in x_prev_dict:
                    x_prev_dict[dataset_name] = current_data
                    P_prev_dict[dataset_name] = np.full_like(current_data, 30) # 使用固定值初始化

                # 动态估计 Q 和 R (如果不是第一个时间步)
                if t_index > 0:
                    Q, R = estimate_Q_R_single(x_prev_dict[dataset_name], current_data, default_Q=config["Q"], default_R=config["R"])
                    print(f"Estimated Q: {Q}, Estimated R:{R}")
                else:
                    Q = config["Q"]
                    R = config["R"]
                
                # 设置 F 和 H
                F = 1.0
                H = 1.0

                # 执行卡尔曼滤波更新
                x_current, P_current = kalman_filter_update(current_data, F, H, Q, R, x_prev_dict[dataset_name], P_prev_dict[dataset_name])

                # 存储当前数据集的估计值和不确定性
                estimated_values.append(x_current)
                uncertainty_list.append(P_current)

                # 保存当前数据集的结果到指定文件夹
                time_str = current_time.strftime('%Y%m%d')
                output_dir = os.path.join(output_base_dir, hemisphere, f"{dataset_name}-Kalman")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{dataset_name}_{hemisphere}_25000_kalman_{time_str}.nc")
                dataset_data = xr.Dataset({
                    'ice_con': (('y', 'x'), x_current),
                    'ice_con_P': (('y', 'x'), P_current)
                }, coords={'time': current_time, 'x': x, 'y': y})
                dataset_data.to_netcdf(output_file)
                dataset_data.close()

                # 释放内存，并更新 x_prev 和 P_prev
                del x_prev_dict[dataset_name], P_prev_dict[dataset_name]
                x_prev_dict[dataset_name] = x_current
                P_prev_dict[dataset_name] = P_current
                del x_current, P_current, current_data

            # --- 所有数据集处理完毕，执行融合 ---
            if estimated_values:
                combined_values, combined_uncertainty = calculate_combined_uncertainty(estimated_values, uncertainty_list)
                
                # 保存融合结果到指定文件夹
                output_dir = os.path.join(output_base_dir, hemisphere, "Icecon_Combined")
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"icecon_{hemisphere}_25000_combined_{time_str}.nc")
                combined_data = xr.Dataset({
                        'ice_con': (('y', 'x'), combined_values),
                        'ice_con_P': (('y', 'x'), combined_uncertainty),
                        'count_of_datasets': ((), len(estimated_values))
                    }, coords={'time': current_time, 'x': x, 'y': y})
                combined_data.to_netcdf(output_file)
                combined_data.close()
            
            # 释放内存
            del estimated_values, uncertainty_list, combined_values, combined_uncertainty

# --- 画图函数 --- 
def create_north_polar_plot(data_file, output_dir):
    """
    创建北极立体投影图
    """
    colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
    cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)
    proj_crs = ccrs.NorthPolarStereo(central_longitude=-45)
    proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)

    combined_data = xr.open_dataset(data_file)
    date_str = os.path.basename(data_file).split('_')[4].split('.')[0] 

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj_crs})
    extent = [-180, 180, 58.5, 90]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon_grid, lat_grid = np.meshgrid(combined_data.x, combined_data.y)
    lon, lat = proj(lon_grid, lat_grid, inverse=True)

    ice_data = combined_data['ice_con'].astype(float)
    ice_data = np.ma.masked_invalid(ice_data)

    mappable = ax.pcolormesh(
        lon, lat, ice_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading='auto',
        vmin=0, vmax=100
    )

    ax.set_title(f'海冰密集度融合产品 ({date_str})', fontproperties='SimHei')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.coastlines(resolution='50m', color='black', linewidth=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([60, 70, 80])
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.ylabels_left = True
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 10, 'color': 'black'}
    ax.text(-45, 60, '45°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
    ax.text(-135, 62, '135°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
    ax.text(45, 61.5, '45°E', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')

    cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8, ax=ax)
    cbar.set_label('海冰密集度 (%)')

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_25000_{date_str}.png")
    plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")

    plt.close(fig)
    combined_data.close()

def create_south_polar_plot(data_file, output_dir):
    """
    创建南极立体投影图
    """
    colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
    cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)
    proj_crs = ccrs.SouthPolarStereo()
    proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=0)

    combined_data = xr.open_dataset(data_file)
    date_str = os.path.basename(data_file).split('_')[4].split('.')[0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj_crs})
    extent = [-180, 180, -90, -60]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon_grid, lat_grid = np.meshgrid(combined_data.x, combined_data.y)
    lon, lat = proj(lon_grid, lat_grid, inverse=True)

    ice_data = combined_data['ice_con'].astype(float)
    ice_data = np.ma.masked_invalid(ice_data)

    mappable = ax.pcolormesh(
        lon, lat, ice_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        shading='auto',
        vmin=0, vmax=100
    )

    ax.set_title(f'海冰密集度融合产品 ({date_str})', fontproperties='SimHei')
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.coastlines(resolution='50m', color='black', linewidth=1)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                      linewidth=1, color='gray', alpha=0.5, linestyle='--')
    gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
    gl.ylocator = mticker.FixedLocator([-80,-70,-60])
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = False
    gl.ylabels_left = True
    gl.xlabels_top = False
    gl.xlabels_bottom = False
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black'}
    gl.ylabel_style = {'size': 11, 'color': 'black'}
    ax.text(0, -61, '0°', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
    ax.text(180, -61.5, '180°', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
    ax.text(-90, -62.5, '90°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
    ax.text(90, -62.5, '90°E', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')

    cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8, ax=ax)
    cbar.set_label('海冰密集度 (%)')

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_25000_{date_str}.png")
    plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")

    plt.close(fig)
    combined_data.close()

data_dir = os.path.join(base_data_dir, "ICECON_merge_25000")
for hemisphere in hemispheres:
    output_dir = os.path.join(data_dir, "snapshots", hemisphere)
    os.makedirs(output_dir, exist_ok=True)
    for filename in sorted(os.listdir(os.path.join(data_dir, hemisphere,"Icecon_Combined"))):
        if filename.endswith(".nc") and "combined" in filename: # 只画融合后的结果
            data_file = os.path.join(data_dir, hemisphere,"Icecon_Combined", filename)
            if hemisphere == "north":
                create_north_polar_plot(data_file, output_dir)
            elif hemisphere == "south":
                create_south_polar_plot(data_file, output_dir)