"""
12.5km海冰密集度融合产品制作，采用kalman滤波算法
融合数据包括：NSIDC-AA、OSI-401-d、OSTIA、AMSRE
极点空白问题：设置极点周围150000公里内均为100%密集度
融合时，不为nan值计算权重。
"""
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

def calculate_combined_uncertainty(estimated_values, uncertainty_list, y_len, x_len):
    """
    结合多个数据集的估计值和不确定性
    使用加权平均法结合多个数据源，生成融合的海冰密集度数据
    """
    estimates_arrays = np.stack(estimated_values, axis=0)
    uncer_arrays = np.stack(uncertainty_list, axis=0)

    combined_values = np.full((y_len, x_len), np.nan)
    combined_uncertainty = np.full((y_len, x_len), np.nan)

    for i in range(y_len):
        for j in range(x_len):
            # 提取像素点的估计值和不确定性
            pixel_estimates = estimates_arrays[:, i, j]
            pixel_uncer = uncer_arrays[:, i, j]

            # 去除无效值
            valid_data_mask = ~np.isnan(pixel_estimates)
            valid_estimates = pixel_estimates[valid_data_mask]
            valid_uncer = pixel_uncer[valid_data_mask]

            if valid_estimates.size > 0:  # 如果有有效值
                # 计算权重
                uncer_inv = np.array([1 / u if u > 0 else 0 for u in valid_uncer])
                sum_uncer_inv = np.sum(uncer_inv)

                if sum_uncer_inv > 0:
                    # 计算加权平均值
                    weights = uncer_inv / sum_uncer_inv

                    # 如果权重和估计值的形状相同，则进行加权平均
                    if weights.shape == valid_estimates.shape:
                        combined_values[i, j] = np.sum(valid_estimates * weights)
                        combined_uncertainty[i, j] = 1 / sum_uncer_inv
                    else:
                        print(f"Warning: Shape mismatch at pixel ({i}, {j}). Skipping.")
                else:
                    # 如果权重和估计值的和为0，则直接取平均值
                    combined_values[i, j] = np.mean(valid_estimates)
                    combined_uncertainty[i, j] = np.var(valid_estimates) if len(valid_estimates) > 1 else valid_uncer[0] if valid_uncer.size > 0 else np.nan
            # else: combined_values[i, j] remains NaN (default)

    return combined_values, combined_uncertainty

# --- 数据处理函数 ---
def process_dataset_at_time(dataset_name, config, hemisphere, current_time, t_index, x_prev_dict, P_prev_dict, base_data_dir, output_base_dir, x, y):
    """
    处理给定数据集的给定时间步
    """

    print(f"  Processing dataset: {dataset_name}")
    folder_path = os.path.join(base_data_dir, config["path"], hemisphere)

    # 调整变量模式以匹配不同数据集
    adjusted_var_pattern = config["var_pattern"]
    if dataset_name == 'OSI-401-d':
        adjusted_var_pattern = f"OSI-401-d_{hemisphere}_icecon"
    elif dataset_name == 'OSTIA':
        adjusted_var_pattern = f"OSTIA_{hemisphere}_icecon"
    elif dataset_name == 'AMSRE':
        adjusted_var_pattern = f"UB-AMSRE_{hemisphere}_icecon"

    current_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                     f.endswith('.nc') and hemisphere in f and pd.to_datetime(f.split('_')[-1].split('.')[0],
                                                                             format='%Y%m%d') == current_time]

    if not current_files:
        print(f"    No data available for {dataset_name} at {current_time}, skipping...")
        return None, None

    current_file = current_files[0]

    try:
        current_data, _, var_name = read_data_single(current_file, adjusted_var_pattern)
    except ValueError as e:
        print(f"    Error reading data for {dataset_name} at {current_time}: {e}")
        return None, None

    # 初始化 x_prev 和 P_prev
    if dataset_name not in x_prev_dict:
        x_prev_dict[dataset_name] = current_data
        P_prev_dict[dataset_name] = np.full_like(current_data, 30)

    # 估算 Q 和 R
    if t_index > 0:
        Q, R = estimate_Q_R_single(x_prev_dict[dataset_name], current_data, default_Q=config["Q"],
                                   default_R=config["R"])
        print(f"Estimated Q: {Q}, Estimated R:{R}")
    else:
        Q = config["Q"]
        R = config["R"]

    # 设置F和H
    F = 1.0
    H = 1.0

    # 运行卡尔曼滤波
    x_current, P_current = kalman_filter_update(current_data, F, H, Q, R, x_prev_dict[dataset_name],
                                                P_prev_dict[dataset_name])

    # 保存卡尔曼滤波结果
    save_kalman_filter_results(dataset_name, hemisphere, current_time, x_current, P_current, output_base_dir, x, y)

    # 更新 x_prev 和 P_prev
    x_prev_dict[dataset_name] = x_current
    P_prev_dict[dataset_name] = P_current

    return x_current, P_current

def save_kalman_filter_results(dataset_name, hemisphere, current_time, x_current, P_current, output_base_dir, x, y):
    """
    保存kalman滤波结果为 NetCDF 文件.
    """
    time_str = current_time.strftime('%Y%m%d')
    output_dir = os.path.join(output_base_dir, hemisphere, f"{dataset_name}-Kalman")
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset_name}_{hemisphere}_12500_kalman_{time_str}.nc")
    dataset_data = xr.Dataset({
        'ice_con': (('y', 'x'), x_current),
        'ice_con_P': (('y', 'x'), P_current)
    }, coords={'time': current_time, 'x': x, 'y': y})
    dataset_data.to_netcdf(output_file)
    dataset_data.close()

def fuse_and_save_data(estimated_values, uncertainty_list, current_time, output_base_dir, hemisphere, x, y):
    """
    融合估计值和不确定性，保存融合结果为 NetCDF 文件.
    """
    if estimated_values:
        # 得到 x 和 y 的长度
        y_len = len(y)
        x_len = len(x)

        combined_values, combined_uncertainty = calculate_combined_uncertainty(
            estimated_values, uncertainty_list, y_len, x_len  
        )

        if hemisphere == "north":
            # 北极点坐标
            center_x = 3125
            center_y = 3125
            max_distance = 125000  # 125 km

            # 计算每个像素到北极点的距离
            distances_squared = (x - center_x)**2 + (y[:, np.newaxis] - center_y)**2

            # 确定在半径范围内的像素
            within_radius_mask = distances_squared <= max_distance**2

            # 将半径范围内的值设置为100
            combined_values[within_radius_mask] = 100

        # 保存融合结果
        time_str = current_time.strftime('%Y%m%d')
        output_dir = os.path.join(output_base_dir, hemisphere, "Icecon_Combined")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, f"icecon_{hemisphere}_12500_combined_{time_str}.nc")
        combined_data = xr.Dataset({
            'ice_con': (('y', 'x'), combined_values),
            'ice_con_P': (('y', 'x'), combined_uncertainty),
            'count_of_datasets': ((), len(estimated_values))
        }, coords={'time': current_time, 'x': x, 'y': y})
        combined_data.to_netcdf(output_file)
        combined_data.close()

        del combined_values, combined_uncertainty
        

# --- 画图 ---
def create_north_polar_plot(data_file, output_dir):
    """
    北极.
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

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_12500_{date_str}.png")
    plt.savefig(output_filename)

    plt.close(fig)
    combined_data.close()

def create_south_polar_plot(data_file, output_dir):
    """
    南极.
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

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_12500_{date_str}.png")
    plt.savefig(output_filename)

    plt.close(fig)
    combined_data.close()

# --- 主函数 ---
def main():
    base_data_dir = r"D:\zmh\icecon12500_data"
    hemispheres = ["north", "south"]
    dataset_configs = {
        "NSIDC-AA": {"path": "NSIDC-AA/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "OSI-401-d": {"path": "OSI-401-d/0_pre_com", "var_pattern": "OSI-401-d", "Q": 30, "R": 10},
        "OSTIA": {"path": "OSTIA/0_pre_com", "var_pattern": "OSTIA", "Q": 30, "R": 10},
        "AMSRE": {"path": "UB-AMSRE/0_pre_com", "var_pattern": "AMSRE", "Q": 30, "R": 10}
    }

    for hemisphere in hemispheres:
        print(f"Processing {hemisphere} hemisphere")
        output_base_dir = os.path.join(base_data_dir, "ICECON_merge_12500_kalmanV1.4")

        # 得到所有时间点
        all_times = set()
        for dataset_name, config in dataset_configs.items():
            folder_path = os.path.join(base_data_dir, config["path"], hemisphere)
            files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith('.nc') and hemisphere in f])
            for file in files:
                try:
                    _, time, _ = read_data_single(file, config["var_pattern"])
                    all_times.add(time)
                except ValueError as e:
                    print(f"Skipping file {file} due to error: {e}")
        sorted_times = sorted(list(all_times))

        # 初始化 x_prev 和 P_prev 字典
        x_prev_dict = {}
        P_prev_dict = {}

        # 得到 x 和 y
        for dataset_name, config in dataset_configs.items():
            folder_path = os.path.join(base_data_dir, config["path"], hemisphere)
            files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith('.nc') and hemisphere in f])
            if files:
                sample_ds = xr.open_dataset(files[0])
                x = sample_ds.x.values
                y = sample_ds.y.values
                sample_ds.close()
                break

        # 遍历所有时间点
        for t_index, current_time in enumerate(sorted_times):
            print(f"Processing time: {current_time}")

            estimated_values = []
            uncertainty_list = []

            # 处理每个数据集
            for dataset_name, config in dataset_configs.items():
                x_current, P_current = process_dataset_at_time(
                    dataset_name, config, hemisphere, current_time, t_index, x_prev_dict, P_prev_dict,
                    base_data_dir, output_base_dir, x, y
                )
                if x_current is not None and P_current is not None:
                    estimated_values.append(x_current)
                    uncertainty_list.append(P_current)

            # 融合数据
            fuse_and_save_data(estimated_values, uncertainty_list, current_time, output_base_dir, hemisphere, x, y)

            del estimated_values, uncertainty_list

    # --- 画图 ---
    data_dir = os.path.join(base_data_dir, "ICECON_merge_12500_kalmanV1.4")
    for hemisphere in hemispheres:
        output_dir = os.path.join(data_dir, "snapshots", hemisphere)
        os.makedirs(output_dir, exist_ok=True)
        for filename in sorted(os.listdir(os.path.join(data_dir, hemisphere, "Icecon_Combined"))):
            if filename.endswith(".nc") and "combined" in filename:
                data_file = os.path.join(data_dir, hemisphere, "Icecon_Combined", filename)
                if hemisphere == "north":
                    create_north_polar_plot(data_file, output_dir)
                elif hemisphere == "south":
                    create_south_polar_plot(data_file, output_dir)

if __name__ == "__main__":
    main()
    print("SUCCESS!")
