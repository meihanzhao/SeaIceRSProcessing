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
def kalman_filter(z, F, H, Q, R, x0, P0):
    """
    实现1D时间序列的卡尔曼滤波算法
    输入：
    z - 观测值序列
    F - 状态转移矩阵
    H - 观测矩阵
    Q - 过程噪声协方差
    R - 观测噪声协方差
    x0 - 初始状态估计
    P0 - 初始协方差估计
    输出：
    x_estimates - 状态估计序列
    P_estimates - 协方差估计序列
    核心思想：将观测数据 z 与模型预测结合，通过卡尔曼滤波公式获得更精确的估计值。
    """
    T = len(z)
    x_estimates = np.zeros(T)
    P_estimates = np.zeros(T)
    x_prev = x0
    P_prev = P0

    for k in range(T):
        x_pred = F * x_prev # 预测状态
        P_pred = F * P_prev * F + Q # 预测协方差
        if np.isnan(z[k]): # 如果观测值为缺失值
            x_est = x_pred
            P_est = P_pred
        else:
            y = z[k] - (H * x_pred) # 计算残差
            S = H * P_pred * H + R # 计算残差协方差
            K = P_pred * H / S # 计算卡尔曼增益
            x_est = x_pred + K * y # 更新状态估计
            P_est = (1 - K * H) * P_pred # 更新协方差估计
        # 保存状态估计和协方差估计
        x_estimates[k] = x_est
        P_estimates[k] = P_est
        x_prev = x_est
        P_prev = P_est
    return x_estimates, P_estimates

def apply_kalman_filter_3d(data, F, H, Q, R, x0_func, P0_func):
    """"
    对3D时间序列数据应用卡尔曼滤波
    输入：
    data - 3D时间序列数据
    F - 状态转移矩阵
    H - 观测矩阵
    Q - 过程噪声协方差
    R - 观测噪声协方差
    x0_func - 初始状态估计函数
    P0_func - 初始协方差估计函数
    输出：
    smoothed_data - 平滑后的数据
    smoothed_P - 平滑后的协方差
    核心：遍历空间网格。为每个像素时间序列单独应用卡尔曼滤波。
    """
    T, M, N = data.shape
    smoothed_data = np.full_like(data, np.nan, dtype=np.float64)
    smoothed_P = np.full_like(data, np.nan, dtype=np.float64)
    for m in range(M):
        for n in range(N):
            z = data[:, m, n] # 提取每个像素点的时间序列
            first_valid = np.argmax(~np.isnan(z)) # 找到第一个非NaN值索引
            if np.all(np.isnan(z)): # 若全为NaN，跳过
                continue
            x0 = x0_func(z, first_valid) # 初始化状态
            P0 = P0_func(z, first_valid) # 初始化协方差
            z_valid = z[first_valid:] # 仅对有效数据应用滤波
            estimates, P_estimates = kalman_filter(z_valid, F, H, Q, R, x0, P0)
            smoothed = np.full(T, np.nan)
            smoothed_P_pixel = np.full(T, np.nan)
            smoothed[first_valid:] = estimates
            smoothed_P_pixel[first_valid:] = P_estimates
            smoothed_data[:, m, n] = smoothed
            smoothed_P[:, m, n] = smoothed_P_pixel
    return smoothed_data, smoothed_P

def estimate_Q_R(data):
    """估算过程噪声Q和测量噪声R"""
    ice_conc_diff = np.diff(data, axis=0) # 时间差分
    process_variances = np.nanvar(ice_conc_diff, axis=0)
    Q = np.nanmean(process_variances) # 平均过程噪声
    measurement_variances = np.nanvar(data, axis=0)
    R = np.nanmean(measurement_variances) # 平均测量噪声
    return Q, R

def initialize_x0(z, first_valid):
    """Initializes the initial state estimate x0."""
    return z[first_valid] if not np.isnan(z[first_valid]) else 0.0

def initialize_P0(z, first_valid):
    """Initializes the initial estimate covariance P0."""
    return 30

def visualize_covariance(original, filtered, P_estimates, m, n, T):
    """Plots original data, filtered estimates, and confidence intervals."""
    try:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        ax.set_xlabel("Time Steps")
        ax.set_ylabel("Sea Ice Concentration (%)")
        ax.plot(range(T), original, '.', color='#0868ac', label="Original Data", alpha=0.5)
        ax.plot(range(T), filtered, '-', color="#d73027", label="Kalman Filter Estimate", linewidth=2)
        std_dev = np.sqrt(P_estimates)
        ax.fill_between(
            range(T),
            filtered - 2 * std_dev,
            filtered + 2 * std_dev,
            color="#fdae61", alpha=0.3, label="95% Confidence Interval"
        )
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)
        ax.legend(loc="upper left")
        plt.title(f"Kalman Filter Results for Pixel ({m}, {n})", fontsize=15)
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Error in visualize_covariance: {e}")

# --- Data Reading and Kalman Filter Application ---

def read_data(folder_path, var_name_pattern, hemisphere):
    """读取文件夹中的海冰密集度数据（NetCDF 格式），并合并为 xarray.Dataset"""
    files = sorted([os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.nc')])
    
    # 根据半球过滤文件
    if hemisphere == 'north':
        files = [f for f in files if 'north' in f]
    elif hemisphere == 'south':
        files = [f for f in files if 'south' in f]

    # 提取时间戳并打开数据集    
    time_stamps = [pd.to_datetime(file.split('_')[-1].split('.')[0], format='%Y%m%d') for file in files]
    datasets = [xr.open_dataset(file) for file in files]

    # 确定变量名称
    var_name = None
    for ds in datasets:
        var_name = next((name for name in ds.data_vars if var_name_pattern in name), None)
        if var_name:
            break
    if var_name is None:
        raise ValueError(f"No variable found in datasets matching pattern '{var_name_pattern}' for {hemisphere} hemisphere in {folder_path}")
    
    # 合并数据集
    datasets = [ds.assign_coords(time=time).rename({var_name: 'ice_con'}) for ds, time in zip(datasets, time_stamps)]
    combined_data = xr.concat(datasets, dim='time').sortby('time')
    return combined_data

def apply_kalman_dataset(combined_data, Q=30, R=10, m=400, n=305):
    """Applies Kalman Filter to the dataset and visualizes results."""
    measurements = combined_data['ice_con'].values
    T = measurements.shape[0]
    Q, R = estimate_Q_R(measurements)
    # print(f"Estimated Q: {Q}, Estimated R: {R}")
    F = 1.0
    H = 1.0
    smoothed, smoothed_P = apply_kalman_filter_3d(
        data=measurements, F=F, H=H, Q=Q, R=R,
        x0_func=initialize_x0, P0_func=initialize_P0
    )
    original_series = measurements[:, m, n]
    filtered_series = smoothed[:, m, n]
    P_estimates_series = smoothed_P[:, m, n]
    visualize_covariance(original_series, filtered_series, P_estimates_series, m, n, T)

    # (Optional) Spatial distribution visualization
    time_step = T - 1
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(measurements[time_step], cmap='viridis')
    plt.title(f'Original Data at Time {time_step}')
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.imshow(smoothed[time_step], cmap='viridis')
    plt.title(f'Kalman Filtered Data at Time {time_step}')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

    combined_data['ice_concentration_smoothed'] = (('time', 'y', 'x'), smoothed)
    combined_data['ice_concentration_smoothed_P'] = (('time', 'y', 'x'), smoothed_P)
    return combined_data

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


if __name__ == "__main__":
    base_data_dir = r"D:\zmh\icecon12500_data"
    hemispheres = ["north", "south"]
    dataset_configs = {
        "NSIDC-AA": {"path": "NSIDC-AA/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "OSI-401-d": {"path": "OSI-401-d/0_pre_com", "var_pattern": "OSI-401-d", "Q": 30, "R": 10},
        "OSTIA": {"path": "OSTIA/0_pre_com", "var_pattern": "OSTIA", "Q": 30, "R": 10}
    }

    kalman_filtered_data = {}

    for hemisphere in hemispheres:
        kalman_filtered_data[hemisphere] = {}
        for dataset_name, config in dataset_configs.items():
            folder_path = os.path.join(base_data_dir, config["path"], hemisphere)
            # print(f"Processing {dataset_name} for {hemisphere} hemisphere from {folder_path}")

            if dataset_name == 'OSI-401-d':
                if hemisphere == 'south':
                    config["var_pattern"] = "OSI-401-d_south_icecon"
                    config["Q"] = 8
                else:
                    config["var_pattern"] = "OSI-401-d_north_icecon"
                    config["Q"] = 30
            elif dataset_name == 'OSTIA':
                if hemisphere == 'south':
                    config["var_pattern"] = "OSTIA_south_icecon"
                    config["Q"] = 6
                else:
                    config["var_pattern"] = "OSTIA_north_icecon"
                    config["Q"] = 30

            combined_data = read_data(folder_path, config["var_pattern"], hemisphere)

            m = 400 if hemisphere == "north" else 210
            n = 305 if hemisphere == "north" else 210

            kalman_filtered = apply_kalman_dataset(
                combined_data, Q=config["Q"], R=config["R"], m=m, n=n
            )
            kalman_filtered_data[hemisphere][dataset_name] = kalman_filtered


    for hemisphere in hemispheres:
        x = kalman_filtered_data[hemisphere]['NSIDC-AA']['x']
        y = kalman_filtered_data[hemisphere]['NSIDC-AA']['y']

        # 联合时间维度
        time_union = np.unique(np.concatenate([
            kalman_filtered_data[hemisphere][dataset_name]['time'].values
            for dataset_name in dataset_configs
        ]))

        output_dir = os.path.join(base_data_dir, "ICECON_merge_12500", hemisphere)
        os.makedirs(output_dir, exist_ok=True)

        for time in time_union:
            time_str = pd.to_datetime(time).strftime('%Y%m%d')
            # print(f"Processing {time_str} for {hemisphere} hemisphere")
            estimated_values = []
            uncertainty_list = []
            for dataset_name in dataset_configs:
                if time in kalman_filtered_data[hemisphere][dataset_name]['time']:
                    data_slice = kalman_filtered_data[hemisphere][dataset_name].sel(time=time)
                    estimated_values.append(data_slice['ice_concentration_smoothed'].values)
                    uncertainty_list.append(data_slice['ice_concentration_smoothed_P'].values)

            if len(estimated_values) > 0:
                combined_values, combined_uncertainty = calculate_combined_uncertainty(
                    estimated_values, uncertainty_list
                )

                combined_data = xr.Dataset({
                    'ice_con': (('y', 'x'), combined_values),
                    'ice_con_P': (('y', 'x'), combined_uncertainty),
                    'count_of_datasets': ((), len(estimated_values))
                }, coords={'time': time, 'x': x, 'y': y})

                output_file = os.path.join(output_dir, f"icecon_{hemisphere}_12500_combined_{time_str}.nc")
                combined_data.to_netcdf(output_file)

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

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_12500_{date_str}.png")
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

    output_filename = os.path.join(output_dir, f"海冰密集度融合产品_12500_{date_str}.png")
    plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")

    plt.close(fig)
    combined_data.close()
    
data_dir = os.path.join(base_data_dir, "ICECON_merge_12500")
for hemisphere in hemispheres:
    output_dir = os.path.join(data_dir, "snapshots", hemisphere)
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(os.path.join(data_dir, hemisphere)):
        if filename.endswith(".nc"):
            data_file = os.path.join(data_dir, hemisphere, filename)
            if hemisphere == "north":
                create_north_polar_plot(data_file, output_dir)
            elif hemisphere == "south":
                create_south_polar_plot(data_file, output_dir)