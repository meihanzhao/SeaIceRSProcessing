# -*- coding: utf-8 -*-
"""
海冰厚度最优插值数据融合 - 自动日期处理版
作者：ZMH
日期：2025-02-17

说明：
    本代码实现基于背景场（APP-X）和观测数据（CMEMS、CS2SMOS、NSIDC‑rdeft4）的最优插值数据融合，
    并通过自动扫描多个日期文件进行批量处理。文件路径使用模板，日期部分由YYYYMMDD替换得到。
    
    核心特点：
      1. 针对观测数据中的负值进行过滤，确保数据合理性；
      2. 模块化设计的主函数，结构清晰，便于维护与扩展；
      3. 每个工具函数均包含功能说明、输入参数和输出说明，便于调试和后续优化。
      
增强：
    1. 自动扫描文件夹以获取日期文件；
    2. 错误捕获和日志记录便于处理大量文件；
    3. 集中路径管理，便于扩展配置。
"""

import os
import re
import numpy as np
import netCDF4 as nc
from scipy.spatial import KDTree
from glob import glob
import paths as paths

# ------------------------------
# 全局参数及配置
# ------------------------------

# 影响半径和相关长度尺度 ξ（单位：米）
R_influence = 250000.0  # 250 km
xi = 100000.0  # 100 km

# 待融合数据不确定性（单位：m）
uncertainty_dict = {
    "appx": 0.6,    # APP-X 模式背景场
    "cmems": 0.7,   # CMEMS 数据
    "CS2": 0.5,     # CS2SMOS 数据
    "rdeft4": 0.6,  # NSIDC-rdeft4 融合数据
}

# 路径，利用 paths 模块集中管理路径
file_path_templates = {
    "appx":  os.path.join(paths.paths['appx_OI_path'], 'APPX_north_SIT_25000_{date}.nc'),
    "cmems": os.path.join(paths.paths['cmems_OI_path'], 'cmems_pdfcorrected_north_SIT_25000_{date}.nc'),
    "CS2":   os.path.join(paths.paths['CS2_OI_path'], 'CS2_pdfcorrected_north_SIT_25000_{date}.nc'),
    "rdeft4": os.path.join(paths.paths['rdeft4_OI_path'], 'rdeft4_pdfcorrected_north_SIT_25000_{date}.nc'),
}

# 变量名称
var_names = {
    "appx": "appx_sit",
    "cmems": "cmems_sit",
    "CS2": "CS2_sit",
    "rdeft4": "rdeft4_sit",
}

# ------------------------------
# 工具函数
# ------------------------------

def read_nc_data(file_path, var_name):
    """
    功能：
        读取NetCDF文件，提取指定变量数据及其坐标信息。
    输入参数：
        file_path - NetCDF文件路径（字符串）。
        var_name  - 待读取变量名称（字符串）。
    输出：
        返回元组 (data, x, y)：
            data - 变量数据数组；
            x    - x方向坐标数组；
            y    - y方向坐标数组。
    """
    ds = nc.Dataset(file_path)
    data = ds.variables[var_name][:].data  # 读取变量数据
    x = ds.variables["x"][:]                # x 坐标数组
    y = ds.variables["y"][:]                # y 坐标数组
    ds.close()
    return data, x, y

def preprocess_data(data):
    """
    功能：
        对观测数据进行预处理，过滤掉小于0的非物理值。
    输入参数：
        data - 原始数据数组。
    输出：
        返回预处理后的数据数组。
    """
    data[data < 0] = 0
    return data

def gaussian_weight(distance, xi, sigma):
    """
    功能：
        根据欧氏距离、相关尺度xi以及数据不确定性sigma计算高斯权重。
    输入参数：
        distance - 分析点与观测点间距离（单位：米）。
        xi       - 相关长度尺度（单位：米）。
        sigma    - 数据不确定性（单位：米）。
    输出：
        返回计算得到的权重值；若距离超过影响半径，则返回0。
    """
    if distance > R_influence:
        return 0
    return (1 + distance / xi) * np.exp(-distance / xi) / sigma

def collect_observations(file_paths):
    """
    功能：
        收集并整理观测数据，获取有效观测点位置、不含负值的数据及对应的不确定性。
    输入参数：
        file_paths - 包含各数据产品文件路径的字典。
    输出：
        返回元组 (obs_points, obs_values, obs_uncertainties)：
            obs_points         - 有效观测点坐标列表。
            obs_values         - 观测数据值列表。
            obs_uncertainties  - 观测数据不确定性列表。
    """
    obs_points, obs_values, obs_uncertainties = [], [], []

    for key, path in file_paths.items():
        if key == "appx":   # APP-X数据是背景场，不用于收集观测点
            continue
        if os.path.exists(path):
            print(f"读取 {key} 数据：{path}")
            data, obs_x, obs_y = read_nc_data(path, var_names[key])
            data = preprocess_data(data)  # 过滤非物理负值
            sigma = uncertainty_dict.get(key, 1.0)
            mask = ~np.isnan(data)        # 筛选有效数据点
            points = np.array([(obs_x[j], obs_y[i]) for i, j in zip(*np.where(mask))])
            values = data[mask]
            uncertainties = np.full(len(values), sigma)
            obs_points.extend(points.tolist())
            obs_values.extend(values.tolist())
            obs_uncertainties.extend(uncertainties.tolist())
        else:
            print(f"文件不存在：{path}")

    return obs_points, obs_values, obs_uncertainties

def optimal_interpolation(bg_data, x_coords, y_coords, obs_points, obs_values, obs_uncertainties):
    """
    功能：
        利用最优插值方法，将背景场和观测数据相融合，得到分析场结果。
    输入参数：
        bg_data         - 背景场数据数组（来自APP-X）。
        x_coords, y_coords - 背景场坐标数组。
        obs_points      - 观测点坐标列表。
        obs_values      - 观测数据值列表。
        obs_uncertainties - 观测不确定性列表。
    输出：
        返回融合后的分析场数据数组。
    """
    ny, nx = bg_data.shape
    analysis = np.copy(bg_data)
    obs_kd_tree = KDTree(obs_points)  # 构建KDTree索引

    for i in range(ny):
        for j in range(nx):
            x0, y0 = x_coords[j], y_coords[i]
            bg_val = bg_data[i, j]
            indices = obs_kd_tree.query_ball_point([x0, y0], R_influence)
            if len(indices) == 0:
                analysis[i, j] = bg_val
                continue
            numer, denom = 0.0, 0.0
            for idx in indices:
                obs_x, obs_y = obs_points[idx]
                obs_val = obs_values[idx]
                sigma = obs_uncertainties[idx]
                distance = np.sqrt((x0 - obs_x)**2 + (y0 - obs_y)**2)
                weight = gaussian_weight(distance, xi, sigma)
                if weight > 0:
                    numer += weight * (obs_val - bg_val)
                    denom += weight
            analysis[i, j] = bg_val + numer / denom if denom > 0 else bg_val
    return analysis

def ensure_dir_exists(directory):
    """
    功能：
        检查目录是否存在，若不存在则创建目录。
    输入参数：
        directory - 目标目录路径（字符串）。
    输出：
        无返回值，但确保目录存在。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_to_netcdf(out_file, analysis, x_coords, y_coords):
    """
    功能：
        将融合结果写入NetCDF文件，并自动创建输出目录（若不存在）。
    输入参数：
        out_file - 输出文件完整路径（字符串）。
        analysis - 分析结果数据数组。
        x_coords - x方向坐标数组。
        y_coords - y方向坐标数组。
    输出：
        无返回值，但生成NetCDF格式的输出文件。
    """
    output_dir = os.path.dirname(out_file)
    ensure_dir_exists(output_dir)

    ds_out = nc.Dataset(out_file, "w", format="NETCDF4")
    ds_out.createDimension("x", len(x_coords))
    ds_out.createDimension("y", len(y_coords))

    x_var = ds_out.createVariable("x", "f4", ("x",))
    y_var = ds_out.createVariable("y", "f4", ("y",))
    x_var[:] = x_coords
    y_var[:] = y_coords

    sit_var = ds_out.createVariable("SIT", "f4", ("y", "x"))
    sit_var[:] = analysis

    ds_out.description = "基于最优插值的多观测数据海冰厚度融合结果"
    ds_out.close()
    print(f"融合结果已保存至: {out_file}")

def auto_discover_dates(pattern, path):
    """
    功能：
        自动扫描指定文件夹，匹配符合模式的文件名并提取日期（格式：YYYYMMDD）。
    输入参数：
        pattern - 文件匹配模式（例如："APPX_north_SIT_25000_*.nc"）。
        path    - 扫描的目标目录路径（字符串）。
    输出：
        返回提取到的日期字符串列表（已排序）。
    """
    dates = set()
    for file in glob(os.path.join(path, pattern)):
        match = re.search(r"\d{8}", file)
        if match:
            dates.add(match.group(0))
    return sorted(dates)

def process_one_day(date_str):
    """
    功能：
        对单个日期的各产品数据进行最优插值处理，并保存结果为NetCDF文件。
    输入参数：
        date_str - 日期字符串（格式：YYYYMMDD）。
    输出：
        无返回值，但处理过程中会打印日志信息，输出NetCDF文件。
    """
    print("==============================================")
    print(f"开始处理日期：{date_str}")

    # 构建各数据产品的文件路径字典
    file_paths = {key: template.format(date=date_str) for key, template in file_path_templates.items()}

    # 判断背景场文件是否存在
    if not os.path.exists(file_paths["appx"]):
        print(f"背景场文件不存在：{file_paths['appx']}")
        return
    bg_data, x_coords, y_coords = read_nc_data(file_paths["appx"], var_names["appx"])

    # 收集除背景场外的观测数据
    obs_points, obs_values, obs_uncertainties = collect_observations(file_paths)

    # 进行最优插值数据融合
    analysis = optimal_interpolation(bg_data, x_coords, y_coords, obs_points, obs_values, obs_uncertainties)

    # 构造输出文件路径，输出目录由 paths 模块配置
    out_file = os.path.join(paths.paths['output_OI_path'], f"Arctic_SIT_OI_R25km_{date_str}.nc")
    save_to_netcdf(out_file, analysis, x_coords, y_coords)

# ------------------------------
# 主函数
# ------------------------------

def main():
    """
    功能：
        主程序，通过自动扫描背景场目录中的日期文件，依次对每个日期执行数据融合处理。
    输入参数：
        无参数（直接从配置中加载）。
    输出：
        无返回值，但在处理过程中打印处理进度和日志信息。
    """
    appx_path = paths.paths["appx_OI_path"]
    date_list = auto_discover_dates("APPX_north_SIT_25000_*.nc", appx_path) 
    print(f"发现 {len(date_list)} 个日期文件，开始处理...")

    for date_str in date_list:
        process_one_day(date_str) 

if __name__ == "__main__":
    main()