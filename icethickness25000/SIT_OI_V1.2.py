# -*- coding: utf-8 -*-
"""
海冰厚度最优插值数据融合升级版：补充背景场空缺值

核心功能：
1. 读取背景场（APP-X）数据和其他观测数据（CMEMS、CS2、NSIDC‑rdeft4）；
2. 对观测数据进行预处理，过滤非物理负值；
3. 针对APP-X存在大量空缺值时，用其他数据产品按不确定性优先级补充填充背景场空缺；
4. 利用最优插值方法融合背景场与各观测数据，生成最终海冰厚度场；
5. 自动扫描文件夹对多个日期依次处理，并将结果保存为NetCDF格式。

作者：ZMH
日期：2025-02-17
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

# 影响半径（单位：米）
R_influence = 250000.0  # 250 km

# 相关长度尺度（单位：米）
xi = 100000.0  # 100 km

# 数据不确定性配置（单位：m）
uncertainty_dict = {
    "appx": 0.6,    # APP-X 背景场
    "cmems": 0.6,   # CMEMS 数据
    "CS2": 0.5,     # CS2SMOS 数据
    "rdeft4": 0.7,  # NSIDC-rdeft4 数据
}

# 路径（使用 paths 模块集中管理路径）
file_path_templates = {
    "appx": os.path.join(paths.paths['appx_OI_path'], 'APPX_north_SIT_25000_{date}.nc'),
    "cmems": os.path.join(paths.paths['cmems_OI_path'], 'cmems_pdfcorrected_north_SIT_25000_{date}.nc'),
    "CS2": os.path.join(paths.paths['CS2_OI_path'], 'CS2_pdfcorrected_north_SIT_25000_{date}.nc'),
    "rdeft4": os.path.join(paths.paths['rdeft4_OI_path'], 'rdeft4_pdfcorrected_north_SIT_25000_{date}.nc'),
}

# NetCDF文件中变量名称的映射配置
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
    函数功能：
        读取 NetCDF 文件，提取指定变量数据和坐标信息。
    输入参数：
        file_path - NetCDF 文件路径（字符串）。
        var_name  - 待读取的变量名称（字符串）。
    输出：
        返回一个元组 (data, x, y)：
            data - 变量数据数组；
            x    - x 方向坐标数组；
            y    - y 方向坐标数组。
    """
    ds = nc.Dataset(file_path)
    data = ds.variables[var_name][:].data
    x = ds.variables["x"][:]
    y = ds.variables["y"][:]
    ds.close()
    return data, x, y

def preprocess_data(data):
    """
    函数功能：
        对观测数据进行预处理，过滤掉小于 0 的非物理值。
    输入参数：
        data - 原始数据数组。
    输出：
        返回预处理后的数据数组。
    """
    data[data < 0] = 0
    return data

def gaussian_weight(distance, xi, sigma):
    """
    函数功能：
        根据点间欧氏距离、相关长度尺度 xi 以及数据不确定性 sigma 计算高斯权重。
    输入参数：
        distance - 分析点与观测点间的距离（米）。
        xi       - 相关长度尺度（米）。
        sigma    - 数据不确定性（米）。
    输出：
        返回权重值；若距离超过影响半径，则返回 0。
    """
    if distance > R_influence:
        return 0
    return (1 + distance / xi) * np.exp(-distance / xi) / sigma

def fill_missing_values(bg_data, filled_data_sources):
    """
    函数功能：
        补充背景场中的缺失值（NaN）数据，利用其他有效数据源按不确定性优先级进行填充。
    输入参数：
        bg_data - 背景场数据数组（其中可能包含 NaN 表示缺失值）。
        filled_data_sources - 数据补充来源列表，每个元素为 (data, uncertainty)，其中：
                              data - 补充数据数组；
                              uncertainty - 对应的数据不确定性（用于优先级判断）。
    输出：
        返回填充后的背景场数据数组。
    """
    filled_data = np.copy(bg_data)
    ny, nx = filled_data.shape
    for i in range(ny):
        for j in range(nx):
            if np.isnan(filled_data[i, j]):
                candidates = []
                for data, unc in filled_data_sources:
                    if not np.isnan(data[i, j]):
                        candidates.append((data[i, j], unc))
                if candidates:
                    candidates.sort(key=lambda x: x[1])
                    filled_data[i, j] = candidates[0][0]
    return filled_data

def collect_observations(file_paths):
    """
    函数功能：
        收集并整理非背景场的观测数据，提取有效观测点坐标、数据值及不确定性。
    输入参数：
        file_paths - 包含各数据文件路径的字典，键为数据类型，值为对应文件路径。
    输出：
        返回一个元组 (obs_points, obs_values, obs_uncertainties) 其中：
            obs_points         - 有效观测点坐标列表。
            obs_values         - 观测数据值列表。
            obs_uncertainties  - 观测数据不确定性列表。
    """
    obs_points, obs_values, obs_uncertainties = [], [], []
    for key, path in file_paths.items():
        if key == "appx":
            continue  # APP-X为背景场，不参与观测数据收集
        if os.path.exists(path):
            print(f"读取 {key} 数据：{path}")
            data, obs_x, obs_y = read_nc_data(path, var_names[key])
            data = preprocess_data(data)
            sigma = uncertainty_dict.get(key, 1.0)
            mask = ~np.isnan(data)
            # 构造观测点坐标 (x, y)
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
    函数功能：
        利用最优插值方法融合背景场与观测数据，生成最终分析场结果。
    输入参数：
        bg_data           - 背景场数据数组（填充后）。
        x_coords, y_coords - 背景场坐标数组。
        obs_points        - 观测数据点坐标列表。
        obs_values        - 观测数据值列表。
        obs_uncertainties - 观测数据不确定性列表。
    输出：
        返回融合后的分析场数据数组。
    """
    ny, nx = bg_data.shape
    analysis = np.copy(bg_data)
    obs_kd_tree = KDTree(obs_points)
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
    函数功能：
        检查指定目录是否存在，若不存在则创建该目录。
    输入参数：
        directory - 目标目录路径（字符串）。
    输出：
        无返回值，但确保目标目录存在。
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_to_netcdf(out_file, analysis, x_coords, y_coords):
    """
    函数功能：
        将融合后的分析场数据写入 NetCDF 文件，同时自动创建输出目录。
    输入参数：
        out_file - 输出文件的完整路径（字符串）。
        analysis - 分析场数据数组。
        x_coords - x 方向坐标数组。
        y_coords - y 方向坐标数组。
    输出：
        无返回值，但生成 NetCDF 文件保存在指定路径。
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
    函数功能：
        自动扫描指定文件夹，根据文件匹配模式提取文件名中的日期（格式：YYYYMMDD）。
    输入参数：
        pattern - 文件匹配模式（例如 "APPX_north_SIT_25000_*.nc"）。
        path    - 待扫描的目标目录路径（字符串）。
    输出：
        返回提取的日期字符串列表（已排序）。
    """
    dates = set()
    for file in glob(os.path.join(path, pattern)):
        match = re.search(r"\d{8}", file)
        if match:
            dates.add(match.group(0))
    return sorted(dates)

def process_one_day(date_str):
    """
    函数功能：
        对单个日期的数据进行处理：
        1. 读取背景场及其他数据；
        2. 用其他数据按优先级填充背景场缺失值；
        3. 收集观测点，并利用最优插值方法融合数据；
        4. 保存最终融合结果至 NetCDF 文件。
    输入参数：
        date_str - 日期字符串（格式：YYYYMMDD）。
    输出：
        无返回值，但在处理过程中打印日志信息并生成融合输出文件。
    """
    print("==============================================")
    print(f"开始处理日期：{date_str}")

    # 构建当前日期的各数据文件路径字典
    file_paths = {key: template.format(date=date_str) for key, template in file_path_templates.items()}

    # 检查背景场文件
    if not os.path.exists(file_paths["appx"]):
        print(f"背景场文件不存在：{file_paths['appx']}")
        return
    bg_data, x_coords, y_coords = read_nc_data(file_paths["appx"], var_names["appx"])

    # 收集其他数据源用于填充背景场缺失值
    filled_data_sources = []
    for key in ["CS2","cmems"]:
        if os.path.exists(file_paths[key]):
            data, _, _ = read_nc_data(file_paths[key], var_names[key])
            data = preprocess_data(data)
            filled_data_sources.append((data, uncertainty_dict[key]))

    # 用其他数据补充背景场中的缺失值（NaN）
    bg_data = fill_missing_values(bg_data, filled_data_sources)

    # 收集除背景场外的观测数据用于最优插值
    obs_points, obs_values, obs_uncertainties = collect_observations(file_paths)
    analysis = optimal_interpolation(bg_data, x_coords, y_coords, obs_points, obs_values, obs_uncertainties)

    # 保存融合结果至 NetCDF 文件
    out_file = os.path.join(paths.paths['output_OI_path'], f"Arctic_SIT_OI_R25km_{date_str}.nc")
    save_to_netcdf(out_file, analysis, x_coords, y_coords)
    print(f"日期 {date_str} 处理完成。")

def main():
    """
    函数功能：
        主程序，通过自动扫描背景场文件目录中的日期文件，对每个日期依次进行数据处理和融合。
    输入参数：
        无参数（直接从配置和文件目录中获取）。
    输出：
        无返回值，但在处理过程中打印日志信息和保存各日期的融合结果。
    """
    appx_path = paths.paths["appx_OI_path"]
    date_list = auto_discover_dates("APPX_north_SIT_25000_*.nc", appx_path)
    print(f"发现 {len(date_list)} 个日期文件，开始处理...")
    for date_str in date_list:
        process_one_day(date_str)

if __name__ == "__main__":
    main()