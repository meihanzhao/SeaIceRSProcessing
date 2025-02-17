# -*- coding: utf-8 -*-
"""
海冰厚度最优插值数据融合 - 升级版（多日期处理）
作者：ZMH
日期：2025-02-17

说明：
    本代码实现基于背景场（APP-X）和观测数据（CMEMS、CS2SMOS、NSIDC‑rdeft4）的最优插值融合，
    并通过循环方式处理多个日期的文件。文件路径采用模板格式，其中日期部分由 YYYYMMDD 替换得到。
    核心特点：
    1. 针对观测数据的负值进行过滤，确保输入数据合理性。
    2. 主函数模块化，方便循环处理，逻辑清晰、易于扩展。
    3. 增强中文注释，便于理解和调试。
"""

# -*- coding: utf-8 -*-
"""
海冰厚度最优插值数据融合 - 自动日期处理版
作者：ZMH
日期：2025-02-17

增强：
1. 自动扫描文件夹以获取日期文件，提升效率；
2. 处理大量文件时支持错误捕获和日志记录；
3. 集中路径管理和灵活的配置。
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

# 影响半径（米）和相关长度尺度 ξ（米，不确定性）
R_influence = 250000.0  # 250 km
xi = 100000.0  # 100 km

# 数据不确定性
uncertainty_dict = {
    "appx": 0.6,    # APP-X 模式背景场
    "cmems": 0.7,   # CMEMS 数据
    "CS2": 0.5,     # CS2SMOS 数据
    "rdeft4": 0.6,  # NSIDC-rdeft4 融合数据
}

# 路径模板配置，使用 paths 模块
file_path_templates = {
    "appx":  os.path.join(paths.paths['appx_OI_path'], 'APPX_north_SIT_25000_{date}.nc'),
    "cmems": os.path.join(paths.paths['cmems_OI_path'], 'cmems_pdfcorrected_north_SIT_25000_{date}.nc'),
    "CS2":   os.path.join(paths.paths['CS2_OI_path'], 'CS2_pdfcorrected_north_SIT_25000_{date}.nc'),
    "rdeft4": os.path.join(paths.paths['rdeft4_OI_path'], 'rdeft4_pdfcorrected_north_SIT_25000_{date}.nc'),
}

# 变量配置
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
    """读取 NetCDF 文件，并返回指定变量数据和坐标"""
    ds = nc.Dataset(file_path)
    data = ds.variables[var_name][:].data  # 读取变量数据
    x = ds.variables["x"][:]  # 获取 x 坐标
    y = ds.variables["y"][:]  # 获取 y 坐标
    ds.close()
    return data, x, y


def preprocess_data(data):
    """将小于 0 的值置为 0，过滤非物理值"""
    data[data < 0] = 0
    return data


def gaussian_weight(distance, xi, sigma):
    """根据距离和不确定性 σ 计算权重"""
    if distance > R_influence:
        return 0
    return (1 + distance / xi) * np.exp(-distance / xi) / sigma


def collect_observations(file_paths):
    """收集观测数据，包括有效值的位置和对应的不确定性"""
    obs_points, obs_values, obs_uncertainties = [], [], []

    for key, path in file_paths.items():
        if key == "appx":  # APP-X 背景场，不收集为观测点
            continue
        if os.path.exists(path):
            print(f"读取 {key} 数据：{path}")
            data, obs_x, obs_y = read_nc_data(path, var_names[key])
            data = preprocess_data(data)  # 移除负值
            sigma = uncertainty_dict.get(key, 1.0)
            # 获取有效点并收集
            mask = ~np.isnan(data)
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
    """最优插值融合"""
    ny, nx = bg_data.shape
    analysis = np.copy(bg_data)
    obs_kd_tree = KDTree(obs_points)  # 建立 KDTree

    for i in range(ny):
        for j in range(nx):
            x0, y0 = x_coords[j], y_coords[i]
            bg_val = bg_data[i, j]

            indices = obs_kd_tree.query_ball_point([x0, y0], R_influence)
            if len(indices) == 0:  # 无观测点，保留背景值
                analysis[i, j] = bg_val
                continue

            numer, denom = 0.0, 0.0
            for idx in indices:
                obs_x, obs_y = obs_points[idx]
                obs_val = obs_values[idx]
                sigma = obs_uncertainties[idx]
                distance = np.sqrt((x0 - obs_x) ** 2 + (y0 - obs_y) ** 2)
                weight = gaussian_weight(distance, xi, sigma)
                if weight > 0:
                    numer += weight * (obs_val - bg_val)
                    denom += weight

            analysis[i, j] = bg_val + numer / denom if denom > 0 else bg_val

    return analysis

def ensure_dir_exists(directory):
    """
    检查目录是否存在，如不存在则创建目录
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_to_netcdf(out_file, analysis, x_coords, y_coords):
    """保存为 NetCDF 文件"""
    
    # 获取输出目录，并确保目录存在
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
    print(f"融合结果已保存：{out_file}")


def auto_discover_dates(pattern, path):
    """自动扫描文件夹匹配日期文件并提取日期"""
    dates = set()
    for file in glob(os.path.join(path, pattern)):
        match = re.search(r"\d{8}", file)
        if match:
            dates.add(match.group(0))
    return sorted(dates)


def process_one_day(date_str):
    """单日期处理函数"""
    print(f"==============================================")
    print(f"开始处理日期：{date_str}")

    file_paths = {key: template.format(date=date_str) for key, template in file_path_templates.items()}

    # 读取背景场
    if not os.path.exists(file_paths["appx"]):
        print(f"背景场文件不存在：{file_paths['appx']}")
        return
    bg_data, x_coords, y_coords = read_nc_data(file_paths["appx"], var_names["appx"])

    # 收集观测点
    obs_points, obs_values, obs_uncertainties = collect_observations(file_paths)

    # 最优插值处理
    analysis = optimal_interpolation(bg_data, x_coords, y_coords, obs_points, obs_values, obs_uncertainties)

    # 保存结果
    out_file = os.path.join(paths.paths['output_OI_path'], f"Arctic_SIT_OI_R25km_{date_str}.nc")
    save_to_netcdf(out_file, analysis, x_coords, y_coords)


# ------------------------------
# 主函数
# ------------------------------
def main():
    # 自动发现所有可用日期
    appx_path = paths.paths["appx_OI_path"]
    date_list = auto_discover_dates("APPX_north_SIT_25000_*.nc", appx_path)

    print(f"发现 {len(date_list)} 个日期文件，开始处理...")
    for date_str in date_list:
        process_one_day(date_str)

if __name__ == "__main__":
    main()