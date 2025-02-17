import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from RegularGrid import north_regular_grid
from SITsnapshot import generate_SIT_snapshot
from paths import paths
import cartopy.crs as ccrs
import matplotlib.pyplot as plt

# 直接使用APP-X文件夹路径，不调用 xr.open_dataset
appx_base_folder_path = paths['appx_base_folder_path']  # APP-X文件夹路径
output_folder_north = os.path.join(appx_base_folder_path, '0_pre_com', 'north')
failure_folder = os.path.join(appx_base_folder_path, '0_pre_com', 'failures')
snapshot_folder = os.path.join(appx_base_folder_path, "0_pre_com", "snapshots")

for folder in [output_folder_north, failure_folder, snapshot_folder]:
    os.makedirs(folder, exist_ok=True)

# 定义北半球CRS
crs_wgs84 = CRS.from_epsg(4326)  # WGS84 (lat, lon)
crs_psn = CRS.from_epsg(3413)    # 北极投影 (x, y)

grid_x_n, grid_y_n = north_regular_grid()  # 北极规则网格

def process_north():
    """
    处理北半球的海冰厚度数据：
    对同一天内的0400与1400观测数据先求平均（即海冰厚度数据的平均），
    然后进行区域裁剪（经纬度>=60）、投影转换、插值到规则网格、快视图生成并保存结果。
    """
    total_days = 0       # 总共处理的天数
    successful_days = 0  # 成功处理的天数
    failed_days = 0      # 处理失败的天数

    # 遍历每个年份的数据
    for year in range(2004, 2024):  # 遍历年份
        year_folder_path = os.path.join(appx_base_folder_path, str(year))  # 年份文件夹路径
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue

        # 按日期分组，每天包含0400与1400数据
        daily_files = {}  # 格式：{ "20220102": { "0400": filepath, "1400": filepath } }
        for filename in os.listdir(year_folder_path):
            if filename.endswith('.nc'):
                parts = filename.split('_')
                # 检查文件名格式是否正确（至少包含6部分）
                if len(parts) < 6:
                    print(f"Filename {filename} 不符合预期格式，跳过。")
                    continue
                time_indicator = parts[3]  # 例如 0400 或 1400
                date_str = parts[4][1:]    # 去掉前面的字母"d", 如 "20220102"
                if date_str not in daily_files:
                    daily_files[date_str] = {}
                daily_files[date_str][time_indicator] = os.path.join(year_folder_path, filename)

        # 对每天的数据进行处理
        with tqdm(total=len(daily_files), desc=f"Processing {year} daily data") as pbar:
            for date_str, files in daily_files.items():
                total_days += 1
                # 判断当天是否同时存在0400与1400的文件
                if "0400" in files and "1400" in files:
                    try:
                        # 分别读取0400和1400数据
                        ds_0400 = xr.open_dataset(files["0400"])
                        ds_1400 = xr.open_dataset(files["1400"])
                        
                        # 提取所需变量，并求平均 (只需海冰厚度，假设经纬度信息一致)
                        sit_0400 = ds_0400[['cdr_sea_ice_thickness', 'longitude', 'latitude']].drop('time')['cdr_sea_ice_thickness'].squeeze()
                        sit_1400 = ds_1400[['cdr_sea_ice_thickness', 'longitude', 'latitude']].drop('time')['cdr_sea_ice_thickness'].squeeze()
                        averaged_sit = (sit_0400 + sit_1400) / 2.0
                        
                        # 使用0400文件的经纬度信息
                        lon = ds_0400['longitude']
                        lat = ds_0400['latitude']
                        
                        # 构建新的Dataset
                        ds_avg = xr.Dataset({
                            'cdr_sea_ice_thickness': averaged_sit,
                            'longitude': lon,
                            'latitude': lat
                        })

                        # 进行区域裁剪（例如只保留纬度>=60的区域）
                        mask = ds_avg['latitude'] >= 60
                        ds_avg = ds_avg.where(mask, drop=True)
                        
                        # 投影转换：先获取经纬度的值
                        transformer = Transformer.from_crs(crs_wgs84, crs_psn, always_xy=True)
                        x, y = transformer.transform(ds_avg['longitude'].values, ds_avg['latitude'].values)
                        
                        # 插值到规则网格
                        sit_interpolated = griddata(
                            (x.flatten(), y.flatten()),
                            ds_avg['cdr_sea_ice_thickness'].values.flatten(),
                            (grid_x_n, grid_y_n)
                        )
                        
                        # 构造DataArray，并生成新的Dataset
                        da = xr.DataArray(
                            sit_interpolated,
                            coords={'y': grid_y_n[:, 0], 'x': grid_x_n[0, :]},
                            dims=['y', 'x']
                        )
                        da_dataset = da.to_dataset(name='appx_sit')
                        
                        # 创建输出文件名，此处日期为 daily date
                        output_filename = os.path.join(
                            output_folder_north,
                            f'APPX_north_SIT_25000_{date_str}.nc'
                        )
                        
                        # 生成快视图
                        generate_SIT_snapshot(da_dataset, grid_x_n, grid_y_n, output_filename, 'appx_sit', snapshot_folder)
                        
                        # 保存为NetCDF文件
                        da_dataset.to_netcdf(output_filename)
                        successful_days += 1

                    except Exception as e:
                        failed_days += 1
                        # 将当天的两个原始文件复制到失败文件夹中
                        for f in files.values():
                            failure_filename = os.path.join(failure_folder, os.path.basename(f))
                            shutil.copy(f, failure_filename)
                        print(f"Error processing date {date_str} in {year_folder_path}: {e}")
                else:
                    failed_days += 1
                    print(f"缺少0400或1400数据：{date_str}，跳过该日期。")
                    
                pbar.update(1)

    # 输出处理结果
    print(f"Total days processed: {total_days}")
    print(f"Successful days: {successful_days}")
    print(f"Failed days: {failed_days}")

if __name__ == '__main__':
    process_north()
    print("Success!")