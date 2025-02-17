# -*- coding: utf-8 -*-
import os
import xarray as xr
import numpy as np
import pandas as pd
import pyproj
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams['font.family'] = 'SimHei'  

def calculate_relative_error(fusion_data, test_data, small_value_threshold=1.0):
    """
    计算融合数据（fusion）与检验数据(test data)之间的相对误差
    """
    valid_mask = ~np.isnan(fusion_data) & ~np.isnan(test_data)
    fusion_data_valid = fusion_data[valid_mask]
    test_data_valid = test_data[valid_mask]

    if fusion_data_valid.size == 0 or test_data_valid.size == 0:
        return np.nan, np.nan

    small_value_mask = test_data_valid < small_value_threshold
    relative_error = np.full_like(test_data_valid, np.nan)

    relative_error[~small_value_mask] = (
        np.abs(fusion_data_valid[~small_value_mask] - test_data_valid[~small_value_mask])
        / test_data_valid[~small_value_mask]
        * 100
    )
    relative_error[small_value_mask] = np.abs(fusion_data_valid[small_value_mask] - test_data_valid[small_value_mask])

    daily_avg_relative_error = np.nanmean(relative_error)
    percentage_within_10 = (np.sum(relative_error <= 10) / relative_error.size) * 100

    return daily_avg_relative_error, percentage_within_10

def get_projection(hemisphere):
    """
    定义南北极地投影
    """
    if hemisphere == 'north':
        proj_crs = ccrs.NorthPolarStereo(central_longitude=-45)
        extent = [-180, 180, 60, 90]
        proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)
    else:  # South Pole
        proj_crs = ccrs.SouthPolarStereo(central_longitude=0)
        extent = [-180, 180, -90, -60]
        proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=0)
    return proj_crs, extent, proj

def create_region_mask(x_coords_2d, y_coords_2d, proj, region_polygon):
    """
    创建2D掩码，用于区域的点在多边形内（识别14个重点海域）
    """
    lon_grid, lat_grid = proj(x_coords_2d, y_coords_2d, inverse=True)
    poly_path = Path(region_polygon)
    points = np.vstack((lon_grid.flatten(), lat_grid.flatten())).T
    region_mask_flat = poly_path.contains_points(points)
    region_mask = region_mask_flat.reshape(x_coords_2d.shape)
    return region_mask

def plot_spatial_distribution(data, x_coords, y_coords, proj_crs, extent, title, output_path, vmin, vmax, cmap, date_str=None):
    """
    绘制数据的空间分布
    """
    try:
        # 北极投影
        if proj_crs == ccrs.NorthPolarStereo(central_longitude=-45):
            proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)
        else:
            proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=0)

        if isinstance(cmap, str) and cmap == 'relative_error_cmap':  
            colors = ["darkblue", "blue", "deepskyblue", "cyan", "limegreen", "yellow", "red", "firebrick"]
            cmap = ListedColormap(colors)
            bounds = [0, 5, 10, 15, 20, 25, 30, 35, 40]
            norm = BoundaryNorm(bounds, cmap.N)
        else:
            norm = None


        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': proj_crs}, dpi=100)
        ax.set_extent(extent, crs=ccrs.PlateCarree())

        # 掩膜无效数据
        data = data.astype(float)
        data = np.ma.masked_invalid(data)  

        # 转换 grid_x 和 grid_y 为 longitude 和 latitude 使用 pyproj
        lon, lat = proj(x_coords, y_coords, inverse=True)
        im = ax.pcolormesh(
            lon, lat, data,
            transform=ccrs.PlateCarree(), cmap=cmap, shading='auto',  
            norm=norm if norm else None, vmin=None if norm else vmin, vmax=None if norm else vmax  
        )

        # 添加 land, coastlines, 和 gridlines
        ax.add_feature(cfeature.LAND, facecolor='lightgray')
        ax.coastlines(resolution='50m', color='black', linewidth=1)
        
        if proj_crs == ccrs.NorthPolarStereo(central_longitude=-45):
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

            gl.xlabel_style = {'size': 13, 'color': 'black'}
            gl.ylabel_style = {'size': 13, 'color': 'black'}

            # 手动添加经度标签
            ax.text(-45, 62, '45°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=13, color='black')
            ax.text(-135, 63, '135°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=13, color='black')
            ax.text(45, 62, '45°E', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=13, color='black')

        elif proj_crs == ccrs.SouthPolarStereo(central_longitude=0):
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

            # 手动添加经度标签
            ax.text(0, -61, '0°', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
            ax.text(180, -61.5, '180°', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
            ax.text(-90, -62.5, '90°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')
            ax.text(90, -62.5, '90°E', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=10, color='black')

        # 添加 colorbar
        if 'Relative Error' in title:
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.8, ticks=bounds)
            cbar.set_label('相对误差 (%)')
        else:
            cbar = plt.colorbar(im, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('海冰密集度相对误差 (%)')

        # 设置标题
        if date_str:
            # 转换 date_str 为 中文时间形式
            year = date_str[:4]
            if len(date_str) == 8:  # 日时间
                month = date_str[4:6]
                day = date_str[6:]
                chinese_date_str = f"{year}年{month}月{day}日"
            elif len(date_str) == 6:  # 月时间
                month = date_str[4:6]
                chinese_date_str = f"{year}年{month}月"
            elif len(date_str) == 4:  # 年时间
                chinese_date_str = f"{year}年"
            else:
                chinese_date_str = date_str  # 保持原始时间

            if "北极" in title:
                hemisphere_chinese = "北极"
            elif "南极" in title:
                hemisphere_chinese = "南极"
            else:
                hemisphere_chinese = ""

            # 构建完整的中文标题
            if "Yearly Relative Error" in title:
                title_chinese = f"{hemisphere_chinese} - 年度相对误差 - {chinese_date_str}"
            elif "Monthly Relative Error" in title:
                title_chinese = f"{hemisphere_chinese} - 月度相对误差 - {chinese_date_str}"
            elif "Daily Relative Error" in title:
                title_chinese = f"{hemisphere_chinese} - 每日相对误差 - {chinese_date_str}"
            else:
                title_chinese = f"{title} - {chinese_date_str}"  

            ax.set_title(title_chinese, fontsize=15)
        else:
            ax.set_title(title, fontsize=15)

        # 保存图像
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        # print(f"Plot saved: {output_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Error generating plot: {e}")
        
def plot_comparison(data1, data2, x_coords, y_coords, proj_crs, extent, title, output_path, vmin, vmax, cmap, date_str=None):
    """
    绘制两个数据集的比较图，包含两个子图，左侧为融合数据，右侧为检验数据
    """
    try:
        if proj_crs == ccrs.NorthPolarStereo(central_longitude=-45):
            proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=-45)
        else:
            proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=0)

        colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
        cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)

        fig, axs = plt.subplots(1, 2, figsize=(12, 6), subplot_kw={'projection': proj_crs})
        for i, (data, ax) in enumerate(zip([data1, data2], axs)):
            ax.set_extent(extent, crs=ccrs.PlateCarree())
            
            data = data.astype(float)
            data = np.ma.masked_invalid(data)

            lon, lat = proj(x_coords, y_coords, inverse=True)

            # PlateCarree 用于绘制经纬度数据
            im = ax.pcolormesh(
                lon, lat, data,
                transform=ccrs.PlateCarree(), cmap=cmap, shading='auto', vmin=vmin, vmax=vmax
            )

            # 添加 land, coastlines, 和 gridlines
            ax.add_feature(cfeature.LAND, facecolor='lightgray')
            ax.coastlines(resolution='50m', color='black', linewidth=1)
            if proj_crs == ccrs.NorthPolarStereo(central_longitude=-45):
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                gl.ylocator = mticker.FixedLocator([60, 70, 80])
                gl.top_labels = False
                gl.right_labels = False
                gl.bottom_labels = False  

                # 绘制左侧纬度标签
                gl.ylabels_left = True
                gl.xlabels_top = False
                gl.xlabels_bottom = False
                gl.yformatter = LATITUDE_FORMATTER

                gl.xlabel_style = {'size': 12, 'color': 'black'}
                gl.ylabel_style = {'size': 12, 'color': 'black'}

                # 手动添加经度标签
                ax.text(-45, 62, '45°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=12, color='black')
                ax.text(-135, 64.5, '135°W', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=12, color='black')
                ax.text(45, 62.9, '45°E', transform=ccrs.PlateCarree(), ha='center', va='top', rotation=0, fontsize=12, color='black')


            elif proj_crs == ccrs.SouthPolarStereo(central_longitude=0):
                gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
                gl.xlocator = mticker.FixedLocator([-180, -135, -90, -45, 0, 45, 90, 135, 180])
                gl.ylocator = mticker.FixedLocator([-80,-70,-60])
                gl.top_labels = False
                gl.right_labels = False
                gl.bottom_labels = False 

                # 绘制左侧 latitude 标签
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

            cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.05, shrink=0.8)
            cbar.set_label('海冰密集度 (%)')

            ax.set_title(f"{'融合数据' if i == 0 else '检验数据'}")

        # 设置标题
        if date_str:
          year = date_str[:4]
          month = date_str[4:6]
          day = date_str[6:]
          chinese_date_str = f"{year}年{month}月{day}日"

          if "北极" in title:
            hemisphere_chinese = "北极"
          elif "南极" in title:
            hemisphere_chinese = "南极"
          else:
            hemisphere_chinese = ""
          
          if "Yearly" in title:
              title_chinese = f"{hemisphere_chinese} - 年度数据 - {year}年"
          elif "Monthly" in title:
              title_chinese = f"{hemisphere_chinese} - 月度数据 - {year}年{month}月"
          elif "Daily" in title:
              title_chinese = f"{hemisphere_chinese} - 每日数据 - {chinese_date_str}"
          else:
              title_chinese = title  
          
          plt.suptitle(title_chinese, fontsize=15)
        else:
            plt.suptitle(title, fontsize=15)  

        # 保存图像
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close(fig)
        # print(f"Comparison plot saved: {output_path}")

    except Exception as e:
        print(f"Error generating comparison plot: {e}")

def process_files(fusion_path, test_path, var_fusion, var_test, hemisphere, sea_regions, results_base_path):
    """
    处理文件以计算相对误差并绘制空间分布
    """
    fusion_files = sorted([f for f in os.listdir(fusion_path) if f.endswith('.nc')])
    test_files = sorted([f for f in os.listdir(test_path) if f.endswith('.nc')])

    results = []
    proj_crs, extent, proj = get_projection(hemisphere)

    for fusion_file, test_file in zip(fusion_files, test_files):
        if fusion_file[-11:-3] != test_file[-11:-3]:
            # print(f"Skipping unmatched files: {fusion_file}, {test_file}")
            continue

        fusion_ds = xr.open_dataset(os.path.join(fusion_path, fusion_file))
        test_ds = xr.open_dataset(os.path.join(test_path, test_file))

        try:
            fusion_data = fusion_ds[var_fusion].values
            test_data = test_ds[var_test].values

            if fusion_data.ndim == 2:
                fusion_data = fusion_data[np.newaxis, :, :]
            if test_data.ndim == 2:
                test_data = test_data[np.newaxis, :, :]

            x_coords = fusion_ds['x'].values
            y_coords = fusion_ds['y'].values

            if x_coords.ndim == 1:
                x_coords_2d, y_coords_2d = np.meshgrid(x_coords, y_coords)
            else:
                x_coords_2d, y_coords_2d = x_coords, y_coords
        except KeyError as e:
            print(f"Variable or coordinate not found in datasets: {e}")
            fusion_ds.close()
            test_ds.close()
            continue

        date_str = fusion_file[-11:-3]
        date = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
        date = pd.to_datetime(date)

        # 计算整个区域的相对误差
        relative_error = np.full_like(fusion_data, np.nan)
        for t in range(fusion_data.shape[0]):
            relative_error[t] = (np.abs(fusion_data[t] - test_data[t]) / np.where(test_data[t] >= 1.0, test_data[t], np.nan)) * 100
            low_value_mask = test_data[t] < 1.0
            relative_error[t][low_value_mask] = np.abs(fusion_data[t][low_value_mask] - test_data[t][low_value_mask])

        daily_avg_relative_error = np.nanmean(relative_error)
        percentage_within_10 = np.nansum(relative_error <= 10) / np.isfinite(relative_error).sum() * 100

        hemisphere_chinese = "北极" if hemisphere == 'north' else "南极"
        results_entry = {'Date': date,
                         f'{hemisphere_chinese}_Relative_Error': daily_avg_relative_error,
                         f'{hemisphere_chinese}_Percentage_within_10': percentage_within_10}
        
        colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
        seaice_cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)

        # 绘制每日相对误差分布图
        output_dir_daily = os.path.join(results_base_path, hemisphere, 'spatial_distribution_daily')
        os.makedirs(output_dir_daily, exist_ok=True)
        output_path_daily = os.path.join(output_dir_daily, f"{hemisphere_chinese}_日相对误差_{date_str}.png")
        plot_spatial_distribution(relative_error[0], x_coords_2d, y_coords_2d, proj_crs, extent,
                                 f'{hemisphere_chinese} - 每日相对误差', output_path_daily, 0, 100, 'relative_error_cmap', date_str)

        # 绘制融合数据与检验数据对比图
        output_dir_comparison = os.path.join(results_base_path, hemisphere, 'comparison_daily')
        os.makedirs(output_dir_comparison, exist_ok=True)
        output_path_comparison = os.path.join(output_dir_comparison, f"{hemisphere_chinese}_融合数据与检验数据对比_{date_str}.png")
        plot_comparison(fusion_data[0], test_data[0], x_coords_2d, y_coords_2d, proj_crs, extent,
                        f'{hemisphere_chinese} - {date_str}', output_path_comparison, 0, 100, seaice_cmap, date_str)

        # 计算月度和年度平均相对误差
        if date.day == 1:
            month_str = date.strftime('%Y%m')
            monthly_fusion_data = fusion_data.copy()
            monthly_test_data = test_data.copy()

            for next_day in range(1, 32):
                try:
                    next_date = date + pd.Timedelta(days=next_day)
                    if next_date.month != date.month:
                        break
                    next_date_str = next_date.strftime('%Y%m%d')
                    next_fusion_file = [f for f in fusion_files if f.endswith(f"{next_date_str}.nc")][0]
                    next_test_file = [f for f in test_files if f.endswith(f"{next_date_str}.nc")][0]

                    next_fusion_ds = xr.open_dataset(os.path.join(fusion_path, next_fusion_file))
                    next_test_ds = xr.open_dataset(os.path.join(test_path, next_test_file))

                    next_fusion_data = next_fusion_ds[var_fusion].values
                    next_test_data = next_test_ds[var_test].values

                    if next_fusion_data.ndim == 2:
                        next_fusion_data = next_fusion_data[np.newaxis, :, :]
                    if next_test_data.ndim == 2:
                        next_test_data = next_test_data[np.newaxis, :, :]

                    monthly_fusion_data = np.concatenate((monthly_fusion_data, next_fusion_data), axis=0)
                    monthly_test_data = np.concatenate((monthly_test_data, next_test_data), axis=0)

                    next_fusion_ds.close()
                    next_test_ds.close()
                except (FileNotFoundError, IndexError):
                    print(f"Skipping missing file for {next_date_str}")
                    break

            # 如果月度数据全为 NaN，则跳过绘图
            if np.all(np.isnan(monthly_fusion_data)) or np.all(np.isnan(monthly_test_data)):
                print(f"Skipping plots for {month_str} due to all-NaN data.")
            else:
                # 计算月度平均相对误差
                monthly_fusion_avg = np.nanmean(monthly_fusion_data, axis=0)
                monthly_test_avg = np.nanmean(monthly_test_data, axis=0)
                monthly_relative_error = (np.abs(monthly_fusion_avg - monthly_test_avg) / np.where(monthly_test_avg >= 1.0, monthly_test_avg, np.nan)) * 100
                low_value_mask = monthly_test_avg < 1.0
                monthly_relative_error[low_value_mask] = np.abs(monthly_fusion_avg[low_value_mask] - monthly_test_avg[low_value_mask])

                # 绘制月度相对误差分布图
                output_dir_monthly = os.path.join(results_base_path, hemisphere, 'spatial_distribution_monthly')
                os.makedirs(output_dir_monthly, exist_ok=True)
                output_path_monthly = os.path.join(output_dir_monthly, f"{hemisphere_chinese}_月相对误差_{month_str}.png")
                plot_spatial_distribution(monthly_relative_error, x_coords_2d, y_coords_2d, proj_crs, extent,
                                         f'{hemisphere_chinese} - 月度相对误差', output_path_monthly, 0, 100, 'relative_error_cmap', month_str)

                # 绘制融合数据与检验数据对比图
                output_dir_comparison_monthly = os.path.join(results_base_path, hemisphere, 'comparison_monthly')
                os.makedirs(output_dir_comparison_monthly, exist_ok=True)
                output_path_comparison_monthly = os.path.join(output_dir_comparison_monthly, f"{hemisphere_chinese}_融合数据与检验数据对比_{month_str}.png")
                plot_comparison(monthly_fusion_avg, monthly_test_avg, x_coords_2d, y_coords_2d, proj_crs, extent,
                                f'{hemisphere_chinese} - {month_str}', output_path_comparison_monthly, 0, 100, seaice_cmap, month_str)

                if date.month == 1:
                    year_str = date.strftime('%Y')
                    yearly_fusion_data = monthly_fusion_data.copy()
                    yearly_test_data = monthly_test_data.copy()

                    for next_month in range(1, 12):
                        try:
                            next_month_date = date + pd.DateOffset(months=next_month)
                            if next_month_date.year != date.year:
                                break

                            for day in range(1, 32):
                                try:
                                    next_date = next_month_date + pd.Timedelta(days=day-1)
                                    if next_date.month != next_month_date.month:
                                        break
                                    next_date_str = next_date.strftime('%Y%m%d')

                                    next_fusion_file = [f for f in fusion_files if f.endswith(f"{next_date_str}.nc")][0]
                                    next_test_file = [f for f in test_files if f.endswith(f"{next_date_str}.nc")][0]

                                    next_fusion_ds = xr.open_dataset(os.path.join(fusion_path, next_fusion_file))
                                    next_test_ds = xr.open_dataset(os.path.join(test_path, next_test_file))

                                    next_fusion_data = next_fusion_ds[var_fusion].values
                                    next_test_data = next_test_ds[var_test].values

                                    if next_fusion_data.ndim == 2:
                                        next_fusion_data = next_fusion_data[np.newaxis, :, :]
                                    if next_test_data.ndim == 2:
                                        next_test_data = next_test_data[np.newaxis, :, :]

                                    yearly_fusion_data = np.concatenate((yearly_fusion_data, next_fusion_data), axis=0)
                                    yearly_test_data = np.concatenate((yearly_test_data, next_test_data), axis=0)

                                    next_fusion_ds.close()
                                    next_test_ds.close()
                                except (FileNotFoundError, IndexError):
                                    print(f"Skipping missing file for {next_date_str}")
                                    break
                        except (FileNotFoundError, IndexError):
                            print(f"Skipping missing files for month {next_month_date.strftime('%Y%m')}")
                            break

                    # 计算年度平均相对误差
                    yearly_fusion_avg = np.nanmean(yearly_fusion_data, axis=0)
                    yearly_test_avg = np.nanmean(yearly_test_data, axis=0)

                    yearly_relative_error = (np.abs(yearly_fusion_avg - yearly_test_avg) / np.where(yearly_test_avg >= 1.0, yearly_test_avg, np.nan)) * 100
                    low_value_mask = yearly_test_avg < 1.0
                    yearly_relative_error[low_value_mask] = np.abs(yearly_fusion_avg[low_value_mask] - yearly_test_avg[low_value_mask])

                    # 绘制年度相对误差分布图
                    output_dir_yearly = os.path.join(results_base_path, hemisphere, 'spatial_distribution_yearly')
                    os.makedirs(output_dir_yearly, exist_ok=True)
                    output_path_yearly = os.path.join(output_dir_yearly, f"{hemisphere_chinese}_年相对误差_{year_str}.png")
                    plot_spatial_distribution(yearly_relative_error, x_coords_2d, y_coords_2d, proj_crs, extent,
                                             f'{hemisphere_chinese} - 年度相对误差', output_path_yearly, 0, 100, 'relative_error_cmap', year_str)

                    # 绘制融合数据与检验数据对比图
                    output_dir_comparison_yearly = os.path.join(results_base_path, hemisphere, 'comparison_yearly')
                    os.makedirs(output_dir_comparison_yearly, exist_ok=True)
                    output_path_comparison_yearly = os.path.join(output_dir_comparison_yearly, f"{hemisphere_chinese}_融合数据与检验数据对比_{year_str}.png")
                    plot_comparison(yearly_fusion_avg, yearly_test_avg, x_coords_2d, y_coords_2d, proj_crs, extent,
                                    f'{hemisphere_chinese} - {year_str}', output_path_comparison_yearly, 0, 100, seaice_cmap, year_str)
        # 计算重点海域的相对误差
        for region_name, region_polygon in sea_regions.items():
            if hemisphere.capitalize() not in region_name:
                continue

            region_mask = create_region_mask(x_coords_2d, y_coords_2d, proj, region_polygon)
            region_mask_3d = region_mask[np.newaxis, :, :]

            fusion_data_region = np.where(region_mask_3d, fusion_data, np.nan)
            test_data_region = np.where(region_mask_3d, test_data, np.nan)

            region_avg_error, region_percentage_within_10 = calculate_relative_error(
                fusion_data_region, test_data_region
            )

            results_entry[f'{region_name}_Relative_Error'] = region_avg_error
            results_entry[f'{region_name}_Percentage_within_10'] = region_percentage_within_10

        results.append(results_entry)

        fusion_ds.close()
        test_ds.close()

    return results

def compute_yearly_monthly_averages(results_df, region_prefix):
    """
    计算14个区域的年度和月度平均相对误差
    """
    results_df['Year'] = results_df['Date'].dt.year
    results_df['Month'] = results_df['Date'].dt.month

    yearly_avg = results_df.groupby('Year')[f'{region_prefix}_Relative_Error'].mean()
    monthly_avg = results_df.groupby('Month')[f'{region_prefix}_Relative_Error'].mean()

    return yearly_avg, monthly_avg

def main():
    # 定义14个重点海域（纬度和经度多边形）
    sea_regions = {
        'Barents Sea North': [(-45, 65), (-45, 80), (60, 80), (60, 65)],
        'Kara Sea North': [(60, 70), (60, 80), (90, 80), (90, 70)],
        'Laptev Sea North': [(100, 75), (100, 80), (140, 80), (140, 75)],
        'East Siberian Sea North': [(140, 75), (140, 80), (180, 80), (180, 75)],
        'Chukchi Sea North': [(-180, 70), (-180, 80), (-140, 80), (-140, 70)],
        'Beaufort Sea North': [(-140, 70), (-140, 80), (-100, 80), (-100, 70)],
        'Canadian Archipelago North': [(-100, 75), (-100, 80), (-60, 80), (-60, 75)],
        'Central Arctic North': [(-45, 80), (135, 80), (180, 90), (-180, 90), (-45, 90)],
        'Weddell Sea South': [(-60, -75), (-60, -60), (0, -60), (0, -75)],
        'Indian Ocean South': [(0, -70), (0, -60), (90, -60), (90, -70)],
        'Pacific Ocean South': [(90, -70), (90, -60), (180, -60), (180, -70)],
        'Ross Sea South': [(180, -80), (180, -65), (-150, -65), (-150, -80)],
        'Bellingshausen Sea South': [(-100, -75), (-100, -65), (-60, -65), (-60, -75)],
        'Amundsen Sea South': [(-150, -75), (-150, -65), (-100, -65), (-100, -75)]
    }

    # 定义路径和变量名称
    fusion_base_path = "D:\\zmh\\icecon12500_data\\ICECON_merge_12500"
    results_base_path = "D:\\zmh\\icecon12500_data\\ICECON_merge_12500\\error_results"
    os.makedirs(results_base_path, exist_ok=True)
    
    test_base_paths = {
        'north': "D:\\zmh\\icecon12500_data\\OISST\\0_pre_com\\north",
        'south': "D:\\zmh\\icecon12500_data\\UB-AMSR2\\0_pre_com\\south"
    }
    hemispheres = ['north', 'south']
    fusion_var_names = {
        'north': 'ice_con',
        'south': 'ice_con'
    }
    test_var_names = {
        'north': 'OISST_north_icecon',
        'south': 'UB-AMSR2_south_icecon'
    }

    # 遍历两个半球
    for hemisphere in hemispheres:
        print(f"Processing {hemisphere.capitalize()} Hemisphere...")

        fusion_path = os.path.join(fusion_base_path, hemisphere)
        test_path = test_base_paths[hemisphere]

        results = process_files(
            fusion_path=fusion_path,
            test_path=test_path,
            var_fusion=fusion_var_names[hemisphere],
            var_test=test_var_names[hemisphere],
            hemisphere=hemisphere,
            sea_regions=sea_regions,
            results_base_path=results_base_path
        )

        results_df = pd.DataFrame(results)

        if results_df.empty:
            print(f"No valid results were computed for {hemisphere}. Skipping.")
            continue

        # 计算并保存误差
        # print(results_df.groupby(results_df['Date'].dt.year))
        hemisphere_chinese = "北极" if hemisphere == 'north' else "南极"
        hemisphere_yearly_avg = results_df.groupby(results_df['Date'].dt.year)[f'{hemisphere_chinese}_Relative_Error'].mean()
        yearly_csv_path = os.path.join(results_base_path, f"{hemisphere_chinese}_年度平均相对误差_12500.csv")
        hemisphere_yearly_avg.to_csv(yearly_csv_path, header=[f"{hemisphere_chinese}_年度平均相对误差"],encoding='utf_8_sig')
        # print(f"Yearly average relative error for {hemisphere_chinese} saved to {yearly_csv_path}")

        hemisphere_monthly_avg = results_df.groupby(results_df['Date'].dt.month)[f'{hemisphere_chinese}_Relative_Error'].mean()
        monthly_csv_path = os.path.join(results_base_path, f"{hemisphere_chinese}_月度平均相对误差_12500.csv")
        hemisphere_monthly_avg.to_csv(monthly_csv_path, header=[f"{hemisphere_chinese}_月度平均相对误差"],encoding='utf_8_sig')
        # print(f"Monthly average relative error for {hemisphere_chinese} saved to {monthly_csv_path}")

        hemisphere_overall_avg = results_df[f'{hemisphere_chinese}_Relative_Error'].mean()
        overall_avg_path = os.path.join(results_base_path, f"{hemisphere_chinese}_总平均相对误差_12500.txt")
        with open(overall_avg_path, 'w', encoding='utf-8') as f:
            f.write(f"{hemisphere_chinese}总平均相对误差: {hemisphere_overall_avg:.2f}%")
        # print(f"Overall average relative error for {hemisphere_chinese} saved to {overall_avg_path}")

        overall_csv_path = os.path.join(results_base_path, f"{hemisphere_chinese}_每日相对误差_12500.csv")
        results_df.to_csv(overall_csv_path, index=False, encoding='utf_8_sig')
        # print(f"Overall results for {hemisphere_chinese} saved to {overall_csv_path}")

        # 计算并保存14个重点海域的误差
        for region_name in sea_regions.keys():
            if hemisphere.capitalize() not in region_name:
                continue

            yearly_avg, monthly_avg = compute_yearly_monthly_averages(results_df, region_name)
            region_dir = os.path.join(results_base_path, f"{region_name}_results")
            os.makedirs(region_dir, exist_ok=True)
            region_name_chinese = region_name.replace(" ", "_").replace("North", "北极").replace("South", "南极").replace("Sea","海").replace("Ocean","洋").replace("Central_Arctic", "中央北冰洋").replace("Canadian_Archipelago", "加拿大群岛").replace("Barents", "巴伦支").replace("Kara", "卡拉").replace("Laptev", "拉普捷夫").replace("East_Siberian", "东西伯利亚").replace("Chukchi", "楚科奇").replace("Beaufort", "波弗特").replace("Weddell", "威德尔").replace("Indian", "印度").replace("Pacific", "太平洋").replace("Ross", "罗斯").replace("Bellingshausen", "别林斯高晋").replace("Amundsen", "阿蒙森")

            yearly_csv_path = os.path.join(region_dir, f"{region_name_chinese}_年度平均相对误差_12500.csv")
            yearly_avg.to_csv(yearly_csv_path, header=[f"{region_name_chinese}_年度平均误差"], encoding='utf-8-sig')
            # print(f"Yearly averages for {region_name_chinese} saved to {yearly_csv_path}")

            monthly_csv_path = os.path.join(region_dir, f"{region_name_chinese}_月度平均相对误差_12500.csv")
            monthly_avg.to_csv(monthly_csv_path, header=[f"{region_name_chinese}_月度平均误差"], encoding='utf-8-sig')
            # print(f"Monthly averages for {region_name_chinese} saved to {monthly_csv_path}")

            overall_avg = results_df[f'{region_name}_Relative_Error'].mean()
            overall_avg_path = os.path.join(region_dir, f"{region_name_chinese}_总平均相对误差_12500.txt")
            with open(overall_avg_path, 'w', encoding='utf-8') as f:
                f.write(f"{region_name_chinese}总平均相对误差: {overall_avg:.2f}%")
            # print(f"Overall average for {region_name_chinese} saved to {overall_avg_path}")

if __name__ == "__main__":
    main()