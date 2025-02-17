# -*- coding: utf-8 -*-
import numpy as np
import xarray as xr
from pyproj import CRS, Transformer, Proj
from pyhdf.SD import SD, SDC
from scipy.interpolate import griddata
from tqdm import tqdm
import shutil
import os
from osgeo import gdal
from RegularGrid import north_regular_grid, south_regular_grid, generate_snapshot

# NISE 数据集的基础路径
base_folder_path_nise = r"D:\zmh\icecon25000_data\NISE"
output_base_folder = os.path.join(base_folder_path_nise, "0_pre_com")
snapshot_folder = os.path.join(output_base_folder, "snapshots")
os.makedirs(output_base_folder, exist_ok=True)
os.makedirs(snapshot_folder, exist_ok=True)

crs_wgs84 = CRS.from_epsg(4326)


def process_nise_hemisphere(hemisphere):
    if hemisphere == 'north':
        grid_x_hem, grid_y_hem = north_regular_grid()  # 北极网格
        output_folder = os.path.join(output_base_folder, "north")
        failure_folder = os.path.join(output_base_folder, "failures")
        crs_ps = CRS.from_epsg(3413)
        proj_epsg = Proj("epsg:3408")
    elif hemisphere == 'south':
        grid_x_hem, grid_y_hem = south_regular_grid()  # 南极网格
        output_folder = os.path.join(output_base_folder, "south")
        failure_folder = os.path.join(output_base_folder, "failures")
        crs_ps = CRS.from_epsg(3976) 
        proj_epsg = Proj("epsg:3409")
    else:
        raise ValueError("Invalid hemisphere specified. Choose 'north' or 'south'.")

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(failure_folder, exist_ok=True)

    total_files = 0
    successful_files = 0
    failed_files = 0

    # 遍历文件
    all_files = []
    for year in range(2022, 2024):
        year_folder_path = os.path.join(base_folder_path_nise, hemisphere, str(year))
        if not os.path.exists(year_folder_path):
            print(f"Year folder {year_folder_path} does not exist, skipping.")
            continue

        for filename in os.listdir(year_folder_path):
            if filename.endswith('.HDFEOS'):
                file_path = os.path.join(year_folder_path, filename)
                all_files.append(file_path)

    with tqdm(total=len(all_files), desc=f"Processing NISE {hemisphere} hemisphere", unit="file") as pbar:
        for file_path in all_files:
            total_files += 1

            # 输出文件路径
            output_file = os.path.join(
                output_folder,
                f'NISE_{hemisphere}_icecon_25000_{os.path.basename(file_path)[-15:-7]}.nc'
            )

            # 跳过存在的文件
            if os.path.exists(output_file):
                print(f"Output file {output_file} already exists, skipping.")
                successful_files += 1
                pbar.update(1)
                continue

            try:
                # 打开文件
                hdf_file = gdal.Open(file_path)
                subdatasets = hdf_file.GetSubDatasets()

                # 提取 extent
                if hemisphere == 'north':
                    extent_ds = gdal.Open(subdatasets[0][0])  # 北极 extent-海冰密集度
                elif hemisphere == 'south':
                    extent_ds = gdal.Open(subdatasets[2][0])  # 南极 extent-海冰密集度

                extent = extent_ds.ReadAsArray()
                fill_value = extent_ds.GetRasterBand(1).GetNoDataValue()

                extent = np.where(extent == fill_value, np.nan, extent)

                ulxmap = -9036842.76
                ulymap = 9036842.76
                grid_size = 25067.53
                rows, cols = 721, 721
                x = np.linspace(ulxmap, ulxmap + grid_size * (cols - 1), cols)
                y = np.linspace(ulymap, ulymap - grid_size * (rows - 1), rows)
                x_grid, y_grid = np.meshgrid(x, y)

                transformer = Transformer.from_proj(proj_epsg, crs_wgs84, always_xy=True)
                lon, lat = transformer.transform(x_grid, y_grid)

                lat = np.where(np.isinf(lat), np.nan, lat)
                lon = np.where(np.isinf(lon), np.nan, lon)

                ds = xr.Dataset(
                    {
                        'ice_concentration': (['y', 'x'], extent),
                    },
                    coords={
                        'lat': (['y', 'x'], lat),
                        'lon': (['y', 'x'], lon),
                    }
                )

                # 裁剪
                if hemisphere == 'north':
                    ds = ds.where((ds['lat'] >= 60) & (ds['lat'] <= 90), drop=True)
                    ds = ds.where((ds['ice_concentration'] > 0) & (ds['ice_concentration'] <= 100), drop=True)
                elif hemisphere == 'south':
                    ds = ds.where((ds['lat'] >= -90) & (ds['lat'] <= -60), drop=True)
                    ds = ds.where((ds['ice_concentration'] > 0) & (ds['ice_concentration'] <= 100), drop=True)


                # 插值到规则网格
                transformer2 = Transformer.from_crs(crs_wgs84, crs_ps, always_xy=True)
                lon = ds['lon'].values
                lat = ds['lat'].values
                x, y = transformer2.transform(lon, lat)
                x_flat = x.flatten()
                y_flat = y.flatten()
                z_flat = ds['ice_concentration'].values.flatten()
                grid_data = griddata((x_flat, y_flat), z_flat, (grid_x_hem, grid_y_hem))

                # 保存 NetCDF
                da = xr.DataArray(
                    grid_data,
                    coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])],
                    dims=['y', 'x']
                )
                da_dataset = da.to_dataset(name=f'NISE_{hemisphere}_icecon')
                generate_snapshot(grid_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'NISE', snapshot_folder)
                da_dataset.to_netcdf(output_file)

                successful_files += 1

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1

            pbar.update(1)

    print(f"Total files processed: {total_files}")
    print(f"Successfully processed files: {successful_files}")
    print(f"Failed files: {failed_files}")

if __name__ == "__main__":
    process_nise_hemisphere('north')
    process_nise_hemisphere('south')