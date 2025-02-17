import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import os
import shutil
from pyproj import CRS, Transformer
from tqdm import tqdm
from RegularGrid import north_regular_grid, south_regular_grid,generate_snapshot

# 文件路径
base_folder_path = r"D:\zmh\icecon25000_data\FY\FY-3C"
output_folder_north = os.path.join(base_folder_path, '0_pre_com', 'north')
output_folder_south = os.path.join(base_folder_path, '0_pre_com', 'south')
failure_folder = os.path.join(base_folder_path, '0_pre_com', 'failures')
snapshot_folder = os.path.join(base_folder_path, '0_pre_com', "snapshots")

os.makedirs(output_folder_north, exist_ok=True)
os.makedirs(output_folder_south, exist_ok=True)
os.makedirs(failure_folder, exist_ok=True)
os.makedirs(snapshot_folder, exist_ok=True)

# CRS 和规则网格
crs_wgs84 = CRS.from_epsg(4326)
crs_psn = CRS.from_epsg(3413)  # 北极投影
crs_pss = CRS.from_epsg(3976)  # 南极投影

# 北极和南极规则网格
grid_x_n, grid_y_n = north_regular_grid()
grid_x_s, grid_y_s = south_regular_grid()

def process_hemisphere(hemisphere: str):
    """
    处理指定半球的FY-3D数据并保存为NetCDF格式
    """
    if hemisphere == 'north':
        crs_ps = crs_psn
        grid_x_hem = grid_x_n
        grid_y_hem = grid_y_n
        output_folder = output_folder_north
        lat_file = r"D:\zmh\icecon25000_data\RegularGrid\psn12lats_v3.txt"
        lon_file = r"D:\zmh\icecon25000_data\RegularGrid\psn12lons_v3.txt"
        target_variable = 'icecon_north_avg'
        output_variable = 'FY3C_north_icecon'
        lat_filter = 60
        lat_condition = lambda lat: lat >= lat_filter
    elif hemisphere == 'south':
        crs_ps = crs_pss
        grid_x_hem = grid_x_s
        grid_y_hem = grid_y_s
        output_folder = output_folder_south
        lat_file = r"D:\zmh\icecon25000_data\RegularGrid\FY3d_south_lat.txt"
        lon_file = r"D:\zmh\icecon25000_data\RegularGrid\FY3d_south_lon.txt"
        target_variable = 'icecon_south_avg'
        output_variable = 'FY3C_south_icecon'
        lat_filter = -60
        lat_condition = lambda lat: lat <= lat_filter
    else:
        raise ValueError("Invalid hemisphere specified. Choose 'north' or 'south'.")

    # 加载经纬度数据
    lat = np.loadtxt(lat_file)
    lon = np.loadtxt(lon_file)

    total_files = 0
    successful_files = 0
    failed_files = 0
    all_files = []

    # 遍历所有年份文件夹
    for year in range(1984, 2024):  # 假设1984-2023年的数据
        year_folder = os.path.join(base_folder_path, str(year))
        if not os.path.exists(year_folder):
            continue
        for filename in os.listdir(year_folder):
            if filename.endswith('.HDF'):
                all_files.append(os.path.join(year_folder, filename))

    with tqdm(total=len(all_files), desc=f"Processing {hemisphere}", unit="file") as pbar:
        for file_path in all_files:
            total_files += 1
            try:
                # 打开数据集
                ds_sic = xr.open_dataset(file_path)
                if hemisphere == 'north':
                    for drop_var in ['icecon_north_asc', 'icecon_north_des', 'icecon_south_asc', 'icecon_south_des', 'icecon_south_avg']:
                        if drop_var in ds_sic:
                            ds_sic = ds_sic.drop_vars(drop_var)
                else:
                    for drop_var in ['icecon_north_asc', 'icecon_north_des', 'icecon_south_asc', 'icecon_south_des', 'icecon_north_avg']:
                        if drop_var in ds_sic:
                            ds_sic = ds_sic.drop_vars(drop_var)

                # 分配经纬度坐标
                if hemisphere == 'north':
                    ds_sic = ds_sic.assign_coords({
                        "lon": (("phony_dim_0", "phony_dim_1"), lon.T),
                        "lat": (("phony_dim_0", "phony_dim_1"), lat.T),
                    })
                else:
                    ds_sic = ds_sic.assign_coords({
                        "lon": (("phony_dim_2", "phony_dim_3"), lon),
                        "lat": (("phony_dim_2", "phony_dim_3"), lat),
                    })

                # 筛选目标纬度
                filtered_ds = ds_sic.where(lat_condition(ds_sic["lat"]), drop=False)
                mask = (filtered_ds[target_variable] >= 0) & (filtered_ds[target_variable] <= 100)
                filtered_ds[target_variable] = filtered_ds[target_variable].where(mask, drop=False)

                # 投影转换
                lon_values = filtered_ds["lon"].values
                lat_values = filtered_ds["lat"].values
                transformer = Transformer.from_crs(crs_wgs84, crs_ps, always_xy=True)
                x, y = transformer.transform(lon_values, lat_values)
                if hemisphere == 'north':
                    filtered_ds = filtered_ds.assign_coords({"x": (("phony_dim_0", "phony_dim_1"), x),
                                                            "y": (("phony_dim_0", "phony_dim_1"), y)})
                else:
                    filtered_ds = filtered_ds.assign_coords({"x": (("phony_dim_2", "phony_dim_3"), x),
                                                            "y": (("phony_dim_2", "phony_dim_3"), y)})

                # 插值到规则网格
                x_flat = x.flatten()
                y_flat = y.flatten()
                z_flat = filtered_ds[target_variable].values.flatten()
                grid_data = griddata((x_flat, y_flat), z_flat, (grid_x_hem, grid_y_hem), method="nearest")

                # 创建 DataArray 并保存
                da = xr.DataArray(grid_data, coords=[('y', grid_y_hem[:, 0]), ('x', grid_x_hem[0, :])], dims=['y', 'x'])
                da_dataset = da.to_dataset(name=output_variable)

                output_file = os.path.join(output_folder, f"{output_variable}_25000_{os.path.basename(file_path)[-26:-18]}.nc")
                generate_snapshot(grid_data, grid_x_hem, grid_y_hem, hemisphere, output_file, 'FY3C', snapshot_folder)
                da_dataset.to_netcdf(output_file)

                successful_files += 1
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                shutil.copy(file_path, failure_folder)
                failed_files += 1

            pbar.update(1)

    print(f"Total files processed for {hemisphere}: {total_files}")
    print(f"Successfully processed files for {hemisphere}: {successful_files}")
    print(f"Failed files for {hemisphere}: {failed_files}")

if __name__ == "__main__":
    # process_hemisphere('north')  # 处理北极
    process_hemisphere('south')  # 处理南极
