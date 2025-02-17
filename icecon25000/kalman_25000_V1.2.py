"""
25km海冰密集度融合产品制作，采用kalman滤波算法，版本V1.2
融合数据包括：NSIDC-BS、NSIDC-NISE、FY3C、FY3D
极点空白问题：设置极点周围250000公里内均为100%密集度
南极不融入NISE，因为基本都是nan
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
from scipy.spatial.distance import cdist
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

plt.rcParams['font.sans-serif'] = ['SimHei']

# --- Kalman Filter Functions ---
def kalman_filter_update(z_current, F, H, Q, R, x_prev, P_prev):
    """
    Performs Kalman filter update for a single time step.
    """
    # Prediction step
    x_pred = F * x_prev
    P_pred = F * P_prev * F + Q

    # Update step
    y = np.where(np.isnan(z_current), 0, z_current - (H * x_pred))
    S = H * P_pred * H + R
    K = np.where(np.isnan(z_current), 0, P_pred * H / S)
    x_current = np.where(np.isnan(z_current), x_pred, x_pred + K * y)
    P_current = np.where(np.isnan(z_current), P_pred, (1 - K * H) * P_pred)

    return x_current, P_current

def estimate_Q_R_single(data_prev, data_current, default_Q=30, default_R=10):
    """Estimates process noise Q and measurement noise R for a single time step."""
    ice_conc_diff = data_current - data_prev
    process_variances = np.nanvar(ice_conc_diff)
    Q = process_variances
    measurement_variances = np.nanvar(data_current)
    R = measurement_variances
    return Q, R

# --- Data Reading Function ---
def read_data_single(file_path, var_name_pattern):
    """Reads a single NetCDF file, returns data for the specified variable and time from filename."""
    ds = xr.open_dataset(file_path)
    var_name = next((name for name in ds.data_vars if var_name_pattern in name), None)
    if var_name is None:
        ds.close()
        raise ValueError(f"No variable found in file matching pattern '{var_name_pattern}'")
    data = ds[var_name].values
    # Extract time from filename
    filename = os.path.basename(file_path)
    try:
        time_str = filename.split('_')[-1].split('.')[0]
        time = pd.to_datetime(time_str, format='%Y%m%d')
    except (ValueError, IndexError):
        ds.close()
        raise ValueError(f"Could not extract time from filename: {filename}")
    ds.close()
    return data, time, var_name

# --- Data Fusion Function ---
def calculate_combined_uncertainty(estimated_values, uncertainty_list):
    """
    Combines estimates and uncertainties from multiple datasets using weighted averaging.
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

# --- Data Processing and Kalman Filter Application ---
def process_dataset_at_time(dataset_name, config, hemisphere, current_time, t_index, x_prev_dict, P_prev_dict, base_data_dir, output_base_dir, x, y):
    """
    Processes a single dataset for a given time step, applies Kalman filter, and saves results.
    """

    print(f"  Processing dataset: {dataset_name}")
    folder_path = os.path.join(base_data_dir, config["path"], hemisphere)

    # Adjust variable pattern based on hemisphere
    adjusted_var_pattern = config["var_pattern"]
    if dataset_name == 'NSIDC-Bootstrap':
        adjusted_var_pattern = f"NSIDC-BS_{hemisphere}_icecon"

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

    # Initialize or update state and covariance
    if dataset_name not in x_prev_dict:
        x_prev_dict[dataset_name] = current_data
        P_prev_dict[dataset_name] = np.full_like(current_data, 30)

    # Estimate Q and R (if not the first time step)
    if t_index > 0:
        Q, R = estimate_Q_R_single(x_prev_dict[dataset_name], current_data, default_Q=config["Q"],
                                   default_R=config["R"])
        print(f"Estimated Q: {Q}, Estimated R:{R}")
    else:
        Q = config["Q"]
        R = config["R"]

    # Set F and H
    F = 1.0
    H = 1.0

    # Perform Kalman filter update
    x_current, P_current = kalman_filter_update(current_data, F, H, Q, R, x_prev_dict[dataset_name],
                                                P_prev_dict[dataset_name])

    # Save results
    save_kalman_filter_results(dataset_name, hemisphere, current_time, x_current, P_current, output_base_dir, x, y)

    # Update x_prev and P_prev
    x_prev_dict[dataset_name] = x_current
    P_prev_dict[dataset_name] = P_current

    return x_current, P_current

def save_kalman_filter_results(dataset_name, hemisphere, current_time, x_current, P_current, output_base_dir, x, y):
    """
    Saves the Kalman filter results to a NetCDF file.
    """
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

def fuse_and_save_data(estimated_values, uncertainty_list, current_time, output_base_dir, hemisphere, x, y):
    """
    Fuses data from multiple datasets and saves the combined result.
    """
    if estimated_values:
        combined_values, combined_uncertainty = calculate_combined_uncertainty(estimated_values, uncertainty_list)

        if hemisphere == "north":
            # Assuming the North Pole is approximately at the center of the grid (0, 0) in the projected coordinates
            center_x = 12500
            center_y = 12500
            max_distance = 250000  # 250 km

            # Calculate squared distances to avoid sqrt for performance
            distances_squared = (x - center_x)**2 + (y[:, np.newaxis] - center_y)**2

            # Identify indices within the radius
            within_radius_mask = distances_squared <= max_distance**2

            # Set ice concentration to 100 within the radius
            combined_values[within_radius_mask] = 100


        # Save combined results
        time_str = current_time.strftime('%Y%m%d')
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

        del combined_values, combined_uncertainty
        
# --- Plotting Functions ---
def create_north_polar_plot(data_file, output_dir):
    """
    Creates a North Polar Stereo plot.
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
    Creates a South Polar Stereo plot.
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

# --- Main Program ---
if __name__ == "__main__":
    base_data_dir = r"D:\zmh\icecon25000_data"
    hemispheres = ["north", "south"]  # Now includes both hemispheres
    dataset_configs = {
        "NISE": {"path": "NISE/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "NSIDC-Bootstrap": {"path": "NSIDC-Bootstrap/0_pre_com", "var_pattern": "NSIDC-Bootstrap", "Q": 30, "R": 10},
        "FY3C": {"path": "FY/FY-3C/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
        "FY3D": {"path": "FY/FY-3D/1_PDFcorrected", "var_pattern": "ice_concentration", "Q": 30, "R": 10},
    }

    for hemisphere in hemispheres:
        print(f"Processing {hemisphere} hemisphere")
        output_base_dir = os.path.join(base_data_dir, "ICECON_merge_25000")

        # Get time list for all datasets
        all_times = set()
        for dataset_name, config in dataset_configs.items():
            folder_path = os.path.join(base_data_dir, config["path"], hemisphere)
            files = sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if
                            f.endswith('.nc') and hemisphere in f])
            for file in files[365:]:
                try:
                    _, time, _ = read_data_single(file, config["var_pattern"])
                    all_times.add(time)
                except ValueError as e:
                    print(f"Skipping file {file} due to error: {e}")
        sorted_times = sorted(list(all_times))

        # Initialize previous state and covariance for each dataset
        x_prev_dict = {}
        P_prev_dict = {}

        # Get x and y coordinates (assuming they are the same for all datasets)
        # This is done outside the time loop to avoid repetition
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

        # Loop through each time point
        for t_index, current_time in enumerate(sorted_times):
            print(f"Processing time: {current_time}")

            estimated_values = []
            uncertainty_list = []

            # Process each dataset
            for dataset_name, config in dataset_configs.items():
                if hemisphere == "south" and dataset_name == "NISE":
                    continue
                x_current, P_current = process_dataset_at_time(
                    dataset_name, config, hemisphere, current_time, t_index, x_prev_dict, P_prev_dict,
                    base_data_dir, output_base_dir, x, y
                )
                if x_current is not None and P_current is not None:
                    estimated_values.append(x_current)
                    uncertainty_list.append(P_current)

            # Fuse and save data
            fuse_and_save_data(estimated_values, uncertainty_list, current_time, output_base_dir, hemisphere, x, y)

            del estimated_values, uncertainty_list

    # --- Plotting ---
    data_dir = os.path.join(base_data_dir, "ICECON_merge_25000")
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