import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LATITUDE_FORMATTER
import numpy as np
import xarray as xr
import os
import pyproj
from scipy import ndimage
import matplotlib.path as mpath  # Import for circular boundary

# --- Define the base data directory ---
base_data_dir = r'D:\zmh\icecon25000_data'
hemispheres = ['north', 'south']

# --- 画图函数 ---
def create_north_polar_plot(data_file, output_dir):
    """
    创建北极立体投影图
    """
    colors = ["#f0f9e8", "#bae4bc", "#7bccc4", "#43a2ca", "#0868ac", "#084081"]
    cmap = LinearSegmentedColormap.from_list("SeaIceCmap", colors)
    # Change central_longitude to 90 to position 90E at the bottom
    proj_crs = ccrs.NorthPolarStereo(central_longitude=90)
    proj = pyproj.Proj(proj='stere', lat_ts=70, lat_0=90, lon_0=90)

    combined_data = xr.open_dataset(data_file)
    date_str = os.path.basename(data_file).split('_')[4].split('.')[0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj_crs})
    extent = [-180, 180, 58.5, 90]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon_grid, lat_grid = np.meshgrid(combined_data.x, combined_data.y)
    lon, lat = proj(lon_grid, lat_grid, inverse=True)

    ice_data = combined_data['ice_con'].astype(float)
    print(ice_data.shape)
    # ice_data = np.rot90(ice_data,k=2)
    threshold = 1e-10
    ice_data = ndimage.rotate(np.rot90(ice_data,k=1),45, reshape=True, cval=np.nan)
    ice_data[np.abs(ice_data) < threshold] = np.nan
    print(ice_data.shape)
    exit()
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
    gl.bottom_labels = True  # Enable bottom labels
    gl.ylabels_left = True
    gl.xlabels_top = False
    gl.xlabels_bottom = True  # Enable bottom x-labels
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black', 'rotation': 0}  # Rotate x-labels if needed
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # --- Add Longitude Labels (North) ---
    # 90E at the bottom
    ax.text(90, 55, '90°E', transform=ccrs.PlateCarree(), ha='center', va='top', fontsize=10, color='black')
    # 90W at the top
    ax.text(-90, 55, '90°W', transform=ccrs.PlateCarree(), ha='center', va='bottom', fontsize=10, color='black')

    # --- Create Circular Boundary (North) ---
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8, ax=ax)
    cbar.set_label('海冰密集度 (%)')

    output_filename = os.path.join(output_dir, f"北极海冰密集度融合产品_25000_{date_str}.png")
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
    # Set central_longitude to -90 for 90W at the bottom
    proj_crs = ccrs.SouthPolarStereo(central_longitude=-90)
    proj = pyproj.Proj(proj='stere', lat_ts=-70, lat_0=-90, lon_0=-90)

    combined_data = xr.open_dataset(data_file)
    date_str = os.path.basename(data_file).split('_')[4].split('.')[0]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': proj_crs})
    extent = [-180, 180, -90, -60]
    ax.set_extent(extent, crs=ccrs.PlateCarree())

    lon_grid, lat_grid = np.meshgrid(combined_data.x, combined_data.y)
    lon, lat = proj(lon_grid, lat_grid, inverse=True)

    ice_data = combined_data['ice_con'].astype(float)
    ice_data = np.rot90(ice_data, k=2)
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
    gl.ylocator = mticker.FixedLocator([-80, -70, -60])
    gl.top_labels = False
    gl.right_labels = False
    gl.bottom_labels = True  # Enable bottom labels
    gl.ylabels_left = True
    gl.xlabels_top = False
    gl.xlabels_bottom = True  # Enable bottom x-labels
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 10, 'color': 'black', 'rotation': 0}  # Rotate x-labels if needed
    gl.ylabel_style = {'size': 10, 'color': 'black'}

    # --- Add Longitude Labels (South) ---
    # 90W at the bottom
    ax.text(-90, -58, '90°W', transform=ccrs.PlateCarree(), ha='center', va='top', fontsize=10, color='black')
    # 90E at the top
    ax.text(90, -58, '90°E', transform=ccrs.PlateCarree(), ha='center', va='bottom', fontsize=10, color='black')

    # --- Create Circular Boundary (South) ---
    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)

    cbar = plt.colorbar(mappable, orientation='horizontal', pad=0.05, shrink=0.8, ax=ax)
    cbar.set_label('海冰密集度 (%)')

    output_filename = os.path.join(output_dir, f"南极海冰密集度融合产品_25000_{date_str}.png")
    plt.savefig(output_filename)
    # print(f"Plot saved to {output_filename}")

    plt.close(fig)
    combined_data.close()


data_dir = os.path.join(base_data_dir, "ICECON_merge_25000")
for hemisphere in hemispheres:
    output_dir = os.path.join(data_dir, "snapshots", hemisphere)
    os.makedirs(output_dir, exist_ok=True)
    # Adjusted path for 'north' data
    input_dir = os.path.join(data_dir, hemisphere, "Icecon_Combined") if hemisphere == "north" else os.path.join(data_dir, hemisphere, "Icecon_Combined")
    for filename in sorted(os.listdir(input_dir)):
        if filename.endswith(".nc") and "combined" in filename:
            data_file = os.path.join(input_dir, filename)
            if hemisphere == "north":
                create_north_polar_plot(data_file, output_dir)
            elif hemisphere == "south":
                create_south_polar_plot(data_file, output_dir)