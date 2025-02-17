import netCDF4
import pyproj
import numpy as np
from scipy.interpolate import griddata

def reproject_icecon_to_latlon(input_filepath, output_filepath=None):
    """
    将海冰密集度数据从极射赤面投影重新投影到地理（纬度/经度）坐标。

    Args:
        input_filepath (str): 输入NetCDF文件 (.nc) 的路径，其中包含极射赤面投影数据。
        output_filepath (str, optional): 保存重新投影数据为新NetCDF文件的路径。
                                         如果为None，则重新投影的数据将作为numpy数组返回。

    Returns:
        tuple or None: 如果 output_filepath 为 None，则返回一个元组 (target_lon_grid, target_lat_grid, interpolated_ice_con)。
                       如果提供了 output_filepath，则将数据保存到NetCDF文件并返回 None。
    """
    try:
        # 1. 打开 NetCDF 文件
        with netCDF4.Dataset(input_filepath, 'r') as nc:
            print("正在读取 NetCDF 文件:", input_filepath)
            # 提取变量
            x_coords = nc.variables['x'][:]
            y_coords = nc.variables['y'][:]
            ice_con_data = nc.variables['ice_con'][:]  
            time_var = nc.variables['time']
            time_data = time_var[:]
            time_units = time_var.units
            time_calendar = time_var.calendar if hasattr(time_var, 'calendar') else 'gregorian'

            # 2. 定义投影
            # 标准的北极极射赤面投影 (EPSG:3413)
            source_crs = pyproj.CRS("EPSG:3413")  # 北极极射赤面投影
            target_crs = pyproj.CRS("EPSG:4326")  # WGS 84 (纬度/经度)
            transformer = pyproj.Transformer.from_crs(source_crs, target_crs, always_xy=True)

            # 3. 为原始投影创建坐标网格
            x_grid, y_grid = np.meshgrid(x_coords, y_coords)

            # 4. 将坐标转换为纬度和经度
            print("正在将坐标转换为纬度/经度...")
            lons, lats = transformer.transform(x_grid, y_grid)

            # **将经度从 -180~180 转换为 0~360**
            lons = (lons + 360) % 360

            # 5. 定义用于插值的目标纬度/经度网格
            # 注意，这里的目标经度直接设置为 0~360 范围
            lon_step = 0.25  # 经度间隔
            lat_step = 0.25  # 纬度间隔
            
            target_lon = np.arange(0, 360 + lon_step, lon_step)  # 经度从 0 到 360
            target_lat = np.arange(np.min(lats), np.max(lats) + lat_step, lat_step)
            target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)

            # 6. 将海冰密集度数据插值到目标网格
            print("正在插值海冰密集度数据...")
            points = np.vstack((lons.flatten(), lats.flatten())).T
            values = ice_con_data.flatten()
            interpolated_ice_con = griddata(points, values, (target_lon_grid, target_lat_grid), method='linear')

            # 将原始网格外部的插值NaN值替换为填充值
            interpolated_ice_con[np.isnan(interpolated_ice_con)] = -9999.0  # 或其他填充值

            if output_filepath:
                # 7. 将重新投影的数据保存到新的 NetCDF 文件
                print("正在保存重新投影的数据到:", output_filepath)
                with netCDF4.Dataset(output_filepath, 'w', format='NETCDF4') as out_nc:
                    # 创建维度
                    out_nc.createDimension('lon', target_lon.size)
                    out_nc.createDimension('lat', target_lat.size)
                    out_nc.createDimension('time', None)  # 假设需要时间维度

                    # 创建坐标变量
                    lon_var = out_nc.createVariable('lon', 'f4', ('lon',))
                    lat_var = out_nc.createVariable('lat', 'f4', ('lat',))
                    time_out_var = out_nc.createVariable('time', 'f8', ('time',))  # 双精度时间
                    ice_con_out_var = out_nc.createVariable('ice_con_latlon', 'f4', ('time', 'lat', 'lon'), fill_value=np.float32(-9999.0))

                    # 设置变量的属性 (对于元数据很重要)
                    lon_var.units = 'degrees_east'
                    lat_var.units = 'degrees_north'
                    time_out_var.units = time_units
                    time_out_var.calendar = time_calendar
                    ice_con_out_var.units = '(%)'  # 根据需要调整单位
                    ice_con_out_var.long_name = 'ice concentration'

                    # 将数据写入变量
                    lon_var[:] = target_lon
                    lat_var[:] = target_lat
                    time_out_var[:] = time_data  # 保留原始时间数据
                    ice_con_out_var[0, :, :] = interpolated_ice_con  # 假设为单时间步长，如果需要多时间步长则进行调整

                print("重新投影和保存完成。")
                return None
            else:
                print("重新投影完成。正在返回数据数组。")
                return target_lon_grid, target_lat_grid, interpolated_ice_con

    except FileNotFoundError:
        print(f"错误: 未找到输入文件 {input_filepath}")
        return None
    except Exception as e:
        print(f"发生错误: {e}")
        return None

# 测试:
input_file = "D:/zmh/icecon12500_data/ICECON_merge_12500_kalmanV1.4/north/Icecon_Combined/icecon_north_12500_combined_20220101.nc"
output_file = "D:/zmh/icecon_12500_latlon_20220101_linear_0-360.nc"  # 指定输出文件路径或设置为 None 仅返回数组

reproject_icecon_to_latlon(input_file, output_file)  # 保存到新文件
