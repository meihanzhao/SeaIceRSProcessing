from scipy.stats import percentileofscore
import numpy as np
import pandas as pd
import xarray as xr
from RegularGrid import north_regular_grid
from SITsnapshot import generate_SIT_snapshot
from paths import paths
import os

def calculate_cdf(data):
    """ 
    计算CDF（累积分布函数）
    """
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return sorted_data, cdf

def pdf_match_segmented(non_ref_data, ref_data, window_size=40):
    """
    为非参考数据中的每个网格单元执行PDF匹配，并计算校正值delta_r。
    """
    rows, cols = non_ref_data.shape
    delta_r = np.zeros_like(non_ref_data)

    for i in range(rows):
        for j in range(cols):
            # 计算局部窗口的边界
            row_min = max(0, i - window_size // 2)
            row_max = min(rows, i + window_size // 2 + 1)
            col_min = max(0, j - window_size // 2)
            col_max = min(cols, j + window_size // 2 + 1)

            # 提取局部窗口数据
            non_ref_window = non_ref_data[row_min:row_max, col_min:col_max]
            if ref_data.ndim == 2:
                ref_window = ref_data[row_min:row_max, col_min:col_max]
                ref_center = ref_data[i, j]
            elif ref_data.ndim == 3:
                ref_window = ref_data[:, row_min:row_max, col_min:col_max]
                ref_center = ref_data[-1, i, j]
            # 排除NaN值
            non_ref_valid = non_ref_window[~np.isnan(non_ref_window)]
            ref_valid = ref_window[~np.isnan(ref_window)]

            # 如果参考数据中心点为0且非参考数据中心点不为0，则不进行校正
            if non_ref_data[i, j] != 0 and ref_center == 0:
                delta_r[i, j] = 0
                continue
            
            # 如果窗口中有效数据点数不足300，则不进行校正
            if len(ref_valid) < 300 or len(non_ref_valid) < 300:
                delta_r[i, j] = 0
                continue

            # 计算当前窗口中的CDF
            ref_sorted, ref_cdf = calculate_cdf(ref_valid)
            non_ref_sorted, non_ref_cdf = calculate_cdf(non_ref_valid)

            # 使用非参考数据中心点的CDF值在参考数据CDF中进行插值
            matched_values = np.interp(non_ref_cdf, ref_cdf, ref_sorted)
            delta_r[i, j] = non_ref_sorted[len(non_ref_sorted) // 2] - matched_values[len(matched_values) // 2]

    return delta_r

def correct_systematic_bias(non_ref_data, ref_data, delta_r):
    """
    根据校正值delta_r校正非参考数据的偏差。
    """
    corrected_data = non_ref_data.copy()
    valid_mask = ~np.isnan(non_ref_data)
    delta_r = np.clip(delta_r, -0.5, 0.5)  # 限制校正值的范围[0.5, 0.5]
    corrected_data[valid_mask] -= delta_r[valid_mask]

    return corrected_data

def pdf_correction(base_input_dir, base_output_dir, base_reference_dir, hemisphere, start_date, end_date, window_size=40, time_window=30):
    """
    在非参考数据中执行PDF匹配并校正偏差。
    参数：
        base_input_dir (str)：非参考数据的基本目录
        base_output_dir (str)：校正数据的基本目录
        base_reference_dir (str)：参考数据的基本目录
        hemisphere (str)：半球（'north'）。
        start_date (str)：开始日期，格式为'YYYY-MM-DD'。
        end_date (str)：结束日期，格式为'YYYY-MM-DD'。
        window_size (int)：PDF匹配的空间窗口大小。
        time_window (int)：参考数据的时间窗口天数。
    """
    input_dir = os.path.join(base_input_dir, hemisphere)
    output_dir = os.path.join(base_output_dir, hemisphere)
    reference_dir = os.path.join(base_reference_dir, hemisphere)
    os.makedirs(output_dir, exist_ok=True)
    date_range = pd.date_range(start=start_date, end=end_date)

    # 确定非参考和参考数据变量名称和文件前缀
    if hemisphere == 'north': 
        non_ref_var = 'CS2_sit'
        non_ref_file_prefix = 'CS2_north_SIT_25000_'
        ref_var = 'appx_sit'
        ref_file_prefix = 'APPX_north_SIT_25000_'
        grid_x_n, grid_y_n = north_regular_grid()

    else:
        raise ValueError("Invalid hemisphere. Choose 'north'.")

    for current_date in date_range:
        print(f"Processing date: {current_date} for {hemisphere} hemisphere")

        # 遍历非参考数据
        non_ref_file = os.path.join(input_dir, f"{non_ref_file_prefix}{current_date.strftime('%Y%m%d')}.nc")
        if not os.path.exists(non_ref_file):
            print(f"Non-reference file missing: {non_ref_file}")
            continue
        non_ref_data = xr.open_dataset(non_ref_file)[non_ref_var].values

        # 遍历参考数据
        window_start = current_date - pd.Timedelta(days=time_window - 1)
        temporal_files = [
            os.path.join(reference_dir, f"{ref_file_prefix}{d.strftime('%Y%m%d')}.nc")
            for d in pd.date_range(window_start, current_date)
            if os.path.exists(os.path.join(reference_dir, f"{ref_file_prefix}{d.strftime('%Y%m%d')}.nc"))
        ]

        if len(temporal_files) < time_window:
            print(f"Not enough reference files for the temporal window for {current_date}. Skipping.")
            continue

        # 计算时间窗口内的参考数据均值
        ref_window_data = xr.concat([xr.open_dataset(f)[ref_var] for f in temporal_files], dim='time')
        ref_data = ref_window_data.values

        # 在空间窗口内，进行PDF匹配
        delta_r = pdf_match_segmented(non_ref_data, ref_data, window_size=window_size)
        corrected_data = correct_systematic_bias(non_ref_data, ref_data, delta_r)

        # 保存校正数据
        corrected_ds = xr.Dataset({
            'CS2_sit': (('y', 'x'), corrected_data)
        }, coords={'x': xr.open_dataset(non_ref_file)['x'], 'y': xr.open_dataset(non_ref_file)['y']})

        output_file = os.path.join(output_dir, f"CS2_pdfcorrected_{hemisphere}_SIT_25000_{current_date.strftime('%Y%m%d')}.nc")
        corrected_ds.to_netcdf(output_file)

        generate_SIT_snapshot(corrected_ds, grid_x_n, grid_y_n, output_file, 'CS2_sit', snapshot_folder)
    
        print(f"Saved corrected file: {output_file}")

if __name__ == "__main__":
    CS2_pdf_base_input_dir = os.path.join(paths['CS2_base_folder_path'], "0_pre_com")
    CS2_pdf_base_output_dir = os.path.join(paths['CS2_base_folder_path'], "1_PDFcorrected")
    base_pdf_reference_dir = os.path.join(paths['appx_base_folder_path'], "0_pre_com")
    snapshot_folder = os.path.join(CS2_pdf_base_output_dir, "snapshots")
    os.makedirs(snapshot_folder, exist_ok=True)  

    start_date = "2004-01-01"
    end_date = "2024-01-01"

    # PDF校正北极海冰厚度数据
    for hemisphere in ["north"]:
        window_size = 40 if hemisphere == "north" else 25
        pdf_correction(CS2_pdf_base_input_dir, CS2_pdf_base_output_dir, base_pdf_reference_dir, hemisphere, start_date, end_date, window_size=window_size, time_window=1)