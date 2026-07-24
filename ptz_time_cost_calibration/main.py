import math
import time
import csv
import statistics
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import curve_fit
import warnings
import os

warnings.filterwarnings('ignore')

from base_control_sdk import *
# from absoute_sdk_utils_gsoa import new_gotopos_sdk_sync_e, focus_sdk, new_gotopos_sdk_sync_e_with_focus

device = None
is_initialized = False


def get_device():
    global device, is_initialized
    if not is_initialized:
        device = Device()
        is_initialized = True
    return device


def test_single_step_size(step_size, axis='P', step_index=None, total_steps=None):
    """
    测试单个步长的双向移动用时

    参数:
        step_size: 要测试的步长
        axis: 测试的轴 ('P', 'T', 'Z')
        step_index: 当前步长索引（用于显示进度）
        total_steps: 总步长数（用于显示进度）
    """

    # 可配置的等待时间参数
    WAIT_CONFIG = {
        'between_tests': 1,  # 测试间等待(秒)
        'between_directions': 3,  # 正反向测试间等待(秒)
        'between_groups': 2,  # 组间等待(秒)
        'stabilization': 0.5  # 稳定时间(秒)
    }

    # 测试配置
    random_groups_per_step = 10
    tests_per_group = 2

    # 显示进度信息
    axis_name = {'P': 'P方向', 'T': 'T方向', 'Z': 'Z方向'}[axis]
    if step_index is not None and total_steps is not None:
        print(f"\n{'=' * 80}")
        print(f"测试{axis_name} 步长 {step_size:.1f}° ({step_index}/{total_steps})")
        print(f"{'=' * 80}")
    else:
        print(f"\n{'=' * 80}")
        print(f"测试{axis_name} 步长 {step_size:.1f}°")
        print(f"{'=' * 80}")

    # def test_single_movement(start_pos, end_pos, movement_type, num_tests=5):
    #     """测试单个移动的用时"""
    #     times = []
    #     ptz_data_list = []
    #
    #     for i in range(num_tests):
    #         # 生成随机位置参数
    #         if axis == 'P':
    #             # 测试P方向时，随机生成T和Z
    #             random_t = round(random.uniform(-5, 90), 1)
    #             random_z = round(random.uniform(0, 25), 1)
    #             start_position = [start_pos, random_t, random_z]
    #             end_position = [end_pos, random_t, random_z]
    #         elif axis == 'T':
    #             # 测试T方向时，随机生成P和Z
    #             random_p = round(random.uniform(10, 350), 1)
    #             random_z = round(random.uniform(0, 25), 1)
    #             start_position = [random_p, start_pos, random_z]
    #             end_position = [random_p, end_pos, random_z]
    #         else:  # axis == 'Z'
    #             # 测试Z方向时，随机生成P和T
    #             random_p = round(random.uniform(10, 350), 1)
    #             random_t = round(random.uniform(-5, 90), 1)
    #             start_position = [random_p, random_t, start_pos]
    #             end_position = [random_p, random_t, end_pos]
    #
    #         print(f"      {movement_type} 测试 {i + 1}/{num_tests}: {start_pos:.1f}° → {end_pos:.1f}°")
    #
    #         # 先移动到起始位置
    #         success_start, _ = new_gotopos_sdk_sync_e(start_position)
    #         if not success_start:
    #             print("移动到起始位置失败")
    #             continue
    #
    #         time.sleep(WAIT_CONFIG['stabilization'])
    #
    #         # 测试移动
    #         success, movement_time = new_gotopos_sdk_sync_e(end_position)
    #
    #         # 保存其他轴的值
    #         if axis == 'P':
    #             other_axis1 = random_t
    #             other_axis2 = random_z
    #         elif axis == 'T':
    #             other_axis1 = random_p
    #             other_axis2 = random_z
    #         else:  # Z
    #             other_axis1 = random_p
    #             other_axis2 = random_t
    #
    #         test_data = {
    #             'movement_time': movement_time,
    #             'success': success,
    #             'target_position': end_position,
    #             'other_axis1': other_axis1,
    #             'other_axis2': other_axis2
    #         }
    #
    #         ptz_data_list.append(test_data)
    #
    #         if success:
    #             times.append(movement_time)
    #             print(f"        成功! 用时: {movement_time:.3f} 秒")
    #         else:
    #             print(f"        失败! 已用时间: {movement_time:.3f} 秒")
    #
    #         # 测试间隔
    #         if i < num_tests - 1:
    #             time.sleep(WAIT_CONFIG['between_tests'])
    #
    #     return times, ptz_data_list

    # def get_random_start_position(step_size, used_positions, axis):
    #     """获取随机的起始位置"""
    #     max_attempts = 100
    #
    #     # 根据轴设置范围
    #     if axis == 'P':
    #         min_val, max_val = 10, 350
    #     elif axis == 'T':
    #         min_val, max_val = -5, 85  # T方向范围
    #     else:  # Z
    #         min_val, max_val = 0, 24  # Z方向范围
    #
    #     margin = 5  # 边界余量
    #
    #     for _ in range(max_attempts):
    #         start_pos = round(random.uniform(min_val + margin, max_val - step_size - margin), 1)
    #         too_close = any(abs(start_pos - used_p) < 2.0 for used_p in used_positions)
    #
    #         if not too_close:
    #             used_positions.append(start_pos)
    #             return start_pos
    #
    #     return round(random.uniform(min_val + margin, max_val - step_size - margin), 1)

    # 结果存储
    step_group_results = []
    all_ptz_data = []
    used_start_positions = []

    # 开始测试该步长
    # for group_id in range(1, random_groups_per_step + 1):
    #     print(f"\n  组 {group_id}/{random_groups_per_step}")
    #
    #     # 获取随机起始位置
    #     start_pos = get_random_start_position(step_size, used_start_positions, axis)
    #     end_pos = round(start_pos + step_size, 1)
    #
    #     # 检查边界
    #     if axis == 'P' and end_pos > 360:
    #         end_pos = 360.0
    #         start_pos = round(360 - step_size, 1)
    #     elif axis == 'T' and end_pos > 90:
    #         end_pos = 90.0
    #         start_pos = round(90 - step_size, 1)
    #     elif axis == 'Z' and end_pos > 25:
    #         end_pos = 25.0
    #         start_pos = round(25 - step_size, 1)
    #
    #     print(f"    正向移动: {start_pos:.1f}° → {end_pos:.1f}° (步长: {step_size:.1f}°)")
    #
    #     # 测试正向移动
    #     forward_times, forward_ptz_data = test_single_movement(start_pos, end_pos, 'forward', tests_per_group)
    #
    #     time.sleep(WAIT_CONFIG['between_directions'])
    #
    #     print(f"    反向移动: {end_pos:.1f}° → {start_pos:.1f}° (步长: {step_size:.1f}°)")
    #
    #     # 测试反向移动
    #     return_times, return_ptz_data = test_single_movement(end_pos, start_pos, 'return', tests_per_group)
    #
    #     # 计算组统计
    #     if forward_times and return_times:
    #         forward_avg = statistics.mean(forward_times)
    #         forward_std = statistics.stdev(forward_times) if len(forward_times) > 1 else 0
    #
    #         return_avg = statistics.mean(return_times)
    #         return_std = statistics.stdev(return_times) if len(return_times) > 1 else 0
    #
    #         bidirectional_avg = (forward_avg + return_avg) / 2
    #
    #         # 获取其他轴的值
    #         if forward_ptz_data:
    #             if axis == 'P':
    #                 other_axis1 = forward_ptz_data[0]['other_axis1']  # T值
    #                 other_axis2 = forward_ptz_data[0]['other_axis2']  # Z值
    #             elif axis == 'T':
    #                 other_axis1 = forward_ptz_data[0]['other_axis1']  # P值
    #                 other_axis2 = forward_ptz_data[0]['other_axis2']  # Z值
    #             else:  # Z
    #                 other_axis1 = forward_ptz_data[0]['other_axis1']  # P值
    #                 other_axis2 = forward_ptz_data[0]['other_axis2']  # T值
    #         else:
    #             other_axis1 = 0
    #             other_axis2 = 0
    #
    #         group_result = {
    #             'axis': axis,
    #             'step_size': step_size,
    #             'group_id': group_id,
    #             'start_pos': start_pos,
    #             'end_pos': end_pos,
    #             'other_axis1': other_axis1,
    #             'other_axis2': other_axis2,
    #             'forward_avg_time': forward_avg,
    #             'forward_std_dev': forward_std,
    #             'return_avg_time': return_avg,
    #             'return_std_dev': return_std,
    #             'bidirectional_avg_time': bidirectional_avg,
    #             'forward_tests': len(forward_times),
    #             'return_tests': len(return_times)
    #         }
    #
    #         step_group_results.append(group_result)
    #
    #         print(f"    组 {group_id} 完成")
    #         print(f"      正向: {forward_avg:.3f} ± {forward_std:.3f}s")
    #         print(f"      反向: {return_avg:.3f} ± {return_std:.3f}s")
    #         print(f"      双向平均: {bidirectional_avg:.3f}s")
    #         if axis == 'P':
    #             print(f"      T值: {other_axis1:.1f}°, Z值: {other_axis2:.1f}x")
    #         elif axis == 'T':
    #             print(f"      P值: {other_axis1:.1f}°, Z值: {other_axis2:.1f}x")
    #         else:  # Z
    #             print(f"      P值: {other_axis1:.1f}°, T值: {other_axis2:.1f}°")
    #
    #     # 组间等待
    #     if group_id < random_groups_per_step:
    #         print(f"    组间等待 {WAIT_CONFIG['between_groups']} 秒...")
    #         time.sleep(WAIT_CONFIG['between_groups'])

    # 计算该步长的统计信息
    step_summary = None
    if step_group_results:
        forward_avgs = [r['forward_avg_time'] for r in step_group_results]
        step_forward_avg = statistics.mean(forward_avgs)
        step_forward_std = statistics.stdev(forward_avgs) if len(forward_avgs) > 1 else 0

        return_avgs = [r['return_avg_time'] for r in step_group_results]
        step_return_avg = statistics.mean(return_avgs)
        step_return_std = statistics.stdev(return_avgs) if len(return_avgs) > 1 else 0

        bidirectional_avgs = [r['bidirectional_avg_time'] for r in step_group_results]
        step_bidirectional_avg = statistics.mean(bidirectional_avgs)
        step_bidirectional_std = statistics.stdev(bidirectional_avgs) if len(bidirectional_avgs) > 1 else 0

        step_summary = {
            'axis': axis,
            'step_size': step_size,
            'step_forward_avg_time': step_forward_avg,
            'step_forward_std_dev': step_forward_std,
            'step_return_avg_time': step_return_avg,
            'step_return_std_dev': step_return_std,
            'step_bidirectional_avg_time': step_bidirectional_avg,
            'step_bidirectional_std_dev': step_bidirectional_std,
            'successful_groups': len(step_group_results),
            'total_tests': len(step_group_results) * tests_per_group * 2
        }

        print(f"\n{axis_name} 步长 {step_size:.1f}° 统计:")
        print(f"  正向平均: {step_forward_avg:.3f} ± {step_forward_std:.3f} 秒")
        print(f"  反向平均: {step_return_avg:.3f} ± {step_return_std:.3f} 秒")
        print(f"  双向平均: {step_bidirectional_avg:.3f} ± {step_bidirectional_std:.3f} 秒")
        print(f"  成功组数: {len(step_group_results)}/{random_groups_per_step}")

    # 保存该步长的数据到单独文件
    save_single_step_data(axis, step_size, step_summary, step_group_results, all_ptz_data)

    return step_summary, step_group_results, all_ptz_data


def save_single_step_data(axis, step_size, step_summary, group_results, ptz_data):
    """保存单个步长的测试数据"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis_name = {'P': 'P', 'T': 'T', 'Z': 'Z'}[axis]

    # 创建文件夹
    folder_name = f"ptz_test_data_{axis_name}_step_{step_size:.1f}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # 保存组统计
    if group_results:
        group_df = pd.DataFrame(group_results)
        group_filename = f"{folder_name}/group_stats.xlsx"
        group_df.to_excel(group_filename, index=False)

    print(f"\n{axis_name}方向 步长 {step_size:.1f}° 数据已保存到文件夹: {folder_name}")


# def collect_all_step_data(axis=None):
#     """收集所有已测试步长的数据并汇总"""
#     # 查找所有步长测试文件夹
#     if axis:
#         folder_prefix = f'ptz_test_data_{axis}_step_'
#     else:
#         folder_prefix = 'ptz_test_data_'
#
#     step_folders = [f for f in os.listdir('.') if f.startswith(folder_prefix) and os.path.isdir(f)]
#
#     if not step_folders:
#         print("未找到任何步长测试数据")
#         return
#
#     all_group_stats = []
#
#     for folder in step_folders:
#         try:
#             # 提取轴和步长值
#             parts = folder.split('_')
#             axis_from_folder = parts[3]  # 从文件夹名中提取轴
#             step_size_str = parts[5]  # 从文件夹名中提取步长
#             step_size = float(step_size_str)
#
#             # 读取组统计
#             group_file = f"{folder}/group_stats.xlsx"
#             if os.path.exists(group_file):
#                 group_df = pd.read_excel(group_file)
#                 all_group_stats.append(group_df)
#
#         except Exception as e:
#             print(f"读取文件夹 {folder} 数据时出错: {e}")
#
#     # 合并所有数据
#     if all_group_stats:
#         combined_groups = pd.concat(all_group_stats, ignore_index=True)
#         combined_groups = combined_groups.sort_values(['axis', 'step_size', 'group_id'])
#
#         # 保存合并后的数据
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         axis_suffix = f"_{axis}" if axis else "_all"
#         combined_filename = f"combined_group_stats{axis_suffix}_{timestamp}.xlsx"
#         combined_groups.to_excel(combined_filename, index=False)
#
#         print(f"\n所有组统计数据已合并保存到: {combined_filename}")
#
#         # 进行拟合分析
#         perform_fitting_analysis(combined_groups.to_dict('records'), timestamp, axis)
#
#         return combined_groups
#     else:
#         print("未找到有效的组统计数据")
#         return None

def collect_all_step_data(axis=None):
    """收集所有已测试步长的数据并汇总（稳健版本，包含自动步长识别 + 强健错误处理）
       每个 (axis, step_size) 分组中，按 group_id 排序：
       - 前 8 条标记为训练：split='train'
       - 后面 2 条标记为验证：split='val'
    """

    import re
    import os
    import pandas as pd
    from datetime import datetime

    # 定义轴文件夹映射
    axis_folders = {'P': 'p', 'T': 't', 'Z': 'z'}

    all_group_stats = []
    searched_folders = []

    # 选择要扫描的轴
    if axis:
        if axis in axis_folders:
            folders_to_search = [axis_folders[axis]]
        else:
            print(f"错误: 无效的轴名称 {axis}")
            return None
    else:
        folders_to_search = list(axis_folders.values())

    # 遍历每个轴文件夹
    for axis_folder in folders_to_search:
        if not os.path.exists(axis_folder):
            print(f"警告: 轴文件夹 {axis_folder} 不存在")
            continue

        print(f"\n在文件夹 {axis_folder} 中查找测试数据...")

        # 根据轴判断文件夹命名规则
        if axis_folder == 'p':
            test_folders = [f for f in os.listdir(axis_folder)
                            if f.startswith('ptz_test_data_P_step_') and
                            os.path.isdir(os.path.join(axis_folder, f))]
        elif axis_folder == 't':
            test_folders = [f for f in os.listdir(axis_folder)
                            if f.startswith('ptz_test_data_T_step_') and
                            os.path.isdir(os.path.join(axis_folder, f))]
        elif axis_folder == 'z':
            test_folders = [f for f in os.listdir(axis_folder)
                            if f.startswith('ptz_test_data_Z_step_') and
                            os.path.isdir(os.path.join(axis_folder, f))]

        if not test_folders:
            print(f"  在 {axis_folder} 中未找到测试数据")
            continue

        print(f"  找到 {len(test_folders)} 个测试文件夹")

        # 遍历每个 step 测试文件夹
        for test_folder in test_folders:
            try:
                full_folder_path = os.path.join(axis_folder, test_folder)
                searched_folders.append(full_folder_path)

                # ----------- 自动解析步长（正则匹配 step_数字） -----------
                match = re.search(r'step_([0-9.]+)', test_folder)
                if not match:
                    print(f"警告：无法从文件夹名中提取步长: {test_folder}")
                    continue

                step_size = float(match.group(1))

                # 找到轴类型
                axis_from_folder = next((k for k, v in axis_folders.items() if v == axis_folder), None)

                print(f"  处理: {test_folder}, 轴: {axis_from_folder}, 步长: {step_size}")

                # 读取 group_stats.xlsx
                group_file = os.path.join(full_folder_path, "group_stats.xlsx")
                if os.path.exists(group_file):
                    group_df = pd.read_excel(group_file)

                    # 写入轴信息
                    group_df['axis'] = axis_from_folder

                    # 写入步长
                    group_df['step_size'] = step_size

                    all_group_stats.append(group_df)
                    print(f"    ✓ 成功读取数据")
                else:
                    print(f"    ✗ 未找到 group_stats.xlsx 文件")

            except Exception as e:
                print(f"读取文件夹 {test_folder} 数据时出错: {e}")
                import traceback
                traceback.print_exc()

    # ------------------ 合并所有 group_stats -------------------
    if not all_group_stats:
        print("\n未找到任何有效的组统计数据")
        print("请检查 p/t/z 文件夹结构、文件名是否符合 step_xxx 格式")
        return None

    combined_groups = pd.concat(all_group_stats, ignore_index=True)

    # 排序（先保持原来的排序逻辑）
    sort_columns = ['axis', 'step_size', 'group_id']
    combined_groups = combined_groups.sort_values(
        [col for col in sort_columns if col in combined_groups.columns]
    )

    # ------------------ 每个 (axis, step_size) 做 8/2 划分 -------------------
    if 'group_id' not in combined_groups.columns:
        print("警告：group_id 列不存在，无法严格做到“前 8 / 后 2”，将全部当作训练数据。")
        combined_groups['split'] = 'train'
    else:
        def mark_train_val(group):
            # group: 同一个 axis + step_size 下的一组数据
            group = group.sort_values('group_id').copy()
            n = len(group)

            if n != 10:
                print(f"  [警告] 轴={group['axis'].iloc[0]}, step={group['step_size'].iloc[0]} "
                      f"的样本数为 {n}，不是 10，仍按前 8 / 后其余划分。")

            train_n = min(8, n)   # 至多 8 条做训练
            group['split'] = 'train'
            if n > train_n:
                # 后面的全部标记为 val
                group.loc[group.index[train_n:], 'split'] = 'val'
            return group

        combined_groups = combined_groups.groupby(
            ['axis', 'step_size'], as_index=False, group_keys=False
        ).apply(mark_train_val)

    # 打印一下训练 / 验证的数量
    print("\n训练/验证划分统计：")
    print(combined_groups['split'].value_counts())

    # 保存结果
    output_dir = "ptz_fitting_res"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    axis_suffix = f"_{axis}" if axis else "_all"
    combined_filename = os.path.join(output_dir, f"combined_group_stats{axis_suffix}_{timestamp}.xlsx")
    combined_groups.to_excel(combined_filename, index=False)

    # ------------------ 输出日志 ------------------
    print("\n" + "=" * 60)
    print(f"数据汇总完成!")
    print(f"搜索的文件夹: {len(searched_folders)} 个")
    print(f"成功读取的数据文件: {len(all_group_stats)} 个")
    print(f"总数据条数: {len(combined_groups)}")
    print(f"包含的轴: {sorted(combined_groups['axis'].unique())}")

    for ax in combined_groups['axis'].unique():
        ax_data = combined_groups[combined_groups['axis'] == ax]
        step_sizes = ax_data['step_size'].unique()
        print(f"  {ax}方向: {len(ax_data)} 条数据, {len(step_sizes)} 个步长")
        if len(step_sizes) > 0:
            print(f"    步长范围: {min(step_sizes):.4f}° - {max(step_sizes):.4f}°")

    print(f"合并文件: {combined_filename}")
    print("=" * 60)

    # ------------------ 启动后续拟合+验证分析 ------------------
    perform_fitting_analysis(combined_groups.to_dict('records'), timestamp, axis)
    return combined_groups




# def perform_fitting_analysis(all_results, timestamp, axis=None):
#     """Perform power-law fitting analysis and save outputs into ptz_fitting_res/"""
#
#     import numpy as np
#     import matplotlib.pyplot as plt
#     from scipy.optimize import curve_fit
#     import pandas as pd
#     import os
#
#     # -------------------------
#     #  Prepare output directory
#     # -------------------------
#     output_dir = "ptz_fitting_res"
#     os.makedirs(output_dir, exist_ok=True)
#
#     if not all_results:
#         print("No data available for fitting analysis")
#         return
#
#     # Filter by axis
#     if axis:
#         filtered_results = [r for r in all_results if r['axis'] == axis]
#         axis_name = {'P': 'P', 'T': 'T', 'Z': 'Z'}[axis]
#     else:
#         filtered_results = all_results
#         axis_name = "All"
#
#     if not filtered_results:
#         print(f"No data available for {axis_name} axis")
#         return
#
#     # Prepare arrays
#     x = np.array([float(r['step_size']) for r in filtered_results])
#     y = np.array([float(r['bidirectional_avg_time']) for r in filtered_results])
#
#     print(f"Data statistics: N={len(x)} | x=[{x.min():.1f}, {x.max():.1f}] | y=[{y.min():.4f}, {y.max():.4f}]")
#
#     # Prepare figure
#     fig, ax = plt.subplots(figsize=(12, 8))
#
#     # ---- Color palette ----
#     colors = {
#         "scatter": "#4C72B0",
#         "power": "#C44E52",
#     }
#
#     # Scatter plot
#     ax.scatter(
#         x, y,
#         s=55, alpha=0.85,
#         color=colors["scatter"],
#         edgecolors="white",
#         linewidth=0.7,
#         label="Measured Data"
#     )
#
#     # R² helper
#     ss_tot = np.sum((y - np.mean(y)) ** 2)
#     fits = []
#
#     # ---------------------------------------------------------
#     # POWER MODEL v = a * x^b
#     # ---------------------------------------------------------
#     def power_model(x, a, b):
#         return a * x**b
#
#     try:
#         power_params, _ = curve_fit(power_model, x, y, p0=[0.01, 0.8], maxfev=8000)
#         a_power, b_power = power_params
#
#         y_pow = power_model(x, a_power, b_power)
#         r2_pow = 1 - np.sum((y - y_pow) ** 2) / ss_tot
#
#         ax.plot(x, y_pow, color=colors["power"], linewidth=2.5, linestyle="--",
#                 label=f'Power: y={a_power:.4f}x^{b_power:.4f} (R²={r2_pow:.4f})')
#
#         fits.append(("Power", r2_pow, (a_power, b_power)))
#     except Exception as e:
#         print("Power fit failed:", e)
#         r2_pow = 0
#
#     # ---------------------------------------------------------
#     # BEAUTIFY PLOT
#     # ---------------------------------------------------------
#     ax.set_xlabel('Step Size (°)', fontsize=15, fontweight='bold')
#     ax.set_ylabel('Movement Time (s)', fontsize=15, fontweight='bold')
#     ax.set_title(f'PTZ {axis_name} Axis — Movement Time vs Step Size',
#                  fontsize=19, fontweight='bold', pad=20)
#
#     ax.grid(True, linestyle="--", alpha=0.25)
#     ax.legend(fontsize=10, frameon=True, loc="upper left")
#
#     plt.tight_layout()
#
#     # Save figure
#     pic_name = f"{output_dir}/ptz_fitting_{axis_name}_{timestamp}.png"
#     plt.savefig(pic_name, dpi=350, bbox_inches="tight")
#     plt.close()
#
#     print(f"\nSaved figure → {pic_name}")
#
#     # ---------------------------------------------------------
#     # EXPORT FITTING RESULTS TO CSV
#     # ---------------------------------------------------------
#     df_out = pd.DataFrame([
#         {
#             "model": m,
#             "R2": r2,
#             "params": str(p)
#         }
#         for m, r2, p in fits
#     ])
#
#     csv_name = f"{output_dir}/ptz_fitting_table_{axis_name}_{timestamp}.csv"
#     df_out.to_csv(csv_name, index=False, encoding="utf-8-sig")
#
#     print(f"Saved fitting table → {csv_name}")
#
#     # ---------------------------------------------------------
#     # Return best model
#     # ---------------------------------------------------------
#     fits_sorted = sorted(fits, key=lambda x: x[1], reverse=True)
#     best_fit = fits_sorted[0] if fits_sorted else None
#     if best_fit:
#         print("\n=== FITTING SUMMARY ===")
#         for f in fits_sorted:
#             print(f"{f[0]}  →  R²={f[1]:.4f}")
#         print(f"\nRecommended best model → {best_fit[0]}")
#
#     return {
#         "best_fit": best_fit,
#         "all_fits": fits_sorted,
#         "picture": pic_name,
#         "table": csv_name
#     }

def perform_fitting_analysis(all_results, timestamp, axis=None):
    """Power-law + constant fitting with calibration/validation split.

    Model: y = a * x^b + c
    - Calibration data: split != 'val' (typically first 8 per step)
    - Validation data:  split == 'val' (typically last 2 per step)
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    import os
    from scipy.optimize import curve_fit

    output_dir = "ptz_fitting_res"
    os.makedirs(output_dir, exist_ok=True)

    if not all_results:
        print("No data available for fitting analysis")
        return

    # -------------------------
    #  Filter by axis
    # -------------------------
    if axis:
        filtered_results = [r for r in all_results if r.get('axis') == axis]
        axis_name = axis
    else:
        filtered_results = all_results
        axis_name = "All"

    if not filtered_results:
        print(f"No data available for axis={axis_name}")
        return

    df = pd.DataFrame(filtered_results)

    # target column: prefer bidirectional_avg_time
    if 'bidirectional_avg_time' in df.columns:
        y_col = 'bidirectional_avg_time'
    elif 'forward_avg_time' in df.columns:
        y_col = 'forward_avg_time'
    else:
        raise ValueError("Missing time column: expected 'bidirectional_avg_time' or 'forward_avg_time'")

    # -------------------------
    #  calibration / validation split
    # -------------------------
    if 'split' in df.columns:
        train_df = df[df['split'] != 'val'].copy()
        val_df   = df[df['split'] == 'val'].copy()
    else:
        print("Warning: 'split' column not found. Using all data as calibration, no validation analysis.")
        train_df = df.copy()
        val_df   = df.iloc[0:0].copy()

    if train_df.empty:
        print("No calibration data available for fitting.")
        return

    # calibration data (前8条已经在 collect_all_step_data 里选好了)
    x_train = train_df['step_size'].astype(float).to_numpy()
    y_train = train_df[y_col].astype(float).to_numpy()

    # -------------------------
    #  model: y = a * x^b + c
    # -------------------------
    def power_model_with_const(x, a, b, c):
        return a * np.power(x, b) + c

    # initial guess
    x_mean = max(np.mean(x_train), 1e-6)
    y_mean = np.mean(y_train)
    c0 = np.min(y_train)
    a0 = (y_mean - c0) / x_mean if x_mean > 0 else 1.0
    p0 = [a0, 1.0, c0]

    # fit on calibration data
    popt, pcov = curve_fit(power_model_with_const, x_train, y_train, p0=p0, maxfev=20000)
    a_fit, b_fit, c_fit = popt

    # ---- R^2 on calibration data ----
    y_train_pred = power_model_with_const(x_train, a_fit, b_fit, c_fit)
    ss_res = np.sum((y_train - y_train_pred) ** 2)
    ss_tot = np.sum((y_train - np.mean(y_train)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    print(f"\nAxis = {axis_name}")
    # 拟合参数打印为 3 位小数
    print(f"  Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}, c = {c_fit:.3f}")
    print(f"  R^2 (coefficient of determination, calibration data) = {r2:.2f}")

    # smooth curve for plotting
    x_plot = np.linspace(x_train.min(), x_train.max(), 200)
    y_plot = power_model_with_const(x_plot, a_fit, b_fit, c_fit)

    # -------------------------
    #  plot: blue (calibration) + orange (validation) + red (fit)
    # -------------------------
    fig, ax = plt.subplots(figsize=(6, 5.5))

    # calibration data
    ax.scatter(
        x_train, y_train,
        color='blue',
        label='Calibration data',
        alpha=0.7
    )

    # validation data
    if not val_df.empty:
        x_val_plot = val_df['step_size'].astype(float).to_numpy()
        y_val_plot = val_df[y_col].astype(float).to_numpy()
        ax.scatter(
            x_val_plot, y_val_plot,
            color='orange',
            edgecolors='black',
            label='Validation data',
            alpha=0.9
        )

    # 拟合曲线；图例里用简短文本 + 3 位小数，R² 换行
    fit_label = (
        f"Model fit: y = {a_fit:.3f} x^{b_fit:.3f} + {c_fit:.3f}\n"
        f"(R² = {r2:.2f})"
    )
    ax.plot(
        x_plot, y_plot,
        color='red',
        linewidth=2.0,
        label=fit_label
    )

    # axis labels（字体再大两号）
    ax.set_xlabel('Step size (degrees)', fontsize=16)
    if y_col == 'bidirectional_avg_time':
        ax.set_ylabel('Bidirectional PTZ motion time (s)', fontsize=16)
    else:
        ax.set_ylabel('Forward PTZ motion time (s)', fontsize=16)

    # 简洁学术标题；字体加大一点
    ax.set_title(f'PTZ motion-time fitting (axis = {axis_name})',
                 fontsize=16, pad=16)

    # 给数据一点边距
    ax.margins(x=0.03, y=0.08)

    # 刻度字体加大
    ax.tick_params(axis='both', labelsize=14)

    ax.grid(True, linestyle='--', alpha=0.4)

    # 图例左上角；字体也加两号
    ax.legend(fontsize=13, loc='upper left', frameon=True)

    fig.tight_layout()

    # 保存为 PDF，文件名不含时间戳
    fig_name = os.path.join(output_dir, f"fit_axis_{axis_name}.pdf")
    fig.savefig(fig_name, bbox_inches='tight')
    plt.close(fig)
    print(f"Figure saved to: {fig_name}")
    # -------------------------------------------------------------------------

    # -------------------------
    #  validation error analysis
    # -------------------------
    if not val_df.empty:
        x_val = val_df['step_size'].astype(float).to_numpy()
        y_val = val_df[y_col].astype(float).to_numpy()
        y_val_pred = power_model_with_const(x_val, a_fit, b_fit, c_fit)

        abs_err = np.abs(y_val - y_val_pred)
        rel_err = abs_err / np.maximum(y_val, 1e-9)

        val_df = val_df.copy()
        val_df['y_pred']    = y_val_pred
        val_df['abs_error'] = abs_err
        val_df['rel_error'] = rel_err

        mae  = abs_err.mean()
        rmse = np.sqrt((abs_err ** 2).mean())
        mape = (np.abs(rel_err).mean() * 100.0)

        print("\n=== Validation error (held-out last 2 per step) ===")
        print(f"Number of validation points: {len(val_df)}")
        print(f"MAE  (mean absolute error): {mae:.6f}")
        print(f"RMSE (root mean squared error): {rmse:.6f}")
        print(f"MAPE (mean absolute percentage error): {mape:.2f}%")

        val_filename = os.path.join(output_dir, f"validation_errors_axis_{axis_name}_{timestamp}.xlsx")
        val_df.to_excel(val_filename, index=False)
        print(f"Validation details saved to: {val_filename}")
    else:
        print("\nNo validation data (split == 'val') found; skipping validation analysis.")

    # -------------------------
    #  save parameters (including R^2)
    # -------------------------
    param_file = os.path.join(output_dir, f"fit_params_{axis_name}.csv")
    pd.DataFrame(
        [{
            'axis': axis_name,
            'a': a_fit,
            'b': b_fit,
            'c': c_fit,
            'R2_calibration': r2,
            'timestamp': timestamp
        }]
    ).to_csv(param_file, index=False)
    print(f"Fitting parameters saved to: {param_file}")










def generate_random_step_sizes(start_range, end_range, points_per_interval=2, decimal_places=1):
    """
    生成随机步长（确保不重复）

    参数:
        start_range: 起始范围
        end_range: 结束范围
        points_per_interval: 每个整数间隔内取的点数
        decimal_places: 小数位数
    """
    step_sizes = set()  # 使用集合来避免重复
    max_attempts_per_interval = 50  # 每个间隔的最大尝试次数

    # 生成每个整数间隔内的随机点
    for integer_part in range(int(start_range), int(end_range)):
        attempts = 0
        points_generated = 0

        while points_generated < points_per_interval and attempts < max_attempts_per_interval:
            # 在 [integer_part, integer_part+1) 范围内生成随机数
            random_step = round(random.uniform(integer_part, integer_part + 1), decimal_places)

            # 检查是否重复
            if random_step not in step_sizes:
                step_sizes.add(random_step)
                points_generated += 1

            attempts += 1

        # 如果无法生成足够的唯一点，使用均匀分布的点
        if points_generated < points_per_interval:
            remaining_points = points_per_interval - points_generated
            interval_points = np.linspace(integer_part, integer_part + 1, remaining_points + 2)[1:-1]
            for point in interval_points:
                unique_point = round(point, decimal_places)
                if unique_point not in step_sizes:
                    step_sizes.add(unique_point)

    # 转换为列表并排序
    step_sizes = sorted(list(step_sizes))

    return step_sizes

def test_random_step_sizes(axis='P'):
    """测试随机步长"""
    axis_name = {'P': 'P方向', 'T': 'T方向', 'Z': 'Z方向'}[axis]
    print(f"\n{axis_name}随机步长测试")
    print("=" * 50)

    try:
        start_range = float(input("请输入起始范围: "))
        end_range = float(input("请输入结束范围: "))
        points_per_interval = int(input("请输入每个间隔内的点数 (如 2): "))

        # 生成随机步长
        step_sizes = generate_random_step_sizes(start_range, end_range, points_per_interval)

        print(f"\n生成的随机步长: {step_sizes}")
        print(f"总共 {len(step_sizes)} 个步长")

        confirm = input("确定开始测试吗? (y/n): ").lower()
        if confirm == 'y':
            for i, step_size in enumerate(step_sizes, 1):
                test_single_step_size(step_size, axis, i, len(step_sizes))
                if i < len(step_sizes):
                    wait_time = 10  # 步长间等待10秒
                    print(f"\n等待 {wait_time} 秒后测试下一个步长...")
                    time.sleep(wait_time)

            print(f"\n所有{axis_name}随机步长测试完成!")

    except ValueError as e:
        print(f"输入错误: {e}")
    except Exception as e:
        print(f"测试过程中出现错误: {e}")


# 计算速度函数用时 与 实际用时
# ============ 6. 时间代价函数(可根据需求自定义) =============
def _axis_motion_time(delta, a, b, c):
    delta = abs(float(delta))
    if delta < 1e-12:
        return 0.0
    return a * math.pow(delta, b) + c


def t_x(x):
    return _axis_motion_time(x, 0.106, 0.555, 0.219)


def t_y(y):
    return _axis_motion_time(y, 0.060, 0.753, 0.233)


def t_z(z):
    return _axis_motion_time(z, 0.034, 1.371, 0.382)


def stay_time(dx, dy, dz):
    if abs(dx) < 1e-12 and abs(dy) < 1e-12 and abs(dz) < 1e-12:
        return 0.0
    return 0.323 + 0.015 * abs(dz)

def travel_time(pA, pB):
    """
    从 pA=(x1,y1,z1) 到 pB=(x2,y2,z2) 的时间代价:
        max(t_x(|x2-x1|), t_y(|y2-y1|), t_z(|z2-z1|)) + stay_t(z2)
    可按需修改/扩展。
    """
    dx = abs(pB[0] - pA[0])
    dy = abs(pB[1] - pA[1])
    dz = abs(pB[2] - pA[2])
    return max(t_x(dx), t_y(dy), t_z(dz)) + stay_time(dx,dy,dz)
    # return max(t_x(dx), t_y(dy), t_z(dz))

def get_path_time_cost(pts, closed):
    total_cost = 0.0
    for i in range(1, len(pts)):
        total_cost += travel_time(pts[i - 1], pts[i])
    if closed and len(pts) > 1:
        total_cost += travel_time(pts[-1], pts[0])
    return total_cost

import random
import time
import statistics

# =======================================================
# 随机生成合法 PTZ 点（带差值限制：P<55 T<35 Z<23）
# =======================================================
def random_ptz_point():
    p = round(random.uniform(0, 360), 1)
    t = round(random.uniform(-5, 90), 1)
    z = round(random.uniform(1, 23), 1)
    return (p, t, z)

def random_ptz_pair():
    """生成两个点，保证差值限制：P<55，T<35，Z<23"""
    pA, tA, zA = random_ptz_point()

    pB = round(random.uniform(max(0,   pA - 55), min(360, pA + 55)), 1)
    tB = round(random.uniform(max(-5,  tA - 35), min(90,  tA + 35)), 1)
    zB = round(random.uniform(max(1,   zA - 23), min(23,  zA + 23)), 1)

    return (pA, tA, zA), (pB, tB, zB)


# =======================================================
# 实际测试一次 + 理论模型时间
# =======================================================
# def get_time_cost(start_position, end_position):
#     success_start, _ = new_gotopos_sdk_sync_e_with_focus(start_position)
#     if not success_start:
#         return -1, -1
#     time.sleep(0.05)
#
#     success, movement_time = new_gotopos_sdk_sync_e_with_focus(end_position)
#
#     if not success:
#         return -1, -1
#
#     theoretical_time = travel_time(start_position, end_position)
#     return movement_time, theoretical_time


# =======================================================
# 主测试函数：随机 A→B 测试并记录
# =======================================================
# def random_AB_test(num_tests=10):
#     results = []
#     records = []
#
#     print(f"\n========== 开始随机测试（共 {num_tests} 次） ==========")
#
#     for i in range(1, num_tests + 1):
#         pA, pB = random_ptz_pair()
#         # pA = random_ptz_point()
#         # pB = random_ptz_point()
#
#         actual_t, theory_t = get_time_cost(pA, pB)
#         if actual_t == -1:
#             print("⚠️ 测试失败，跳过此点")
#             continue
#
#         error_percent = abs(actual_t - theory_t) / theory_t * 100
#
#         # ============================
#         # 🔥 保存测试数据（A、B 放一列）
#         # ============================
#         records.append({
#             "Test_ID": i,
#             "Start_PTZ": f"({pA[0]:.1f}, {pA[1]:.1f}, {pA[2]:.1f})",
#             "End_PTZ":   f"({pB[0]:.1f}, {pB[1]:.1f}, {pB[2]:.1f})",
#             "Delta_P": abs(pA[0] - pB[0]),
#             "Delta_T": abs(pA[1] - pB[1]),
#             "Delta_Z": abs(pA[2] - pB[2]),
#             "Actual_Time": actual_t,
#             "Theory_Time": theory_t,
#             "Error_s": abs(actual_t - theory_t),
#             "Error_%": error_percent
#         })
#
#         results.append(error_percent)
#         print(f"测试 {i}/{num_tests} 完成  误差 = {error_percent:.2f}%")
#
#     # =======================================================
#     # 汇总统计 + 保存 Excel
#     # =======================================================
#     if results:
#         avg_error = statistics.mean(results)
#         max_error = max(results)
#         min_error = min(results)
#
#         save_random_test_results(records, avg_error, max_error, min_error)
#
#         return avg_error, results
#     else:
#         print("❌ 所有测试均失败")
#         return None, None


# =======================================================
# 保存结果到 Excel (res/ 目录)
# =======================================================
def save_random_test_results(records, avg_err, max_err, min_err):
    os.makedirs("ptz_fitting_res", exist_ok=True)

    df = pd.DataFrame(records)
    summary_df = pd.DataFrame({
        "Metric": ["Avg_Error_%", "Max_Error_%", "Min_Error_%"],
        "Value": [avg_err, max_err, min_err]
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ptz_fitting_res/random_AB_results_{timestamp}.xlsx"

    with pd.ExcelWriter(filename) as writer:
        df.to_excel(writer, sheet_name="Test_Records", index=False)
        summary_df.to_excel(writer, sheet_name="Summary", index=False)

    print(f"\n📁 数据已保存至: {filename}")

def main():
    """主函数 - 提供测试选项"""
    print("PTZ 移动用时测试系统")
    print("=" * 50)
    print("1. 测试P方向")
    print("2. 测试T方向")
    print("3. 测试Z方向")
    print("4. 汇总所有已测试数据")
    print("5. 测试理论用时与实际用时")
    print("6. 退出")

    while True:
        choice = input("\n请选择操作 (1-5): ").strip()

        if choice == '1':
            # 测试P方向
            print("\nP方向测试选项:")
            print("1. 测试单个步长")
            print("2. 测试多个步长")
            print("3. 测试随机步长")

            sub_choice = input("请选择P方向测试方式 (1-3): ").strip()

            if sub_choice == '1':
                try:
                    step_size = float(input("请输入要测试的步长 (如 0.1, 1.0, 5.0): "))
                    print(f"\n开始测试P方向步长 {step_size:.1f}°...")
                    test_single_step_size(step_size, 'P')
                    print(f"\nP方向步长 {step_size:.1f}° 测试完成!")
                except ValueError:
                    print("无效的步长输入!")

            elif sub_choice == '2':
                try:
                    start_step = float(input("请输入起始步长: "))
                    end_step = float(input("请输入结束步长: "))
                    step_interval = float(input("请输入步长间隔: "))

                    step_sizes = []
                    current = start_step
                    while current <= end_step:
                        step_sizes.append(round(current, 1))
                        current += step_interval

                    total_steps = len(step_sizes)
                    print(f"\n将测试以下P方向步长: {step_sizes}")
                    print(f"总共 {total_steps} 个步长")

                    confirm = input("确定开始测试吗? (y/n): ").lower()
                    if confirm == 'y':
                        for i, step_size in enumerate(step_sizes, 1):
                            test_single_step_size(step_size, 'P', i, total_steps)
                            if i < total_steps:
                                wait_time = 10
                                print(f"\n等待 {wait_time} 秒后测试下一个步长...")
                                time.sleep(wait_time)
                except ValueError:
                    print("无效的输入!")

            elif sub_choice == '3':
                test_random_step_sizes('P')
            else:
                print("无效选择!")

        elif choice == '2':
            # 测试T方向
            print("\nT方向测试选项:")
            print("1. 测试单个步长")
            print("2. 测试多个步长")
            print("3. 测试随机步长")

            sub_choice = input("请选择T方向测试方式 (1-3): ").strip()

            if sub_choice == '1':
                try:
                    step_size = float(input("请输入要测试的步长 (如 0.1, 1.0, 5.0): "))
                    print(f"\n开始测试T方向步长 {step_size:.1f}°...")
                    test_single_step_size(step_size, 'T')
                    print(f"\nT方向步长 {step_size:.1f}° 测试完成!")
                except ValueError:
                    print("无效的步长输入!")

            elif sub_choice == '2':
                try:
                    start_step = float(input("请输入起始步长: "))
                    end_step = float(input("请输入结束步长: "))
                    step_interval = float(input("请输入步长间隔: "))

                    step_sizes = []
                    current = start_step
                    while current <= end_step:
                        step_sizes.append(round(current, 1))
                        current += step_interval

                    total_steps = len(step_sizes)
                    print(f"\n将测试以下T方向步长: {step_sizes}")
                    print(f"总共 {total_steps} 个步长")

                    confirm = input("确定开始测试吗? (y/n): ").lower()
                    if confirm == 'y':
                        for i, step_size in enumerate(step_sizes, 1):
                            test_single_step_size(step_size, 'T', i, total_steps)
                            if i < total_steps:
                                wait_time = 10
                                print(f"\n等待 {wait_time} 秒后测试下一个步长...")
                                time.sleep(wait_time)
                except ValueError:
                    print("无效的输入!")

            elif sub_choice == '3':
                test_random_step_sizes('T')
            else:
                print("无效选择!")

        elif choice == '3':
            # 测试Z方向
            print("\nZ方向测试选项:")
            print("1. 测试单个步长")
            print("2. 测试多个步长")
            print("3. 测试随机步长")

            sub_choice = input("请选择Z方向测试方式 (1-3): ").strip()

            if sub_choice == '1':
                try:
                    step_size = float(input("请输入要测试的步长 (如 0.1, 1.0, 5.0): "))
                    print(f"\n开始测试Z方向步长 {step_size:.1f}°...")
                    test_single_step_size(step_size, 'Z')
                    print(f"\nZ方向步长 {step_size:.1f}° 测试完成!")
                except ValueError:
                    print("无效的步长输入!")

            elif sub_choice == '2':
                try:
                    start_step = float(input("请输入起始步长: "))
                    end_step = float(input("请输入结束步长: "))
                    step_interval = float(input("请输入步长间隔: "))

                    step_sizes = []
                    current = start_step
                    while current <= end_step:
                        step_sizes.append(round(current, 1))
                        current += step_interval

                    total_steps = len(step_sizes)
                    print(f"\n将测试以下Z方向步长: {step_sizes}")
                    print(f"总共 {total_steps} 个步长")

                    confirm = input("确定开始测试吗? (y/n): ").lower()
                    if confirm == 'y':
                        for i, step_size in enumerate(step_sizes, 1):
                            test_single_step_size(step_size, 'Z', i, total_steps)
                            if i < total_steps:
                                wait_time = 10
                                print(f"\n等待 {wait_time} 秒后测试下一个步长...")
                                time.sleep(wait_time)
                except ValueError:
                    print("无效的输入!")

            elif sub_choice == '3':
                test_random_step_sizes('Z')
            else:
                print("无效选择!")

        elif choice == '4':
            # 汇总所有数据
            print("\n数据汇总选项:")
            print("1. 汇总P方向数据")
            print("2. 汇总T方向数据")
            print("3. 汇总Z方向数据")
            print("4. 汇总所有方向数据")

            sub_choice = input("请选择汇总方式 (1-4): ").strip()

            if sub_choice == '1':
                print("\n正在汇总P方向测试数据...")
                collect_all_step_data('P')
            elif sub_choice == '2':
                print("\n正在汇总T方向测试数据...")
                collect_all_step_data('T')
            elif sub_choice == '3':
                print("\n正在汇总Z方向测试数据...")
                collect_all_step_data('Z')
            elif sub_choice == '4':
                print("\n正在汇总所有方向测试数据...")
                collect_all_step_data()
            else:
                print("无效选择!")

        elif choice == '5':
            num_tests = int(input("\n请输入测试点数量: ").strip())
            # random_AB_test(num_tests)

        elif choice == '6':
            print("退出系统")
            break

        else:
            print("无效选择，请重新输入!")


if __name__ == '__main__':

    main()
