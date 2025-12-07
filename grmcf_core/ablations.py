import os
import math
import random
import time
from itertools import combinations
from PIL import Image, ImageDraw, ImageFont
import heapq
import math
import multiprocessing
import random
import time
from itertools import combinations
import struct
import numpy as np
from mayavi import mlab
from PIL import Image, ImageDraw, ImageFont
import json
import requests
from scipy.spatial import Delaunay
import sys
import csv


# =========================
# 1) 共享判定与 PSR 计算
# =========================

def can_share(b1, b2, proportion, W, H, margin=0.95):
    """
    判定两目标框是否存在“共同变焦 + 合理平移(pan/tilt)”可以一视多拍。
    b = [x1, y1, x2, y2]
    proportion: 面积占比阈值（如 0.06 表示目标面积至少为整图的 6%）
    W,H: 画布宽高
    margin: 安全边距（避免刚好贴边导致溢出），建议 0.9~0.98
    """
    w1 = max(1, b1[2]-b1[0]); h1 = max(1, b1[3]-b1[1])
    w2 = max(1, b2[2]-b2[0]); h2 = max(1, b2[3]-b2[1])

    x1 = min(b1[0], b2[0]); y1 = min(b1[1], b2[1])
    x2 = max(b1[2], b2[2]); y2 = max(b1[3], b2[3])
    w12 = max(1, x2 - x1);  h12 = max(1, y2 - y1)

    A = W * H
    a1, a2 = w1*h1, w2*h2

    s1 = math.sqrt((proportion * A) / a1)
    s2 = math.sqrt((proportion * A) / a2)
    s_min = max(s1, s2)

    s_max = margin * min(W / w12, H / h12)

    return s_min <= s_max


def psr_exact(boxes, proportion, W, H, margin=0.95):
    N = len(boxes)
    if N < 2:
        return 0.0
    T = N*(N-1)//2
    S = 0
    for i in range(N):
        bi = boxes[i]
        for j in range(i+1, N):
            bj = boxes[j]
            if can_share(bi, bj, proportion, W, H, margin=margin):
                S += 1
    return S / T


def psr_bin_check(boxes, proportion, W, H, lo, hi, margin=0.95):
    """
    早停版 PSR 计算。
    返回: (accepted, psr_value)
    """
    N = len(boxes)
    if N < 2:
        v = 0.0
        return (lo <= v < hi), v

    T = N*(N-1)//2
    S_lo = int(math.ceil(lo * T))
    S_hi = int(math.floor(min(hi - 1e-12, 1.0) * T))

    S = 0
    checked = 0
    for i in range(N):
        bi = boxes[i]
        for j in range(i+1, N):
            bj = boxes[j]
            checked += 1

            if can_share(bi, bj, proportion, W, H, margin=margin):
                S += 1
                if S > S_hi:
                    v = S / T
                    return (False, v)

            remaining = T - checked
            if S + remaining < S_lo:
                v = S / T
                return (False, v)

    v = S / T
    return (lo <= v < hi), v


# =========================
# 2) 生成二维目标框（簇状采样）
# =========================

def _sample_boxes_clustered(
    N, W, H,
    min_wh=(30, 30), max_wh=(260, 220),
    sigma=50.0, cluster_size_range=(2, 4),
    max_trials_per_box=50,
    pixel_limit=None
):
    """
    簇状采样，新增硬性约束：
    - 若 pixel_limit 非 None 且 >0，则尽量保证每个框面积 w*h > pixel_limit。
    - 不抛异常：必要时自动放宽 pixel_limit 保证可行，并用兜底大框。
    """
    # 预处理 pixel_limit：保证“理论上可行”，不抛异常
    if pixel_limit is not None:
        pixel_limit = float(pixel_limit)
        if pixel_limit <= 0:
            pixel_limit = None
        else:
            max_possible_area = max_wh[0] * max_wh[1]
            # 自动裁剪到略小于可达到的最大面积，避免完全不可能
            if pixel_limit >= max_possible_area:
                pixel_limit = max_possible_area - 1.0

    boxes = []
    remaining = N
    while remaining > 0:
        cx = random.uniform(0.15 * W, 0.85 * W)
        cy = random.uniform(0.15 * H, 0.85 * H)
        m = min(remaining, random.randint(*cluster_size_range))

        for _ in range(m):
            placed = False

            # 先尝试高斯散点 + 随机 w,h
            for _try in range(max_trials_per_box):
                w = random.randint(min_wh[0], max_wh[0])
                h = random.randint(min_wh[1], max_wh[1])

                if pixel_limit is not None and (w * h) <= pixel_limit:
                    continue

                x_center = random.gauss(cx, sigma)
                y_center = random.gauss(cy, sigma)
                x1 = int(round(x_center - w / 2))
                y1 = int(round(y_center - h / 2))
                x1 = max(0, min(W - w, x1))
                y1 = max(0, min(H - h, y1))
                x2, y2 = x1 + w, y1 + h

                boxes.append([x1, y1, x2, y2])
                placed = True
                break

            if not placed:
                # 兜底：用一个“最大框”来保证面积约束，同时仍在画布内
                w = max_wh[0]
                h = max_wh[1]
                # 理论上 w*h > pixel_limit（因为我们裁剪了 pixel_limit），这里一般都满足
                # 就算 pixel_limit 为 None 也没关系
                if w > W:
                    w = W
                if h > H:
                    h = H
                x1 = random.randint(0, max(0, W - w))
                y1 = random.randint(0, max(0, H - h))
                x2, y2 = x1 + w, y1 + h
                boxes.append([x1, y1, x2, y2])

        remaining -= m

    return boxes


# =========================
# 3) 引导式寻参：命中指定 PSR 档位
# =========================

def _generate_group_with_target_psr_guided(
    N, W, H, proportion,
    target_range,        # (lo, hi)
    seed_sigma,          # 初始 sigma
    cluster_size_range,  # 初始簇大小范围
    min_wh=(30,30), max_wh=(260,220),
    tries=200, margin=0.95,
    max_seconds=3.0,
    slack=0.0,
    hard_cap=3000,
    strict=True,
    pixel_limit=None
):
    """
    自适应调 sigma/簇大小，尽量命中指定 PSR 档。
    不抛异常，生成失败时返回 (None, None)，由上层继续调参重试。
    """
    lo, hi = target_range
    best = None  # (abs_err, boxes, v)

    sigma = float(seed_sigma)
    cs_min, cs_max = cluster_size_range
    t0 = time.perf_counter()
    attempts = 0

    while attempts < hard_cap:
        attempts += 1

        if not strict and (time.perf_counter() - t0) > max_seconds:
            break

        boxes = _sample_boxes_clustered(
            N, W, H,
            min_wh=min_wh, max_wh=max_wh,
            sigma=sigma, cluster_size_range=(cs_min, cs_max),
            pixel_limit=pixel_limit
        )

        accepted, v = psr_bin_check(boxes, proportion, W, H, lo, hi, margin=margin)
        if accepted:
            return boxes, v

        if not strict:
            if v < lo:
                err = lo - v
            elif v >= hi:
                err = v - hi
            else:
                err = 0.0

            if (best is None) or (err < best[0]):
                best = (err, boxes, v)

            if err <= slack:
                return boxes, v

        if v < lo:
            sigma = max(5.0, sigma * 0.72)
            cs_max = min(cs_max + 1, max(3, N//8))
        else:
            sigma = min(0.6*min(W,H), sigma * 1.28)
            if cs_max - cs_min > 0:
                cs_max = max(cs_min, cs_max - 1)

        if attempts % 20 == 0:
            sigma = min(0.6*min(W,H), max(5.0, sigma * (1.6 if v >= hi else 0.55)))

    if not strict and best is not None:
        return best[1], best[2]
    return None, None


# =========================
# 4) 主入口：按档位批量生成
# =========================

def generate_datasets_by_bins(
    N=30, W=2560, H=1440,
    proportion=0.06,
    groups_per_bin=20,
    seed=None,
    margin=0.95,
    psr_bins=None,
    max_seconds_per_group=3.0,
    slack=0.0,
    strict=True,
    pixel_limit=2000
):
    """
    返回 outputs = {'low':[], 'med':[], 'high':[]}
    每个元素是 (boxes, psr_value)

    pixel_limit:
        - None: 不限制框面积；
        - >0: 目标框尽量满足 w*h > pixel_limit，内部自动裁剪到可行范围，绝不抛异常。
    """
    if seed is not None:
        random.seed(seed)

    if psr_bins is None:
        psr_bins = {
            'low':  (0.00, 0.10),
            'med':  (0.10, 0.30),
            'high': (0.30, 1.01),
        }

    base = float(min(W, H))
    bin_cfg = {
        'low':  {'sigma': 0.18*base, 'cs': (1, 2), 'min_wh': (25,25), 'max_wh': (180,150)},
        'med':  {'sigma': 0.10*base, 'cs': (2, 4), 'min_wh': (40,40), 'max_wh': (220,190)},
        'high': {'sigma': 0.04*base, 'cs': (4, 8), 'min_wh': (60,60), 'max_wh': (300,240)},
    }

    outputs = {'low': [], 'med': [], 'high': []}

    for name in ['low', 'med', 'high']:
        lo, hi = psr_bins[name]
        seed_sigma = bin_cfg[name]['sigma']
        cs_range   = bin_cfg[name]['cs']
        min_wh     = bin_cfg[name]['min_wh']
        max_wh     = bin_cfg[name]['max_wh']

        while len(outputs[name]) < groups_per_bin:
            boxes, v = _generate_group_with_target_psr_guided(
                N=N, W=W, H=H, proportion=proportion,
                target_range=(lo, hi),
                seed_sigma=seed_sigma,
                cluster_size_range=cs_range,
                min_wh=min_wh, max_wh=max_wh,
                tries=300, margin=margin,
                max_seconds=max_seconds_per_group,
                slack=slack, hard_cap=3000,
                strict=strict,
                pixel_limit=pixel_limit
            )
            if boxes is None:
                # 这一步不会抛异常，只说明当前参数不好命中；放宽 sigma 再试
                seed_sigma *= 1.15
                continue
            outputs[name].append((boxes, v))

    return outputs



# =========================
# 5) 结果保存（绘图+坐标）
# =========================

def save_outputs(outputs, N, outdir="datasets", canvas_width=2560, canvas_height=1440, show_psr=True):
    """
    将 generate_datasets_by_bins 返回的 outputs = {'low':[], 'med':[], 'high':[]}
    绘制并保存到磁盘：
      {N}_{bin}_{idx}_boxs.png
      {N}_{bin}_{idx}_boxs-boxs.txt
    """
    os.makedirs(outdir, exist_ok=True)

    # 准备字体
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    # 固定三档顺序
    for bin_name in ["low", "med", "high"]:
        groups = outputs.get(bin_name, [])
        for idx, item in enumerate(groups, start=1):
            # 兼容两种格式：(boxes, psr) 或 仅 boxes
            if isinstance(item, tuple) and len(item) >= 1:
                boxes = item[0]
                psr_val = item[1] if len(item) > 1 else None
            else:
                boxes = item
                psr_val = None

            base = f"{N}_{bin_name}_{idx}_boxs"
            png_path = os.path.join(outdir, f"{base}.png")
            txt_path = os.path.join(outdir, f"{base}-boxs.txt")

            # 画布
            image = Image.new("RGB", (canvas_width, canvas_height), "white")
            draw = ImageDraw.Draw(image)

            # 可选：在左上角写 PSR
            if show_psr and psr_val is not None:
                draw.text((10, 10), f"PSR={psr_val:.3f}", fill=(0,0,0), font=font)

            # 绘制每个框
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box
                color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
                draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2
                draw.text((cx, cy), str(i), fill=color, font=font)

            image.save(png_path)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(str(boxes))

            if psr_val is not None:
                print(f"Saved {png_path} (PSR={psr_val:.3f}) and {txt_path}")
            else:
                print(f"Saved {png_path} and {txt_path}")



def model_http(modeling_url,boxes,pos_now,proportion,n,s,canvas_width,canvas_height,z_max,va,p_offset=0,t_offset=0,mode=0):




    headers = {"Content-Type": "application/json"}

    # 发送给 C++ crow 接口的 payload
    payload = {
        "proportion": proportion,
        "n": n,
        "s": s,
        "canvas_width": canvas_width,
        "canvas_height": canvas_height,
        "pos_now": pos_now,
        "z_max": z_max,
        "va": va,
        "p_offset": p_offset,  # 如果后端暂时不用也没事
        "t_offset": t_offset,  # 同上
        "mode": mode,
        "boxes": boxes,
    }

    try:
        resp = requests.post(
            modeling_url,
            headers=headers,
            json=payload,  # 让 requests 帮你做 json.dumps
            timeout=10
        )

        print("HTTP status:", resp.status_code)

        # 打印原始返回文本
        # print("model_api response text:")
        # print(resp.text)

        # 如果是合法 JSON，就格式化打印
        try:
            resp_json = resp.json()
            print("建模时间(ms):",resp_json['t'])
            return resp_json
        except ValueError:
            print("\n返回的不是合法 JSON，无法 resp.json() 解析。")
            return None
    except requests.RequestException as e:
        print("请求出错:", e)
        return None

def t_x(x):
    """
    计算 x 方向的时间代价。
    """
    if x >= 0:
        return 0.4244954668441346 * math.pow(x, 0.26633709799724753)
    else:
        x2 = -x
        return 0.4244954668441346 * math.pow(x2, 0.26633709799724753)


def t_y(y):
    """
    计算 y 方向的时间代价。
    """
    if y >= 0:
        return 0.30136612962263964 * math.pow(y, 0.32298884814638495)
    else:
        y2 = -y
        return 0.30136612962263964 * math.pow(y2, 0.32298884814638495)


def t_z(z):
    """
    计算 z 方向的时间代价。
    """
    if z >= 0:
        return 0.3145807285048562 * math.pow(z, 0.790858763302901)
    else:
        z2 = -z
        return 0.3145807285048562 * math.pow(z2, 0.790858763302901)


def stay_time(dx, dy, dz):
    if dx == 0 and dy == 0 and dz == 0:
        return 0.0
    else:
        return 0.3 + max(0.0, abs(dz) - 1.0) / 10



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

def get_path_time_cost(pts, closed):
    total_cost = 0.0
    for i in range(1, len(pts)):
        total_cost += travel_time(pts[i - 1], pts[i])
    if closed and len(pts) > 1:
        total_cost += travel_time(pts[-1], pts[0])
    return total_cost

def generate_new_path(path, pos_now):
    # 转换为numpy数组方便处理
    path = np.array(path)

    # 查找pos_now在path中的索引
    idx = np.where(np.all(path == pos_now, axis=1))[0]

    if len(idx) == 0:
        raise ValueError("The point pos_now is not in the path!")

    # 只取第一个匹配点的索引（假设path中只有一个这样的点）
    idx = idx[0]

    # 生成新的path，从pos_now所在的点开始，确保不重复起点
    new_path = np.concatenate((path[idx + 1:], path[:idx]))

    return new_path.tolist()
def snapshot_gsoa_gs(t_l,str,url):
    """
    :param geometrics:所有几何体
    :param pos_now: ptz起点(当前ptz)
    :param proportion: 最小的放大比例
    :param fov_w: 当前视野的像素宽(画布宽)
    :param fov_h: 当前视野的像素长(画布长)
    :return:
    """



    if t_l > 1:

        # 将最终的 JSON 结构转换为字符串ls
        final_json_str = str
        # 发送 POST 请求到 API
        # api_url = "http://172.30.249.71:8888/api/gsoa"  # 本机的wsl2
        # api_url = "http://10.156.2.57:8888/api/gsoa"
        api_url = url
        headers = {"Content-Type": "application/json"}
        # 记录开始时间
        # start_time = time.time()
        response = requests.post(api_url, data=final_json_str, headers=headers)
        # 记录结束时间
        # end_time = time.time()
        # 计算运行时间（微秒）
        # elapsed_time = (end_time - start_time)
        # print("最终请求json:", final_json_str)
        # print("调用gsoa_api用时:", elapsed_time)
        # print("Response time:", response.elapsed.total_seconds(), "s")
        # 检查请求是否成功
        if response.status_code == 200:
            # 解析返回的 JSON 数据为字典
            path_data = response.json()
            path = path_data['coords']
            t = path_data['t']
            # print("接口返回json:", path_data)
            print("路线长度（包含起点）:", len(path), ",目标个数（包含起点）:", t_l)
            print("gsoa计算时间(ms):", t)
            # 利用起点重新排序，并把起点删除
            new_path = generate_new_path(path, pos_now)
            cost = get_path_time_cost(path,True)
            print("本次路线时间代价:", cost)

            # # 用另一个进程来显示几何体
            # p = multiprocessing.Process(target=generate_geometric_shapes_show, args=(geometrics, pos_now,))
            # # 启动进程
            # p.start()
            # # 用另一个进程来显示路线
            # p2 = multiprocessing.Process(target=generate_geometric_shapes_show_path,
            #                              args=(geometrics, pos_now, path_data))
            # # 启动进程
            # p2.start()

            # 判断路线中的访问点是否真正的访问到了所有的几何体
            # if check_path_access(new_path, geometrics):
            #     print("所有的几何体都被访问到啦")
            # else:
            #     print("有几何体都被漏啦！！！！！！！！！！！")

            return new_path,cost,t
        else:
            # 如果请求失败，抛出异常或返回错误信息
            raise Exception(f"API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}")


    else:
        return []


def append_result(
    sample_id, Lref,
    # OGSOA_GRO
    ogsoa_gro_pdb, ogsoa_gro_pdm, ogsoa_gro_std_l, ogsoa_gro_std_m,
    ogsoa_gro_best, ogsoa_gro_avg, ogsoa_gro_time,
    # GSOA_GRO
    gsoa_gro_pdb, gsoa_gro_pdm, gsoa_gro_std_l, gsoa_gro_std_m,
    gsoa_gro_best, gsoa_gro_avg, gsoa_gro_time,
    # OGSOA
    ogsoa_pdb, ogsoa_pdm, ogsoa_std_l, ogsoa_std_m,
    ogsoa_best, ogsoa_avg, ogsoa_time,
    # OGSOA_GRO_z0.5
    ogsoa_gro_z05_pdb, ogsoa_gro_z05_pdm, ogsoa_gro_z05_std_l, ogsoa_gro_z05_std_m,
    ogsoa_gro_z05_best, ogsoa_gro_z05_avg, ogsoa_gro_z05_time,
    # OGSOA_GRO_d4
    ogsoa_gro_d4_pdb, ogsoa_gro_d4_pdm, ogsoa_gro_d4_std_l, ogsoa_gro_d4_std_m,
    ogsoa_gro_d4_best, ogsoa_gro_d4_avg, ogsoa_gro_d4_time,
    # OGSOA_GRO_d16
    ogsoa_gro_d16_pdb, ogsoa_gro_d16_pdm, ogsoa_gro_d16_std_l, ogsoa_gro_d16_std_m,
    ogsoa_gro_d16_best, ogsoa_gro_d16_avg, ogsoa_gro_d16_time,
    filename='results_ablations.csv'
):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            # 前两列：Sample_ID, BestCost
            sample_id, f"{Lref:.2f}",

            # OGSOA_GRO-*
            f"{ogsoa_gro_pdb:.2f}", f"{ogsoa_gro_pdm:.2f}",
            f"{ogsoa_gro_std_l:.2f}", f"{ogsoa_gro_std_m:.2f}",
            f"{ogsoa_gro_best:.2f}", f"{ogsoa_gro_avg:.2f}",
            f"{ogsoa_gro_time:.2f}",

            # GSOA_GRO-*
            f"{gsoa_gro_pdb:.2f}", f"{gsoa_gro_pdm:.2f}",
            f"{gsoa_gro_std_l:.2f}", f"{gsoa_gro_std_m:.2f}",
            f"{gsoa_gro_best:.2f}", f"{gsoa_gro_avg:.2f}",
            f"{gsoa_gro_time:.2f}",

            # OGSOA-*
            f"{ogsoa_pdb:.2f}", f"{ogsoa_pdm:.2f}",
            f"{ogsoa_std_l:.2f}", f"{ogsoa_std_m:.2f}",
            f"{ogsoa_best:.2f}", f"{ogsoa_avg:.2f}",
            f"{ogsoa_time:.2f}",

            # OGSOA_GRO_z0.5-*
            f"{ogsoa_gro_z05_pdb:.2f}", f"{ogsoa_gro_z05_pdm:.2f}",
            f"{ogsoa_gro_z05_std_l:.2f}", f"{ogsoa_gro_z05_std_m:.2f}",
            f"{ogsoa_gro_z05_best:.2f}", f"{ogsoa_gro_z05_avg:.2f}",
            f"{ogsoa_gro_z05_time:.2f}",

            # OGSOA_GRO_d4-*
            f"{ogsoa_gro_d4_pdb:.2f}", f"{ogsoa_gro_d4_pdm:.2f}",
            f"{ogsoa_gro_d4_std_l:.2f}", f"{ogsoa_gro_d4_std_m:.2f}",
            f"{ogsoa_gro_d4_best:.2f}", f"{ogsoa_gro_d4_avg:.2f}",
            f"{ogsoa_gro_d4_time:.2f}",

            # OGSOA_GRO_d16-*
            f"{ogsoa_gro_d16_pdb:.2f}", f"{ogsoa_gro_d16_pdm:.2f}",
            f"{ogsoa_gro_d16_std_l:.2f}", f"{ogsoa_gro_d16_std_m:.2f}",
            f"{ogsoa_gro_d16_best:.2f}", f"{ogsoa_gro_d16_avg:.2f}",
            f"{ogsoa_gro_d16_time:.2f}",
        ])

def append_model_result(
    sample_id, Lref,
    # OGSOA_GRO
    ogsoa_gro_m_avg, ogsoa_gro_m_std_l, ogsoa_gro_m_std_m,
    # GSOA_GRO
    gsoa_gro_m_avg, gsoa_gro_m_std_l, gsoa_gro_m_std_m,
    # OGSOA
    ogsoa_m_avg, ogsoa_m_std_l, ogsoa_m_std_m,
    # OGSOA_GRO_z0.5
    ogsoa_gro_z05_m_avg, ogsoa_gro_z05_m_std_l, ogsoa_gro_z05_m_std_m,
    # OGSOA_GRO_d4
    ogsoa_gro_d4_m_avg, ogsoa_gro_d4_m_std_l, ogsoa_gro_d4_m_std_m,
    # OGSOA_GRO_d16
    ogsoa_gro_d16_m_avg, ogsoa_gro_d16_m_std_l, ogsoa_gro_d16_m_std_m,
    filename='results_ablations_model.csv'
):
    """
    建模时间结果写入第二个 CSV：
    每个算法三列：M-AvgTcpu(ms), M-STD-l, M-STD-m
    """
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            sample_id, f"{Lref:.2f}",
            # OGSOA_GRO
            f"{ogsoa_gro_m_avg:.2f}", f"{ogsoa_gro_m_std_l:.2f}", f"{ogsoa_gro_m_std_m:.2f}",
            # GSOA_GRO
            f"{gsoa_gro_m_avg:.2f}",  f"{gsoa_gro_m_std_l:.2f}",  f"{gsoa_gro_m_std_m:.2f}",
            # OGSOA
            f"{ogsoa_m_avg:.2f}",     f"{ogsoa_m_std_l:.2f}",     f"{ogsoa_m_std_m:.2f}",
            # OGSOA_GRO_z0.5
            f"{ogsoa_gro_z05_m_avg:.2f}", f"{ogsoa_gro_z05_m_std_l:.2f}", f"{ogsoa_gro_z05_m_std_m:.2f}",
            # OGSOA_GRO_d4
            f"{ogsoa_gro_d4_m_avg:.2f}",  f"{ogsoa_gro_d4_m_std_l:.2f}",  f"{ogsoa_gro_d4_m_std_m:.2f}",
            # OGSOA_GRO_d16
            f"{ogsoa_gro_d16_m_avg:.2f}", f"{ogsoa_gro_d16_m_std_l:.2f}", f"{ogsoa_gro_d16_m_std_m:.2f}",
        ])
# 小工具：计算均值和两种标准差（建模时间用）
def calc_model_stats(mts):
        mts = np.asarray(mts, dtype=float)
        if mts.size == 0:
            return 0.0, 0.0, 0.0
        if mts.size == 1:
            v = float(mts[0])
            return v, 0.0, 0.0
        avg = float(np.mean(mts))
        std_l = float(np.std(mts, ddof=1))  # 样本标准差
        std_m = float(np.std(mts, ddof=0))  # 总体标准差
        return avg, std_l, std_m
# =========================
if __name__ == "__main__":

    NS = [10,20,30,40,50,60,70,80]
    # NS = [30]
    run_count = 50
    W, H = 2560, 1440
    canvas_width = W
    canvas_height = H
    n = 1
    s = 0
    z_max = 23
    va = [51.2400016784668, 30.190000534057617]
    seed = 666
    random.seed(seed)  # 42 是种子值，你可以换成任意整数

    # ===================== 原始结果表头 =====================
    header = [
        'Sample_ID', 'BestCost',
        "OGSOA_GRO-PDB", "OGSOA_GRO-PDM", "OGSOA_GRO-STD-l", "OGSOA_GRO-STD-m",
        "OGSOA_GRO-BestCost", "OGSOA_GRO-AvgCost", "OGSOA_GRO-AvgTcpu(ms)",
        "GSOA_GRO-PDB", "GSOA_GRO-PDM", "GSOA_GRO-STD-l", "GSOA_GRO-STD-m",
        "GSOA_GRO-BestCost", "GSOA_GRO-AvgCost", "GSOA_GRO-AvgTcpu(ms)",
        "OGSOA-PDB", "OGSOA-PDM", "OGSOA-STD-l", "OGSOA-STD-m",
        "OGSOA-BestCost", "OGSOA-AvgCost", "OGSOA-AvgTcpu(ms)",
        "OGSOA_GRO_z0.5-PDB", "OGSOA_GRO_z0.5-PDM", "OGSOA_GRO_z0.5-STD-l", "OGSOA_GRO_z0.5-STD-m",
        "OGSOA_GRO_z0.5-BestCost", "OGSOA_GRO_z0.5-AvgCost", "OGSOA_GRO_z0.5-AvgTcpu(ms)",
        "OGSOA_GRO_d4-PDB", "OGSOA_GRO_d4-PDM", "OGSOA_GRO_d4-STD-l", "OGSOA_GRO_d4-STD-m",
        "OGSOA_GRO_d4-BestCost", "OGSOA_GRO_d4-AvgCost", "OGSOA_GRO_d4-AvgTcpu(ms)",
        "OGSOA_GRO_d16-PDB", "OGSOA_GRO_d16-PDM", "OGSOA_GRO_d16-STD-l", "OGSOA_GRO_d16-STD-m",
        "OGSOA_GRO_d16-BestCost", "OGSOA_GRO_d16-AvgCost", "OGSOA_GRO_d16-AvgTcpu(ms)",
    ]

    with open('results_ablations.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    # ===================== 新建“建模时间”结果表 =====================
    model_header = [
        'Sample_ID', 'BestTime',
        "OGSOA_GRO-M-AvgTcpu(ms)", "OGSOA_GRO-M-STD-l", "OGSOA_GRO-M-STD-m",
        "GSOA_GRO-M-AvgTcpu(ms)", "GSOA_GRO-M-STD-l", "GSOA_GRO-M-STD-m",
        "OGSOA-M-AvgTcpu(ms)", "OGSOA-M-STD-l", "OGSOA-M-STD-m",
        "OGSOA_GRO_z0.5-M-AvgTcpu(ms)", "OGSOA_GRO_z0.5-M-STD-l", "OGSOA_GRO_z0.5-M-STD-m",
        "OGSOA_GRO_d4-M-AvgTcpu(ms)", "OGSOA_GRO_d4-M-STD-l", "OGSOA_GRO_d4-M-STD-m",
        "OGSOA_GRO_d16-M-AvgTcpu(ms)", "OGSOA_GRO_d16-M-STD-l", "OGSOA_GRO_d16-M-STD-m",
    ]

    with open('results_ablations_model.csv', mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(model_header)

    modeling_url = "http://127.0.0.1:18080/model"
    # modeling_url = "http://172.30.249.71:18080/model"
    for N in NS:

        # 每个 N 随机一个 proportion，并据此限制像素质量
        proportion = random.randint(100, 1000) / 10000.0  # 0.01 ~ 0.10
        pixel_limit = int(20000 * proportion)  # 影响 generate_datasets_by_bins
        # 论文用严格分档：strict=True, slack=0.0
        outs = generate_datasets_by_bins(
            N=N, W=W, H=H,
            proportion=proportion,
            groups_per_bin=20,
            seed=seed,
            margin=0.95,
            psr_bins={'low':(0.0,0.05), 'med':(0.05,0.15), 'high':(0.15,1.01)},
            max_seconds_per_group=5.0,  # 严格模式下仅作为保护，不回退
            slack=0.0,
            strict=True,
            pixel_limit = pixel_limit
        )
        save_outputs(outs, N=N, outdir=f"ablations_datasets_{N}", canvas_width=W, canvas_height=H)

        for key, boxes_s in outs.items():
            for b_i, value in enumerate(boxes_s):
                # 每个样本一个随机起始 PTZ
                pn_p = random.randint(10, 3000) / 10.0  # 1.0 ~ 300.0
                pn_t = random.randint(10, 500) / 10.0  # 1.0 ~ 50.0
                pos_now = [pn_p, pn_t, 1.0]

                boxes = value[0]
                best_cost = float('inf')
                best_path = None
                best_Mtime = float('inf')

                f_name = f"{key}-{N}-{b_i}"
                start_time = time.time()
                print("***********", f_name, "开始**********")

                # ========= 1) OGSOA_GRO =========
                base_best_cost = float('inf')
                base_best_t = float('inf')
                base_cs, base_ts, base_mts = [], [], []
                base_best_path = []
                gsoa_base_url = "http://127.0.0.1:18888/api/gsoa"


                for a in range(run_count):
                    # 建模（g_type = 0）
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 0,  # 最后一个 0 表示基础建模
                    )
                    gsoa_str = r_json["json"]
                    base_mt = r_json["t"]
                    base_mts.append(base_mt)

                    if base_mt < best_Mtime:
                        best_Mtime = base_mt


                    # 算法求解
                    ps, cost, t = snapshot_gsoa_gs(N + 1, gsoa_str, gsoa_base_url)
                    if cost < base_best_cost:
                        base_best_cost = cost
                        base_best_path = ps
                    if 0 < t < base_best_t:
                        base_best_t = t
                    base_cs.append(cost)
                    base_ts.append(t)
                    time.sleep(0.1)

                if base_best_cost < best_cost:
                    best_cost = base_best_cost
                    best_path = base_best_path

                base_std_l = np.std(base_cs, ddof=1)
                base_std_m = np.std(base_cs, ddof=0)
                base_avg_cost = np.mean(base_cs)
                base_avg_time = np.mean(base_ts)

                print("==============================================")

                # ========= 2) GSOA_GRO =========
                jiao_best_cost = float('inf')
                jiao_best_t = float('inf')
                jiao_cs, jiao_ts, jiao_mts = [], [], []
                jiao_best_path = []
                gsoa_jiao_url = "http://127.0.0.1:8890/api/gsoa"

                for a in range(run_count):
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 0,  # 同样 g_type = 0
                    )
                    gsoa_str = r_json["json"]
                    jiao_mt = r_json["t"]
                    jiao_mts.append(jiao_mt)
                    if jiao_mt < best_Mtime:
                        best_Mtime = jiao_mt



                    j_ps, j_cost, j_t = snapshot_gsoa_gs(N + 1, gsoa_str, gsoa_jiao_url)
                    if j_cost < jiao_best_cost:
                        jiao_best_cost = j_cost
                        jiao_best_path = j_ps
                    if 0 < j_t < jiao_best_t:
                        jiao_best_t = j_t
                    jiao_cs.append(j_cost)
                    jiao_ts.append(j_t)
                    time.sleep(0.1)

                if jiao_best_cost < best_cost:
                    best_cost = jiao_best_cost
                    best_path = jiao_best_path

                jiao_std_l = np.std(jiao_cs, ddof=1)
                jiao_std_m = np.std(jiao_cs, ddof=0)
                jiao_avg_cost = np.mean(jiao_cs)
                jiao_avg_time = np.mean(jiao_ts)

                print("==============================================")

                # ========= 3) OGSOA =========
                j_g_best_cost = float('inf')
                j_g_best_t = float('inf')
                j_g_cs, j_g_ts, j_g_mts = [], [], []
                j_g_best_path = []
                gsoa_j_g_url = "http://127.0.0.1:9999/api/gsoa"

                for a in range(run_count):
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 0,  # g_type = 0
                    )
                    gsoa_str = r_json["json"]
                    j_g_mt = r_json["t"]
                    j_g_mts.append(j_g_mt)

                    if j_g_mt < best_Mtime:
                        best_Mtime = j_g_mt

                    j_g_ps, j_g_cost, j_g_t = snapshot_gsoa_gs(N + 1, gsoa_str, gsoa_j_g_url)
                    if j_g_cost < j_g_best_cost:
                        j_g_best_cost = j_g_cost
                        j_g_best_path = j_g_ps
                    if 0 < j_g_t < j_g_best_t:
                        j_g_best_t = j_g_t
                    j_g_cs.append(j_g_cost)
                    j_g_ts.append(j_g_t)
                    time.sleep(0.1)

                if j_g_best_cost < best_cost:
                    best_cost = j_g_best_cost
                    best_path = j_g_best_path

                j_g_std_l = np.std(j_g_cs, ddof=1)
                j_g_std_m = np.std(j_g_cs, ddof=0)
                j_g_avg_cost = np.mean(j_g_cs)
                j_g_avg_time = np.mean(j_g_ts)

                print("==============================================")

                # ========= 4) OGSOA_GRO_z0.5 =========
                z05_best_cost = float('inf')
                z05_best_t = float('inf')
                z05_cs, z05_ts, z05_mts = [], [], []
                z05_best_path = []
                z05_url = "http://127.0.0.1:18888/api/gsoa"

                for a in range(run_count):
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 1,  # 最后一个 1 表示 z0.5 建模
                    )
                    gsoa_str = r_json["json"]
                    z05_mt = r_json["t"]
                    z05_mts.append(z05_mt)
                    if z05_mt < best_Mtime:
                        best_Mtime = z05_mt


                    ps, cost, t = snapshot_gsoa_gs(N + 1, gsoa_str, z05_url)
                    if cost < z05_best_cost:
                        z05_best_cost = cost
                        z05_best_path = ps
                    if 0 < t < z05_best_t:
                        z05_best_t = t
                    z05_cs.append(cost)
                    z05_ts.append(t)
                    time.sleep(0.1)

                if z05_best_cost < best_cost:
                    best_cost = z05_best_cost
                    best_path = z05_best_path

                z05_std_l = np.std(z05_cs, ddof=1)
                z05_std_m = np.std(z05_cs, ddof=0)
                z05_avg_cost = np.mean(z05_cs)
                z05_avg_time = np.mean(z05_ts)

                print("==============================================")

                # ========= 5) OGSOA_GRO_d4 =========
                d4_best_cost = float('inf')
                d4_best_t = float('inf')
                d4_cs, d4_ts, d4_mts = [], [], []
                d4_best_path = []
                d4_url = "http://127.0.0.1:18888/api/gsoa"

                for a in range(run_count):
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 2,  # 最后一个 2 表示 d4 建模
                    )
                    gsoa_str = r_json["json"]
                    d4_mt = r_json["t"]
                    d4_mts.append(d4_mt)
                    if d4_mt < best_Mtime:
                        best_Mtime = d4_mt


                    ps, cost, t = snapshot_gsoa_gs(N + 1, gsoa_str, d4_url)
                    if cost < d4_best_cost:
                        d4_best_cost = cost
                        d4_best_path = ps
                    if 0 < t < d4_best_t:
                        d4_best_t = t
                    d4_cs.append(cost)
                    d4_ts.append(t)
                    time.sleep(0.1)

                if d4_best_cost < best_cost:
                    best_cost = d4_best_cost
                    best_path = d4_best_path

                d4_std_l = np.std(d4_cs, ddof=1)
                d4_std_m = np.std(d4_cs, ddof=0)
                d4_avg_cost = np.mean(d4_cs)
                d4_avg_time = np.mean(d4_ts)

                print("==============================================")

                # ========= 6) OGSOA_GRO_d16 =========
                d16_best_cost = float('inf')
                d16_best_t = float('inf')
                d16_cs, d16_ts, d16_mts = [], [], []
                d16_best_path = []
                d16_url = "http://127.0.0.1:18888/api/gsoa"

                for a in range(run_count):
                    r_json = model_http(
                        modeling_url,
                        boxes, pos_now, proportion, n, s,
                        canvas_width, canvas_height,
                        z_max, va, 0, 0, 3,  # 最后一个 3 表示 d16 建模
                    )
                    gsoa_str = r_json["json"]
                    d16_mt = r_json["t"]
                    d16_mts.append(d16_mt)

                    if d16_mt < best_Mtime:
                        best_Mtime = d16_mt

                    # 保留你原来的 N 参数逻辑：d16 用 2
                    ps, cost, t = snapshot_gsoa_gs(N+1, gsoa_str, d16_url)
                    if cost < d16_best_cost:
                        d16_best_cost = cost
                        d16_best_path = ps
                    if 0 < t < d16_best_t:
                        d16_best_t = t
                    d16_cs.append(cost)
                    d16_ts.append(t)
                    time.sleep(0.1)

                if d16_best_cost < best_cost:
                    best_cost = d16_best_cost
                    best_path = d16_best_path

                d16_std_l = np.std(d16_cs, ddof=1)
                d16_std_m = np.std(d16_cs, ddof=0)
                d16_avg_cost = np.mean(d16_cs)
                d16_avg_time = np.mean(d16_ts)

                print("==============================================")

                # ========= 统一计算 PDB / PDM（相对最终 best_cost） =========
                base_PDB = (base_best_cost - best_cost) / base_best_cost * 100 if base_best_cost > 0 else 0.0
                base_PDM = (base_avg_cost - best_cost) / base_avg_cost * 100 if base_avg_cost > 0 else 0.0

                jiao_PDB = (jiao_best_cost - best_cost) / jiao_best_cost * 100 if jiao_best_cost > 0 else 0.0
                jiao_PDM = (jiao_avg_cost - best_cost) / jiao_avg_cost * 100 if jiao_avg_cost > 0 else 0.0

                j_g_PDB = (j_g_best_cost - best_cost) / j_g_best_cost * 100 if j_g_best_cost > 0 else 0.0
                j_g_PDM = (j_g_avg_cost - best_cost) / j_g_avg_cost * 100 if j_g_avg_cost > 0 else 0.0

                z05_PDB = (z05_best_cost - best_cost) / z05_best_cost * 100 if z05_best_cost > 0 else 0.0
                z05_PDM = (z05_avg_cost - best_cost) / z05_avg_cost * 100 if z05_avg_cost > 0 else 0.0

                d4_PDB = (d4_best_cost - best_cost) / d4_best_cost * 100 if d4_best_cost > 0 else 0.0
                d4_PDM = (d4_avg_cost - best_cost) / d4_avg_cost * 100 if d4_avg_cost > 0 else 0.0

                d16_PDB = (d16_best_cost - best_cost) / d16_best_cost * 100 if d16_best_cost > 0 else 0.0
                d16_PDM = (d16_avg_cost - best_cost) / d16_avg_cost * 100 if d16_avg_cost > 0 else 0.0

                # ========= 建模时间统计（用刚才的 mts 列表） =========
                base_m_avg, base_m_std_l, base_m_std_m = calc_model_stats(base_mts)
                jiao_m_avg, jiao_m_std_l, jiao_m_std_m = calc_model_stats(jiao_mts)
                j_g_m_avg, j_g_m_std_l, j_g_m_std_m = calc_model_stats(j_g_mts)
                z05_m_avg, z05_m_std_l, z05_m_std_m = calc_model_stats(z05_mts)
                d4_m_avg, d4_m_std_l, d4_m_std_m = calc_model_stats(d4_mts)
                d16_m_avg, d16_m_std_l, d16_m_std_m = calc_model_stats(d16_mts)

                # ========= 文本记录 =========
                r_str = f"ID:{f_name}\n"
                r_str += f"pos_now:{pos_now}\n"
                r_str += f"best_cost:{best_cost}\n"
                r_str += f"best_path:{best_path}\n"

                r_str += f"OGSOA_GRO-cs:{base_cs}\n"
                r_str += f"OGSOA_GRO-ts:{base_ts}\n"

                r_str += f"GSOA_GRO-cs:{jiao_cs}\n"
                r_str += f"GSOA_GRO-ts:{jiao_ts}\n"

                r_str += f"OGSOA-cs:{j_g_cs}\n"
                r_str += f"OGSOA-ts:{j_g_ts}\n"

                r_str += f"OGSOA_GRO_z0.5-cs:{z05_cs}\n"
                r_str += f"OGSOA_GRO_z0.5-ts:{z05_ts}\n"

                r_str += f"OGSOA_GRO_d4-cs:{d4_cs}\n"
                r_str += f"OGSOA_GRO_d4-ts:{d4_ts}\n"

                r_str += f"OGSOA_GRO_d16-cs:{d16_cs}\n"
                r_str += f"OGSOA_GRO_d16-ts:{d16_ts}\n"

                # 新增：建模时间列表（各算法 50 次）
                r_str += f"OGSOA_GRO-mts:{base_mts}\n"
                r_str += f"GSOA_GRO-mts:{jiao_mts}\n"
                r_str += f"OGSOA-mts:{j_g_mts}\n"
                r_str += f"OGSOA_GRO_z0.5-mts:{z05_mts}\n"
                r_str += f"OGSOA_GRO_d4-mts:{d4_mts}\n"
                r_str += f"OGSOA_GRO_d16-mts:{d16_mts}\n"

                with open(f"{f_name}-result.txt", "w", encoding="utf-8") as f:
                    f.write(r_str)

                # ========= 写入 原 CSV（算法性能） =========
                append_result(
                    f_name, best_cost,
                    base_PDB, base_PDM, base_std_l, base_std_m, base_best_cost, base_avg_cost, base_avg_time,
                    jiao_PDB, jiao_PDM, jiao_std_l, jiao_std_m, jiao_best_cost, jiao_avg_cost, jiao_avg_time,
                    j_g_PDB, j_g_PDM, j_g_std_l, j_g_std_m, j_g_best_cost, j_g_avg_cost, j_g_avg_time,
                    z05_PDB, z05_PDM, z05_std_l, z05_std_m, z05_best_cost, z05_avg_cost, z05_avg_time,
                    d4_PDB, d4_PDM, d4_std_l, d4_std_m, d4_best_cost, d4_avg_cost, d4_avg_time,
                    d16_PDB, d16_PDM, d16_std_l, d16_std_m, d16_best_cost, d16_avg_cost, d16_avg_time,
                )

                # ========= 写入 建模时间 CSV =========
                append_model_result(
                    f_name, best_Mtime,
                    # OGSOA_GRO
                    base_m_avg, base_m_std_l, base_m_std_m,
                    # GSOA_GRO
                    jiao_m_avg, jiao_m_std_l, jiao_m_std_m,
                    # OGSOA
                    j_g_m_avg, j_g_m_std_l, j_g_m_std_m,
                    # OGSOA_GRO_z0.5
                    z05_m_avg, z05_m_std_l, z05_m_std_m,
                    # OGSOA_GRO_d4
                    d4_m_avg, d4_m_std_l, d4_m_std_m,
                    # OGSOA_GRO_d16
                    d16_m_avg, d16_m_std_l, d16_m_std_m,
                )

                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"{f_name} 用时:", elapsed_time, "秒,结束！")