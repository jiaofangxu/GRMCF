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
class Logger(object):
    def __init__(self, filename="output.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message+f"\n")  # 输出到终端
        self.log.write(message+f"\n")       # 写入文件

    def flush(self):
        pass  # 可选实现（有些环境需要）




def three_point_determination_of_function_expression(p1, p2, p3, point):
    """
    通过三个点计算平面方程 Ax + By + Cz = D 的系数 A, B, C, D，
    并保证额外的点满足 Ax + By + Cz <= D。

    如果第四个点不满足不等式，则将 A, B, C, D 全部取反。

    :param p1: 第一个点 (x1, y1, z1)
    :param p2: 第二个点 (x2, y2, z2)
    :param p3: 第三个点 (x3, y3, z3)
    :param point: 要满足不等式的点 (x, y, z)
    :return: (A, B, C, D) 表示平面方程 Ax + By + Cz <= D
    """
    # 计算两个向量
    v1 = np.array([p2[i] - p1[i] for i in range(3)])
    v2 = np.array([p3[i] - p1[i] for i in range(3)])

    # 计算法向量 (A, B, C)
    normal_vector = np.cross(v1, v2)
    A, B, C = normal_vector

    # 计算 D 值
    D = A * p1[0] + B * p1[1] + C * p1[2]

    # 检查给定点是否满足不等式
    x, y, z = point
    if A * x + B * y + C * z > D:
        # 如果不满足，则将 A, B, C, D 全部取反
        A, B, C, D = -A, -B, -C, -D

    return A, B, C, D


def absoute_coordinate_test_invert(pos_now, c_x, c_y, fov_w=2560, fov_h=1440, X=5.13, Y=2.89,
                                   f=5.35):  # X为传感器宽度 Y为传感器高度 f为有效焦距
    viewing_angle_dict = {'x': 51.2400016784668, 'y': 30.190000534057617}
    # print(viewing_angle_dict)
    # 起点
    # pos_now = [62.900001525878906, 23.600000381469727, 1.0]
    # print("pos_now:",pos_now)

    x0_sensor = X / 2
    y0_sensor = Y / 2
    center_psycial = [x0_sensor, y0_sensor]
    r_test = {}
    pos_object = []

    center_coordinate = [c_x, c_y]
    object_psycial = [(center_coordinate[0] * X) / fov_w, (center_coordinate[1] * Y) / fov_h]

    temp = math.sqrt(((object_psycial[1] - center_psycial[1]) ** 2) + (f ** 2))

    temp1 = math.sin(
        math.atan((object_psycial[1] - center_psycial[1]) / f) + np.deg2rad(pos_now[1]))

    a1 = math.asin((math.sqrt(((object_psycial[1] - center_psycial[1]) ** 2) + (f ** 2))) / (math.sqrt(
        (object_psycial[0] - center_psycial[0]) ** 2 + (
                object_psycial[1] - center_psycial[1]) ** 2 + f ** 2)) * math.sin(
        math.atan((object_psycial[1] - center_psycial[1]) / f) + np.deg2rad(pos_now[1])))
    ac = a1 - np.deg2rad(pos_now[1])  # 与中心的垂直夹角
    # bc为与中心的水平角
    if object_psycial[0] - center_psycial[0] >= 0:
        bc = math.asin(math.sin(math.atan((object_psycial[0] - center_psycial[0]) / math.sqrt(
            (object_psycial[1] - center_psycial[1]) ** 2 + f ** 2))) / math.cos(a1))
        bc_now = np.rad2deg(bc)
        pos_object.append(pos_now[0] - bc_now)

        # pos_object.append(pos_now[0] + bc_now)  # 改摄像头反正
    else:
        bc = math.asin(math.sin(math.atan((center_psycial[0] - object_psycial[0]) / math.sqrt(
            (object_psycial[1] - center_psycial[1]) ** 2 + f ** 2))) / math.cos(a1))
        bc_now = np.rad2deg(bc)

        pos_object.append(pos_now[0] + bc_now)  # 改摄像头反正
    ac_now = np.rad2deg(ac)

    # if (pos_now[1] - ac_now) < -5:
    #     pos_object.append(-5)            #摄像头反正
    if (pos_now[1] - ac_now) < -5.0:
        pos_object.append(-5.0)
    else:

        pos_object.append(pos_now[1] - ac_now)  # 改摄像头反正
    # 保留小数点后一位
    pos_object[0] = np.round(pos_object[0], 1)
    pos_object[1] = np.round(pos_object[1], 1)
    return pos_object


def generate_hierarchical_point_set(pos_now, c_x, c_y, fov_w=640, fov_h=360, X=5.13, Y=2.89, f=5.35):
    return 0


def generate_sequence(x1, x2, y):
    # print("generate_sequence ", x1, x2, y)
    # 生成序列并确保最后一个值不超过 x2
    sequence = np.round(np.arange(x1, x2 + y, y), 10)  # 解决浮点误差
    if len(sequence) == 0:
        print("generate_sequence sequence长度为0,", x1, x2, y)
        return [x2]
    # 如果序列最后一个元素超过 x2，将其替换为 x2
    if sequence[-1] > x2:
        sequence = sequence[:-1]  # 去掉超出范围的最后一个元素
    # 将最后一个元素确保为 x2
    if sequence[-1] != x2:
        sequence = np.append(sequence, x2)
    return sequence


def inverse_translate_Z(z, viewing_angle_dict):
    viewing_angle_x = viewing_angle_dict['x']
    viewing_angle_y = viewing_angle_dict['y']
    # 模型参数
    kH1 = 1.87837538
    kH2 = 0.89869125
    kH3 = 0.27214642
    kV1 = 1.11636674
    kV2 = 0.83942483
    kV3 = 0.25487924
    z = sdk_ztoonvif(z)
    # 计算水平和垂直视场角
    FOVH_need = kH1 * np.exp(kH2 / (z + kH3))
    FOVV_need = kV1 * np.exp(kV2 / (z + kV3))

    # 原视场角转换为弧度
    original_h_half = np.deg2rad(viewing_angle_x) / 2
    original_v_half = np.deg2rad(viewing_angle_y) / 2

    # 计算新视场角的半角正切值
    tan_new_h_half = np.tan(np.deg2rad(FOVH_need) / 2)
    tan_new_v_half = np.tan(np.deg2rad(FOVV_need) / 2)

    # 计算宽高比例
    length_ratio = tan_new_h_half / np.tan(original_h_half)
    height_ratio = tan_new_v_half / np.tan(original_v_half)

    # 计算像素尺寸并四舍五入
    length = np.round(2560 * length_ratio)
    height = np.round(1440 * height_ratio)

    return length, height


def onviftosdk_z(x):
    y = (22 * x + 1)
    return y


def sdk_ztoonvif(y):
    x = (y - 1) / 22
    return x


def translate_Z(length, height, viewing_angle_dict):  # 要使用实时的视场角
    # print("开始算z了！！！！！！！！！！！！！！！！")

    kH1 = 1.87837538
    kH2 = 0.89869125
    kH3 = 0.27214642
    kV1 = 1.11636674
    kV2 = 0.83942483
    kV3 = 0.25487924
    # viewing_angle_dict = {'x': 51.2400016784668, 'y': 30.190000534057617}
    # print("getGisInfo:", viewing_angle_dict)

    FOVH_need = np.rad2deg(
        2 * np.arctan((length / 2560) * np.tan(np.deg2rad((viewing_angle_dict['x']) / 2))))
    FOVV_need = np.rad2deg(
        2 * np.arctan((height / 1440) * np.tan(np.deg2rad((viewing_angle_dict['y']) / 2))))
    # print(FOVH_need)
    # print(FOVV_need)
    z1 = (kH2 / (np.log(FOVH_need / kH1))) - kH3
    z2 = (kV2 / (np.log(FOVV_need / kV1))) - kV3
    z = (z1 + z2) / 2  # 放大到z倍
    new_z = onviftosdk_z(z)
    # 保留小数点后一位
    # new_z = np.round(new_z, 1)
    return new_z


def vector_cross(a, b):
    """计算向量叉积"""
    return np.cross(a, b)


def vector_dot(a, b):
    """计算向量点积"""
    return np.dot(a, b)


def compute_volume_of_tetrahedron(A, B, C, D):
    """计算四面体的体积"""
    AB = np.array(B) - np.array(A)
    AC = np.array(C) - np.array(A)
    AD = np.array(D) - np.array(A)
    return np.abs(vector_dot(AB, vector_cross(AC, AD))) / 6


def compute_centroid_of_tetrahedron(A, B, C, D):
    """计算四面体的形心"""
    return (np.array(A) + np.array(B) + np.array(C) + np.array(D)) / 4


def compute_wedge_centroid_h(A, B, C, D, E, F):
    """
    计算楔形体形心 横向
    A, B, C, D: 底面四边形的四个点
    E, F: 顶面两点
    """
    # 四面体 1: 底面三角形 ABC 和顶面点 E
    volume1 = compute_volume_of_tetrahedron(A, B, C, E)
    centroid1 = compute_centroid_of_tetrahedron(A, B, C, E)

    # 四面体 2: 底面三角形 ACD 和顶面点 E
    volume2 = compute_volume_of_tetrahedron(A, C, D, E)
    centroid2 = compute_centroid_of_tetrahedron(A, C, D, E)

    # 四面体 3: 底面三角形 ADF 和顶面点 F
    volume3 = compute_volume_of_tetrahedron(B, C, F, E)
    centroid3 = compute_centroid_of_tetrahedron(B, C, F, E)

    # 计算总体积
    total_volume = volume1 + volume2 + volume3

    # 体积加权的整体形心
    centroid = (volume1 * centroid1 + volume2 * centroid2 + volume3 * centroid3) / total_volume

    return centroid


def compute_wedge_centroid_v(A, B, C, D, E, F):
    """
    计算楔形体形心 竖向
    A, B, C, D: 底面四边形的四个点
    E, F: 顶面两点
    """
    # 四面体 1: 底面三角形 ABC 和顶面点 E
    volume1 = compute_volume_of_tetrahedron(A, B, C, E)
    centroid1 = compute_centroid_of_tetrahedron(A, B, C, E)

    # 四面体 2: 底面三角形 ACD 和顶面点 E
    volume2 = compute_volume_of_tetrahedron(A, C, D, E)
    centroid2 = compute_centroid_of_tetrahedron(A, C, D, E)

    # 四面体 3: 底面三角形 ADF 和顶面点 F
    volume3 = compute_volume_of_tetrahedron(A, E, C, F)
    centroid3 = compute_centroid_of_tetrahedron(A, E, C, F)

    # 计算总体积
    total_volume = volume1 + volume2 + volume3

    # 体积加权的整体形心
    centroid = (volume1 * centroid1 + volume2 * centroid2 + volume3 * centroid3) / total_volume

    return centroid


def compute_pyramid_centroid(A, B, C, D, E):
    """
        计算四棱锥体形心
        A, B, C, D: 底面四边形的四个点
        E: 顶面两点
        """
    # 四面体 1: 底面三角形 ABC 和顶面点 E
    volume1 = compute_volume_of_tetrahedron(A, B, C, E)
    centroid1 = compute_centroid_of_tetrahedron(A, B, C, E)

    # 四面体 2: 底面三角形 ACD 和顶面点 E
    volume2 = compute_volume_of_tetrahedron(A, C, D, E)
    centroid2 = compute_centroid_of_tetrahedron(A, C, D, E)

    # 计算总体积
    total_volume = volume1 + volume2

    # 体积加权的整体形心
    centroid = (volume1 * centroid1 + volume2 * centroid2) / total_volume

    return centroid


import numpy as np


def polygon_area_and_centroid(vertices):
    x, y = vertices[:, 0], vertices[:, 1]
    x_next, y_next = np.roll(x, -1), np.roll(y, -1)
    double_area = np.sum(x * y_next - x_next * y)
    signed_area = 0.5 * double_area
    area = abs(signed_area)

    if area == 0:
        return 0, np.mean(x), np.mean(y)

    cx = np.sum((x + x_next) * (x * y_next - x_next * y)) / (6 * signed_area)
    cy = np.sum((y + y_next) * (x * y_next - x_next * y)) / (6 * signed_area)

    return area, cx, cy


def compute_prism_volume(A1, A2, z1, z2):
    return (1 / 3) * abs(A1 + A2 + np.sqrt(A1 * A2)) * abs(z2 - z1)


def compute_3d_centroid(slices):
    if len(slices) == 1:
        z, vertices = slices[0]
        A, Cx, Cy = polygon_area_and_centroid(np.array(vertices))
        return Cx, Cy, z

    total_volume = 0
    weighted_cx, weighted_cy, weighted_cz = 0, 0, 0
    all_points = []

    for i in range(len(slices) - 1):
        z1, vertices1 = slices[i]
        z2, vertices2 = slices[i + 1]

        A1, Cx1, Cy1 = polygon_area_and_centroid(np.array(vertices1))
        A2, Cx2, Cy2 = polygon_area_and_centroid(np.array(vertices2))

        if A1 == 0 and A2 == 0:
            all_points.extend(vertices1)
            all_points.extend(vertices2)
            continue  # 跳过没有体积的部分

        V = compute_prism_volume(A1, A2, z1, z2)
        total_volume += V

        dz = abs(z2 - z1)
        if A1 == 0:  # 退化情况，直接用 A2 计算
            weight1, weight2 = 0, 1
        elif A2 == 0:
            weight1, weight2 = 1, 0
        else:
            weight1 = (A1 + np.sqrt(A1 * A2)) / (A1 + A2 + np.sqrt(A1 * A2))
            weight2 = 1 - weight1

        Cz = z1 * weight1 + z2 * weight2
        Cx = Cx1 * weight1 + Cx2 * weight2
        Cy = Cy1 * weight1 + Cy2 * weight2

        weighted_cx += V * Cx
        weighted_cy += V * Cy
        weighted_cz += V * Cz

        all_points.extend(vertices1)
        all_points.extend(vertices2)

    if total_volume == 0:
        all_points = np.array(all_points)
        return np.mean(all_points[:, 0]), np.mean(all_points[:, 1]), np.mean([s[0] for s in slices])

    return (weighted_cx / total_volume,
            weighted_cy / total_volume,
            weighted_cz / total_volume)


class PointIndexSet:
    def __init__(self):
        self.point_dict = {}  # 存储点及其对应的点索引和几何体索引列表
        self.index_counter = 0  # 索引计数器
        self.index_to_point = []  # 索引对应存储点
        self.geometrics = []  # 几何体列表

    def add_point(self, point, g_index):
        point_tuple = tuple(point)  # 将列表转换为元组以保持唯一性
        if point_tuple not in self.point_dict:
            self.point_dict[point_tuple] = [self.index_counter, [g_index]]
            self.index_to_point.append(point)
            self.index_counter += 1  # 更新索引计数器
            return self.index_counter - 1  # 返回点的索引
        else:
            p_d = self.point_dict[point_tuple]
            p_d[1].append(g_index)
            return p_d[0]  # 返回点的索引

    def p_self(self):
        return {
            "point_dict": self.point_dict,
            "index_counter": self.index_counter,
            "index_to_point": self.index_to_point,
            "geometrics": self.geometrics,
        }

    def fill_MESHER(self, geometrics, step):
        self.geometrics = geometrics
        g_index = 0
        for g in geometrics:
            g.add_p_to_set(step, self, g_index)
            g_index = g_index + 1

    def get_index_by_point(self, point):
        point_tuple = tuple(point)  # 将列表转换为元组以保持唯一性
        return self.point_dict[point_tuple][0]

    def get_g_indexs_by_point(self, point):
        point_tuple = tuple(point)  # 将列表转换为元组以保持唯一性
        return self.point_dict[point_tuple][1]

    def get_all_points(self):
        return self.index_to_point  # 返回所有的点

    def get_all_points_len(self):
        return self.index_counter + 1  # 从0开始计数的

    def get_point_by_index(self, index):
        return self.index_to_point[index]

    # def cost(self):
    #     #求出 当前 几何体到 其他几何体最短的两条边（共点，两条边之和最短）
    #     for g in self.geometrics:


class Geometric:
    def __init__(self, g_index, pos_now, box, proportion, fov_w, fov_h, z_max, viewing_angle_dict,
                 shrink=0):  # shrink_b底面系数 0是不收缩，1是收缩到一个点
        # 起点
        self.g_index = g_index
        self.pos_now = pos_now
        self.is_shortest_neighborhood = False
        # [[1,2],[3,4],5] 点1到点2，几何体3到几何体4，代价是5。
        self.shortest_neighborhood1 = []
        self.shortest_neighborhood2 = []
        self.centroid = []
        # 多面体的函数表达式
        self.function_expression = []
        # 私有元素初始化
        self.bottom_vertex1 = []
        self.bottom_vertex2 = []
        self.bottom_vertex3 = []
        self.bottom_vertex4 = []
        self.bottom_vertex12 = []
        self.bottom_vertex23 = []
        self.bottom_vertex34 = []
        self.bottom_vertex41 = []

        self.hierarchical_point_set = []

        self.min_z = 0
        self.max_z = 0
        # 几何类型，0初始目标框已经满足比例则无几何体生成，1四面锥体(top顶点只有1个)，2横向楔形体(top顶点2个，是横向的)，3竖向楔形体(top顶点2个，是竖向的)，4只有一个横线，5只有一个竖线
        self.geometric_type = 0
        self.p_list = []
        # 目标框宽
        box_w = box[2] - box[0]
        # 目标框高
        box_h = box[3] - box[1]
        # 目标框像素面积
        box_area = box_w * box_h
        # 视野像素面积
        fov_area = fov_w * fov_h
        # 视野的高宽比
        fov_aspect_ratio = fov_h / fov_w
        # 目标框高宽比
        box_aspect_ratio = box_h / box_w
        now_p = box_area / fov_area
        self.access_point = []
        if now_p >= proportion:
            # 无几何体生成
            self.geometric_type = 0

        else:
            # 最大的视野面积
            fov_max_area = box_area / proportion
            fov_max_w = (fov_max_area / fov_aspect_ratio) ** 0.5
            fov_max_h = fov_max_w * fov_aspect_ratio
            if fov_max_h < box_h or fov_max_w < box_w:
                # 框太瘦高，达到面积比例，但是超出视野了，那就最小视野和最大视野一致（都是视野），变为一条直线
                if box_aspect_ratio > fov_aspect_ratio:
                    # 横向直线
                    self.geometric_type = 4
                    # box的高作为最小视野的高
                    fov_min_h = box_h
                    fov_min_w = fov_min_h / fov_aspect_ratio
                    # 当box位于最小视野的最左边时，是最小视野中心的最右边
                    fov_min_l_x1 = box[0]
                    fov_min_l_y1 = box[1]
                    fov_min_center_x2 = fov_min_l_x1 + (fov_min_w / 2)
                    fov_min_center_y2 = fov_min_l_y1 + (fov_min_h / 2)
                    # 当box位于最小视野的最右边时，是最小视野中心的最左边
                    fov_min_r_x2 = box[2]
                    fov_min_r_y2 = box[3]
                    fov_min_center_x1 = fov_min_r_x2 - (fov_min_w / 2)
                    fov_min_center_y1 = fov_min_r_y2 - (fov_min_h / 2)
                    z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                    if z2 > z_max or z2 < 0:
                        z2 = z_max
                    if z2 > 1:
                        self.geometric_type = 1
                        z1 = z2 - 1
                        self.min_z = z1
                        zs = generate_sequence(z1, z2, 0.1)
                        for z in zs:
                            l, h = inverse_translate_Z(z, viewing_angle_dict)
                            l_h = (l / 2)
                            h_h = (h / 2)
                            f_c_x1 = box[2] - l_h
                            f_c_y1 = box[3] - h_h
                            f_c_x2 = box[0] + l_h
                            f_c_y2 = box[1] + h_h
                            f_c_h_x = (f_c_x1 + f_c_x2) / 2
                            f_c_h_y = (f_c_y1 + f_c_y2) / 2
                            pt1 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y1)
                            pt2 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y1)
                            pt3 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y2)
                            pt4 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y2)
                            pt12 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y1)
                            pt23 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_h_y)
                            pt34 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y2)
                            pt41 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_h_y)
                            h_ps = [pt1, pt12, pt2, pt23, pt3, pt34, pt4, pt41]

                            if (shrink > 0):
                                self.shrink_towards_center(h_ps, shrink)
                            n_hps = [z, h_ps]
                            self.hierarchical_point_set.append(n_hps)

                        # 求质心
                        self.centroid = compute_3d_centroid(self.hierarchical_point_set)



                    else:
                        pt = absoute_coordinate_test_invert(pos_now, fov_min_center_x1, fov_min_center_y1)
                        self.top_vertex1 = [pt[0], pt[1], z2]
                        pt = absoute_coordinate_test_invert(pos_now, fov_min_center_x2, fov_min_center_y2)
                        self.top_vertex2 = [pt[0], pt[1], z2]
                        self.bottom_vertex1 = [0, 0, 0]
                        self.bottom_vertex2 = [0, 0, 0]
                        self.bottom_vertex3 = [0, 0, 0]
                        self.bottom_vertex4 = [0, 0, 0]
                        self.centroid = [(self.top_vertex1[0] + self.top_vertex2[0]) / 2,
                                         (self.top_vertex1[1] + self.top_vertex2[1]) / 2, z2]
                elif box_aspect_ratio < fov_aspect_ratio:
                    # 竖向直线
                    self.geometric_type = 5
                    # box的宽作为最小视野的宽
                    fov_min_w = box_w
                    fov_min_h = fov_min_w * fov_aspect_ratio
                    # 当box位于最小视野的最上边时，是最小视野中心范围的最下边
                    fov_min_t_x1 = box[0]
                    fov_min_t_y1 = box[1]
                    fov_min_center_x2 = fov_min_t_x1 + (fov_min_w / 2)
                    fov_min_center_y2 = fov_min_t_y1 + (fov_min_h / 2)
                    # 当box位于最小视野的最下边时，是最小视野中心范围的最上边
                    fov_min_b_x2 = box[2]
                    fov_min_b_y2 = box[3]
                    fov_min_center_x1 = fov_min_b_x2 - (fov_min_w / 2)
                    fov_min_center_y1 = fov_min_b_y2 - (fov_min_h / 2)
                    z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                    if z2 > z_max or z2 < 0:
                        z2 = z_max
                    if z2 > 1:
                        self.geometric_type = 1
                        z1 = z2 - 1
                        self.min_z = z1
                        zs = generate_sequence(z1, z2, 0.1)
                        for z in zs:
                            l, h = inverse_translate_Z(z, viewing_angle_dict)
                            l_h = (l / 2)
                            h_h = (h / 2)
                            f_c_x1 = box[2] - l_h
                            f_c_y1 = box[3] - h_h
                            f_c_x2 = box[0] + l_h
                            f_c_y2 = box[1] + h_h
                            f_c_h_x = (f_c_x1 + f_c_x2) / 2
                            f_c_h_y = (f_c_y1 + f_c_y2) / 2
                            pt1 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y1)
                            pt2 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y1)
                            pt3 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y2)
                            pt4 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y2)
                            pt12 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y1)
                            pt23 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_h_y)
                            pt34 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y2)
                            pt41 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_h_y)
                            h_ps = [pt1, pt12, pt2, pt23, pt3, pt34, pt4, pt41]

                            if (shrink > 0):
                                self.shrink_towards_center(h_ps, shrink)
                            n_hps = [z, h_ps]
                            self.hierarchical_point_set.append(n_hps)

                        # 求质心
                        self.centroid = compute_3d_centroid(self.hierarchical_point_set)
                    else:
                        pt1 = absoute_coordinate_test_invert(pos_now, fov_min_center_x1, fov_min_center_y1)
                        pt2 = absoute_coordinate_test_invert(pos_now, fov_min_center_x2, fov_min_center_y2)
                        self.top_vertex1 = [pt1[0], pt1[1], z2]
                        self.top_vertex2 = [pt2[0], pt2[1], z2]
                        self.bottom_vertex1 = [0, 0, 0]
                        self.bottom_vertex2 = [0, 0, 0]
                        self.bottom_vertex3 = [0, 0, 0]
                        self.bottom_vertex4 = [0, 0, 0]
                        self.centroid = [(self.top_vertex1[0] + self.top_vertex2[0]) / 2,
                                         (self.top_vertex1[1] + self.top_vertex2[1]) / 2, z2]
            else:

                # 把z的范围先算出来 最小到最大.去顶化，把z最大值减小0.1，然后还是算8个点。 最终也是一个类锥体
                self.geometric_type = 1
                p = 0
                z1 = translate_Z(fov_max_w, fov_max_h, viewing_angle_dict)
                if z1 < 0 or z1 > z_max:
                    z1 = z_max
                z2 = 0

                if box_aspect_ratio > fov_aspect_ratio:
                    p = 0
                    # box的高作为最小视野的高
                    fov_min_h = box_h
                    fov_min_w = fov_min_h / fov_aspect_ratio
                    z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                elif box_aspect_ratio < fov_aspect_ratio:
                    p = 1
                    # box的宽作为最小视野的宽
                    fov_min_w = box_w
                    fov_min_h = fov_min_w * fov_aspect_ratio
                    z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                else:
                    p = 2
                    fov_min_w = box_w
                    fov_min_h = box_h
                    z2 = translate_Z(box_w, box_h, viewing_angle_dict)
                if z2 < 0 or z2 > z_max:
                    z2 = z_max

                if z2 == z1 == z_max:
                    print("注意！！方框", g_index, "要求最小放大倍数也超出本摄像机最大倍率！")
                    z1 = z2 - 1

                self.min_z = z1
                self.max_z = z2
                # 去顶！
                if z2 - z1 > 1:
                    z2 = z2 - 1
                zs = generate_sequence(z1, z2, 0.1)

                for i, z in enumerate(zs):
                    l, h = inverse_translate_Z(z, viewing_angle_dict)
                    l_h = (l / 2)
                    h_h = (h / 2)
                    f_c_x1 = box[2] - l_h
                    f_c_y1 = box[3] - h_h
                    f_c_x2 = box[0] + l_h
                    f_c_y2 = box[1] + h_h
                    f_c_h_x = (f_c_x1 + f_c_x2) / 2
                    f_c_h_y = (f_c_y1 + f_c_y2) / 2
                    pt1 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y1)
                    pt2 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y1)
                    pt3 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_y2)
                    pt4 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_y2)
                    pt12 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y1)
                    pt23 = absoute_coordinate_test_invert(pos_now, f_c_x2, f_c_h_y)
                    pt34 = absoute_coordinate_test_invert(pos_now, f_c_h_x, f_c_y2)
                    pt41 = absoute_coordinate_test_invert(pos_now, f_c_x1, f_c_h_y)
                    h_ps = [pt1, pt12, pt2, pt23, pt3, pt34, pt4, pt41]

                    if (shrink > 0):
                        self.shrink_towards_center(h_ps, shrink)
                    n_hps = [z, h_ps]
                    self.hierarchical_point_set.append(n_hps)

                # 求质心
                self.centroid = compute_3d_centroid(self.hierarchical_point_set)

                # print(self.hierarchical_point_set)
                # print(self.centroid)

    def compute_centroid(self, points):
        pts = np.array(points)
        if pts.shape[1] == 2:
            z = pts[0][2] if pts.shape[1] > 2 else 0
            pts = np.hstack([pts, np.full((pts.shape[0], 1), z)])
        return np.mean(pts, axis=0)

    def compute_bounds(self, points):
        points = np.array(points)
        return np.max(points, axis=0), np.min(points, axis=0)

    def compute_plane_equation(self, p1, p2, p3, inside_point):
        p1 = np.asarray(p1)
        p2 = np.asarray(p2)
        p3 = np.asarray(p3)
        if p1.shape[0] == 2:
            p1 = np.append(p1, 0)
            p2 = np.append(p2, 0)
            p3 = np.append(p3, 0)

        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        a, b, c = normal
        d = np.dot(normal, p1)

        # 判断 inside_point 是否满足 a^T x ≤ d
        ip = np.asarray(inside_point)
        if np.dot(normal, ip) > d:
            # 方向错了，翻转
            normal = -normal
            d = -d

        return normal.tolist(), d

    def compute_polyhedron_planes(self, top, bottom,q_c):
        planes = []
        inside_point = q_c

        a, d = self.compute_plane_equation(top[0], top[1], top[2], inside_point)
        planes.append((a, d))

        a, d = self.compute_plane_equation(bottom[0], bottom[1], bottom[2], inside_point)
        planes.append((a, d))

        for i in range(4):
            a1, d1 = self.compute_plane_equation(bottom[i], bottom[(i + 1) % 4], top[i], inside_point)
            planes.append((a1, d1))

        return planes

    def generate_polyhedron_string(self):
        result = []
        count = 1
        all_points = []
        for i in range(len(self.hierarchical_point_set)):
            z, pts2d = self.hierarchical_point_set[i]
            all_points.extend([np.array([x, y, z]) for x, y in pts2d])

        q_c = self.compute_centroid(all_points)
        ub, lb = self.compute_bounds(all_points)

        result.append(f"\nS{self.g_index + 1}:")
        result.append(f"\tq_c:\t{q_c[0]:9.4f}\t{q_c[1]:9.4f}\t{q_c[2]:9.4f}")
        result.append(f"\tub:\t{ub[0]:9.4f}\t{ub[1]:9.4f}\t{ub[2]:9.4f}")
        result.append(f"\tlb:\t{lb[0]:9.4f}\t{lb[1]:9.4f}\t{lb[2]:9.4f}")
        result.append("")
        for i in range(len(self.hierarchical_point_set) - 1):
            z_b, bottom_2d = self.hierarchical_point_set[i]
            z_t, top_2d = self.hierarchical_point_set[i + 1]

            bottom = [np.array([x, y, z_b]) for x, y in bottom_2d]
            top = [np.array([x, y, z_t]) for x, y in top_2d]

            def same(p1, p2):
                return np.linalg.norm(np.array(p1) - np.array(p2)) < 1e-4

            pt_12_t = np.mean([top[0], top[1]], axis=0)
            pt_34_t = np.mean([top[2], top[3]], axis=0)
            pt_x_t = np.mean([pt_12_t, pt_34_t], axis=0)

            pt_12_b = np.mean([bottom[0], bottom[1]], axis=0)
            pt_34_b = np.mean([bottom[2], bottom[3]], axis=0)
            pt_x_b = np.mean([pt_12_b, pt_34_b], axis=0)

            is_folded_top_h = (
                    same(top[0], top[7]) and same(top[7], top[4]) and
                    same(top[1], top[2]) and same(top[2], top[3]) and
                    same(top[5], top[6])
            )

            is_folded_top_v = (
                    same(top[0], top[1]) and same(top[1], pt_12_t) and
                    same(top[3], top[4]) and same(top[4], pt_34_t) and
                    same(top[6], top[7]) and same(top[7], pt_x_t)
            )

            if is_folded_top_h or is_folded_top_v:
                wedge_pairs = [
                    ([bottom[0], bottom[1], bottom[2], bottom[3]], [top[0], top[1], top[2], top[3]]),
                    ([bottom[3], bottom[4], bottom[5], bottom[6]], [top[3], top[4], top[5], top[6]])
                ]
            else:
                wedge_pairs = [
                    ([bottom[0], bottom[1], pt_x_b, bottom[7]], [top[0], top[1], pt_x_t, top[7]]),
                    ([bottom[1], bottom[2], bottom[3], pt_x_b], [top[1], top[2], top[3], pt_x_t]),
                    ([pt_x_b, bottom[3], bottom[4], bottom[5]], [pt_x_t, top[3], top[4], top[5]]),
                    ([bottom[7], pt_x_b, bottom[5], bottom[6]], [top[7], pt_x_t, top[5], top[6]])
                ]

            for j, (b_face, t_face) in enumerate(wedge_pairs):
                all_pts = b_face + t_face
                q_c = self.compute_centroid(all_pts)
                ub, lb = self.compute_bounds(all_pts)
                planes = self.compute_polyhedron_planes(t_face, b_face,q_c)
                A = [p[0] for p in planes]
                B = [p[1] for p in planes]
                result.append("\t" + "=" * 82)
                lines = [f"\tQ{count}:", "\t\tShape: Polyhedra"]
                lines.append(f"\t\tq_c:\t{q_c[0]:9.4f}\t{q_c[1]:9.4f}\t{q_c[2]:9.4f}")
                lines.append(f"\t\tub:\t{ub[0]:9.4f}\t{ub[1]:9.4f}\t{ub[2]:9.4f}")
                lines.append(f"\t\tlb:\t{lb[0]:9.4f}\t{lb[1]:9.4f}\t{lb[2]:9.4f}")
                lines.append(f"\t\tA:")
                for row in A:
                    lines.append(f"\t\t\t\t{row[0]:9.4f}\t{row[1]:9.4f}\t{row[2]:9.4f}")
                lines.append(f"\t\tb:")
                for b in B:
                    lines.append(f"\t\t\t\t{b:9.4f}")
                result.append("\n".join(lines))
                count += 1

        result.append("\n")
        result.append("=" * 91)
        result.append("=" * 91)
        return "\n".join(result)

    def get_g_index(self):
        return self.g_index

    def get_function_expression(self):
        return self.function_expression

    def is_point_inside(self, point):
        # 1. 检查点的 z 坐标是否在范围内
        if point[2] < self.min_z or point[2] > self.max_z:
            return False

        # 2. 找到最近的两个横截面
        low, high = 0, len(self.hierarchical_point_set) - 1

        while low < high - 1:
            mid = (low + high) // 2
            if self.hierarchical_point_set[mid][0] > point[2]:
                high = mid
            else:
                low = mid

        # 3. 获取上下两个横截面
        lower = self.hierarchical_point_set[low]
        upper = self.hierarchical_point_set[high]

        # 处理特殊情况：如果两个横截面 z 值相同，直接使用 lower
        if upper[0] == lower[0]:
            interpolated_points = lower[1]  # 避免除以零
        else:
            # 4. 计算插值权重
            t = (point[2] - lower[0]) / (upper[0] - lower[0])
            interpolated_points = [
                [
                    p1[0] + t * (p2[0] - p1[0]),
                    p1[1] + t * (p2[1] - p1[1]),
                    point[2]
                ]
                for p1, p2 in zip(lower[1], upper[1])
            ]

        # 5. 使用射线法判断点是否在插值横截面内部
        return self.is_point_in_polygon(point, interpolated_points)

    def is_point_in_polygon(self, point, polygon):
        count = 0
        M = len(polygon)

        for i in range(M):
            p1 = polygon[i]
            p2 = polygon[(i + 1) % M]  # 取下一个点，形成一条边

            # 1. 跳过水平边，避免重复计算交点
            if p1[1] == p2[1]:
                continue

            # 2. 让 p1 在下，p2 在上，保证一致性
            if p1[1] > p2[1]:
                p1, p2 = p2, p1

            # 3. 判断点的 y 是否在 (p1.y, p2.y] 之间
            if point[1] > p1[1] and point[1] <= p2[1]:
                # 计算交点的 x 坐标
                intersect_x = p1[0] + (point[1] - p1[1]) / (p2[1] - p1[1]) * (p2[0] - p1[0])

                # 4. 处理射线穿过顶点的情况
                if point[1] == p2[1] and p2[1] > polygon[(i + 2) % M][1]:
                    continue  # 忽略顶点

                # 5. 交点在 p 右侧，计入交点数
                if intersect_x > point[0]:
                    count += 1

        return count % 2 == 1  # 奇数个交点 -> 在多边形内部

    def bottom_shrink_towards_center(self, factor=0):
        """
        缩进底面
        """
        # 计算中心点（每个坐标分别求平均）
        center = [
            (self.bottom_vertex1[0] + self.bottom_vertex2[0] + self.bottom_vertex3[0] + self.bottom_vertex4[0]) / 4.0,
            (self.bottom_vertex1[1] + self.bottom_vertex2[1] + self.bottom_vertex3[1] + self.bottom_vertex4[1]) / 4.0,
            (self.bottom_vertex1[2] + self.bottom_vertex2[2] + self.bottom_vertex3[2] + self.bottom_vertex4[2]) / 4.0
        ]

        # 计算收缩后的新坐标
        self.bottom_vertex1 = [center[i] + (self.bottom_vertex1[i] - center[i]) * (1 - factor) for i in range(3)]
        self.bottom_vertex2 = [center[i] + (self.bottom_vertex2[i] - center[i]) * (1 - factor) for i in range(3)]
        self.bottom_vertex3 = [center[i] + (self.bottom_vertex3[i] - center[i]) * (1 - factor) for i in range(3)]
        self.bottom_vertex4 = [center[i] + (self.bottom_vertex4[i] - center[i]) * (1 - factor) for i in range(3)]

    def shrink_towards_center(self, vs, factor=0):
        if factor == 0:
            return
        else:
            """
            缩进底面
            """
            # 计算中心点（每个坐标分别求平均）
            center = [
                (vs[0][0] + vs[1][0] + vs[2][0] + vs[3][0] + vs[4][0] + vs[5][0] + vs[6][0] + vs[7][0]
                 ) / 8.0,
                (vs[0][1] + vs[1][1] + vs[2][1] + vs[3][1] + vs[4][1] + vs[5][1] + vs[6][1] + vs[7][1]
                 ) / 8.0,
            ]

            # 计算收缩后的新坐标
            vs[0] = [center[i] + (vs[0][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[1] = [center[i] + (vs[1][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[2] = [center[i] + (vs[2][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[3] = [center[i] + (vs[3][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[4] = [center[i] + (vs[4][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[5] = [center[i] + (vs[5][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[6] = [center[i] + (vs[6][i] - center[i]) * (1 - factor) for i in range(2)]
            vs[7] = [center[i] + (vs[7][i] - center[i]) * (1 - factor) for i in range(2)]

    def top_shrink_towards_center(self, factor=0):
        center = [
            (self.top_vertex1[0] + self.top_vertex2[0]) / 2.0,
            (self.top_vertex1[1] + self.top_vertex2[1]) / 2.0,
            (self.top_vertex1[2] + self.top_vertex2[2]) / 2.0
        ]
        # 计算收缩后的新坐标
        self.top_vertex1 = [center[i] + (self.top_vertex1[i] - center[i]) * (1 - factor) for i in range(3)]
        self.top_vertex2 = [center[i] + (self.top_vertex2[i] - center[i]) * (1 - factor) for i in range(3)]

    def get_all_p(self):
        if self.geometric_type == 1:
            return [self.centroid, self.top_vertex1, self.bottom_vertex1, self.bottom_vertex2, self.bottom_vertex3,
                    self.bottom_vertex4]
        elif self.geometric_type == 2 or self.geometric_type == 3:
            return [self.centroid, self.top_vertex1, self.top_vertex2, self.bottom_vertex1, self.bottom_vertex2,
                    self.bottom_vertex3,
                    self.bottom_vertex4]
        else:
            return []

    def get_geometry_info_2(self):

        centroid_str = " ".join(map(str, self.centroid))

        function_expression_str = "\n".join(
            " ".join(map(str, row)) for row in self.function_expression
        )

        # 返回组合后的字符串
        return f"{centroid_str}\n{function_expression_str}"

    def get_geometry_info(self):

        centroid_str = " ".join(self._float_to_hex(x) for x in self.centroid)

        function_expression_str = "\n".join(
            " ".join(self._float_to_hex(x) for x in row) for row in self.function_expression
        )

        # 返回组合后的字符串
        return f"{centroid_str}\n{function_expression_str}"

    def is_point_in_polyhedron(self, point):
        """
        判断点是否在多面体内，使用射线法。
        :param point: 目标点，格式为 (x, y, z)
        :param faces: 多面体的面，格式为 [(v1, v2, v3), ...]，每个面是三个顶点的索引
        :param vertices: 顶点列表，格式为 [(x1, y1, z1), ...]
        :return: 1 -> 点在多面体内部，0 -> 点在外部，-1 -> 点在边界上
        """
        faces = self.faces
        vertices = self.vertices
        ray_direction = np.array([1, 0, 0])  # 射线方向可以任意选择
        intersections = 0

        for face in faces:
            v0, v1, v2 = [vertices[i] for i in face]
            # 计算平面法向量
            normal = np.cross(v1 - v0, v2 - v0)
            if np.dot(normal, point - v0) == 0:
                # 点在平面上，进一步判断是否在边界上
                if point_on_edge(point, v0, v1) or point_on_edge(point, v1, v2) or point_on_edge(point, v2, v0):
                    return -1  # 点在边界上

            # 判断射线与面是否相交
            # 如果面与射线相交，则增计数
            if is_ray_intersecting_face(point, ray_direction, v0, v1, v2):
                intersections += 1

        return 1 if intersections % 2 == 1 else 0

    def get_centroid(self):
        return self.centroid

    def get_access_point(self):
        return self.access_point

    def set_access_point(self, n_access_point):
        self.access_point = n_access_point

    # 打印自己
    def p_self(self):
        if self.geometric_type == 0:
            return {}
        return {
            'g_index': self.g_index,
            'bottom_vertex1': self.bottom_vertex1,
            'bottom_vertex2': self.bottom_vertex2,
            'bottom_vertex3': self.bottom_vertex3,
            'bottom_vertex4': self.bottom_vertex4,
            'top_vertex1': self.top_vertex1,
            'top_vertex2': self.top_vertex2,
            'centroid': self.centroid,
        }

    # 判断点是否在一个不规则的四边形中
    def is_inside_polygon(self, point, polygon_vertices):
        # 计算点到多边形每个三角形的面积
        def triangle_area(a, b, c):
            return abs((a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2.0)

        total_area = 0
        for i in range(len(polygon_vertices)):
            a = polygon_vertices[i]
            b = polygon_vertices[(i + 1) % len(polygon_vertices)]  # 下一个顶点（循环回到起点）
            total_area += triangle_area(a, b, point)

        # 计算多边形的总面积
        polygon_area = 0
        for i in range(len(polygon_vertices)):
            a = polygon_vertices[i]
            b = polygon_vertices[(i + 1) % len(polygon_vertices)]
            polygon_area += triangle_area(a, b, polygon_vertices[0])  # 使用第一个顶点与其他边构成三角形

        return abs(total_area - polygon_area) < 1e-9  # 使用一个小的阈值判断是否相等

    def is_inside_polygon_ray(self, point, polygon_vertices):
        x, y, z = point
        n = len(polygon_vertices)
        inside = False

        p1x, p1y, _ = polygon_vertices[0]
        for i in range(1, n + 1):
            p2x, p2y, _ = polygon_vertices[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    # 定义用于检查点是否在层四边形内部的函数
    def is_inside_rectangle(self, point, layer_vertices):
        # 判断点是否在矩形范围内
        x, y = point[0], point[1]
        x_min, x_max = min(layer_vertices[:, 0]), max(layer_vertices[:, 0])
        y_min, y_max = min(layer_vertices[:, 1]), max(layer_vertices[:, 1])
        return x_min <= x <= x_max and y_min <= y <= y_max

    def get_gridify_data(self, step):
        if self.geometric_type == 0:
            return []
        """
        #几何类型，0初始目标框已经满足比例则无几何体生成，1四面锥体(top顶点只有1个)，2横向楔形体(top顶点2个，是横向的)，3竖向楔形体(top顶点2个，是竖向的)，4只有一个横线，5只有一个竖线
        self.geometric_type = 0
        """
        if self.geometric_type == 2 or self.geometric_type == 3:
            # 定义楔形体的底面和顶面顶点
            bottom_vertices = np.array([
                self.bottom_vertex1,
                self.bottom_vertex2,
                self.bottom_vertex3,
                self.bottom_vertex4
            ])
            top_vertices = np.array([
                self.top_vertex1,
                self.top_vertex2,
            ])

            # 计算 z 方向的层数和步长
            z_min, z_max = bottom_vertices[0][2], top_vertices[0][2]

            # 确保 z_min 和 z_max 是端点，中间按 step 生成整小数
            z_values = [z_min] + list(
                np.arange(np.ceil((z_min + step) / step) * step, np.floor(z_max / step) * step, step)) + [z_max]

            # 创建用于存储所有网格点的集合
            grid_points = []

            # 逐层生成网格点
            for z in z_values:
                t = (z - z_min) / (z_max - z_min)

                # 计算当前层的矩形顶点，通过插值得到，并保留小数点后1位
                layer_vertices = np.round(
                    (1 - t) * bottom_vertices + t * np.vstack(
                        [top_vertices[0], top_vertices[1], top_vertices[1], top_vertices[0]]
                    ),
                    decimals=1
                )

                # 获取当前层的 x 和 y 范围，并确保中间值是 step 的整小数
                x_min, x_max = np.min(layer_vertices[:, 0]), np.max(layer_vertices[:, 0])
                y_min, y_max = np.min(layer_vertices[:, 1]), np.max(layer_vertices[:, 1])

                x_values = [round(x_min, 1)] + list(
                    np.round(np.arange(np.ceil((x_min + step) / step) * step, np.floor(x_max / step) * step, step), 1)
                ) + [round(x_max, 1)]

                y_values = [round(y_min, 1)] + list(
                    np.round(np.arange(np.ceil((y_min + step) / step) * step, np.floor(y_max / step) * step, step), 1)
                ) + [round(y_max, 1)]

                # 在该层的矩形范围内生成网格点
                for x in x_values:
                    for y in y_values:
                        point = (x, y, z)
                        if self.is_inside_polygon_ray(point, layer_vertices):
                            grid_points.append(point)

            # 转换为列表格式，方便查看结果
            grid_points = np.array(grid_points)
            print(f"g_index:{self.g_index} 点数: {len(grid_points)}")
            # print(grid_points)
            return grid_points

    def add_p_to_set(self, step, set: PointIndexSet, g_index):
        list = self.get_gridify_data(step)
        for d in list:
            index = set.add_point(d, g_index)
            self.p_list.append(index)

    def get_points(self):
        return self.p_list

    def _float_to_hex(self, value):
        """将64位浮点数转换为二进制的十六进制字符串"""
        return struct.pack('>d', value).hex()

    def get_gosa_input_json(self):
        # 将质心转换为十六进制字符串
        centroid_hex = [self._float_to_hex(x) for x in self.centroid]

        # 将每个面的约束条件转换为十六进制字符串
        constraints_hex = []
        for plane in self.function_expression:
            plane_hex = [self._float_to_hex(x) for x in plane]
            constraints_hex.append(plane_hex)

        # 构建最终的 JSON 结构
        gosa_input = {
            "centroid": centroid_hex,
            "constraints": constraints_hex
        }

        # 将字典转换为 JSON 字符串
        return json.dumps(gosa_input)

    def get_gosa_input_json_cs(self):
        centroid = self.centroid

        hierarchical_point_set = self.hierarchical_point_set

        # 构建最终的 JSON 结构
        gosa_input = {
            "centroid": centroid,
            "cross_sections": hierarchical_point_set
        }
        # print(gosa_input)
        # 将字典转换为 JSON 字符串
        return json.dumps(gosa_input)


def float_to_hex(value):
    """将64位浮点数转换为二进制的十六进制字符串"""
    return struct.pack('>d', value).hex()



def draw_custom_face(xy: np.ndarray, z_val: float, color=(0.7, 0.3, 0.3)):
    """
    使用指定的三角形划分方式绘制顶面或底面，避免自动三角剖分。
    xy: (8, 2) 顺时针排列的顶点坐标
    z_val: 所有点的 z 坐标
    """
    triangles = [
        (0, 1, 7),
        (1, 2, 3),
        (3, 4, 5),
        (5, 6, 7),
    ]

    # 计算中点 x = midpoint(p1, p5)
    p1 = xy[1]
    p5 = xy[5]
    x = (p1 + p5) / 2.0
    x_index = 8
    extended_xy = np.vstack([xy, x])  # shape: (9, 2)

    # 添加中心扩展三角形
    triangles += [
        (1, 7, x_index),
        (1, 3, x_index),
        (3, 5, x_index),
        (5, 7, x_index),
    ]

    # 构造顶点
    x_all = extended_xy[:, 0]
    y_all = extended_xy[:, 1]
    z_all = np.full_like(x_all, z_val)

    final_triangles = []
    for tri in triangles:
        i, j, k = tri
        # 如果三点中有重复点就跳过（用 np.allclose 更安全）
        pi = extended_xy[i]
        pj = extended_xy[j]
        pk = extended_xy[k]
        if np.allclose(pi, pj) or np.allclose(pj, pk) or np.allclose(pk, pi):
            continue
        final_triangles.append([i, j, k])

    mlab.triangular_mesh(x_all, y_all, z_all, np.array(final_triangles), color=color, opacity=0.6)


def set_white_background():
    """
    设置白色背景和黑色坐标标签颜色（更适合论文插图）
    """
    fig = mlab.gcf()
    fig.scene.background = (1, 1, 1)  # 背景设为白色


def draw_coordinate_axes_with_ticks(origin=(0, 0, 0), length=10, tick_interval=1,
                                    axis_color=(0, 1, 0), tick_color=(0, 0.5, 0),
                                    line_width=1.0, tick_length=0.2):
    """
    绘制嵌入式坐标轴，含刻度，背景白色

    Parameters:
        origin: 坐标原点 (x, y, z)
        length: 每条轴的长度
        tick_interval: 刻度间隔
        axis_color: 坐标轴主线颜色
        tick_color: 刻度线颜色
        line_width: 主轴线粗细
        tick_length: 刻度线长度
    """
    x0, y0, z0 = origin

    def draw_axis(start, end, label, axis_dir):
        mlab.plot3d([start[0], end[0]],
                    [start[1], end[1]],
                    [start[2], end[2]],
                    color=axis_color,
                    tube_radius=None,
                    line_width=line_width)
        # 终点文字
        mlab.text3d(end[0], end[1], end[2], label, color=axis_color, scale=0.5)

        # 画刻度
        for i in range(1, int(length / tick_interval)):
            pos = np.array(start) + i * tick_interval * axis_dir
            ortho_dir = np.cross(axis_dir, [0, 0, 1])  # 垂直方向
            if np.allclose(ortho_dir, 0):
                ortho_dir = np.cross(axis_dir, [0, 1, 0])  # fallback
            ortho_dir = ortho_dir / np.linalg.norm(ortho_dir) * tick_length / 2
            p1 = pos - ortho_dir
            p2 = pos + ortho_dir
            mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                        color=tick_color, tube_radius=None, line_width=0.5)

    # X 轴
    draw_axis(origin, (x0 + length, y0, z0), 'X', np.array([1, 0, 0]))
    # Y 轴
    draw_axis(origin, (x0, y0 + length, z0), 'Y', np.array([0, 1, 0]))
    # Z 轴
    draw_axis(origin, (x0, y0, z0 + length), 'Z', np.array([0, 0, 1]))



def draw_auto_coordinate_axes_from_geometries(Geometrics, tick_interval=5,
                                              axis_color=(24/255, 104/255, 178/255),
                                              tick_color=(24/255, 104/255, 178/255),
                                              line_width=3.0,
                                              tick_length=0.2,
                                              label_offset=0.8):
    import numpy as np
    from mayavi import mlab

    all_points = []
    for g in Geometrics:
        if g.geometric_type == 1 and hasattr(g, 'hierarchical_point_set'):
            for z_val, layer in g.hierarchical_point_set:
                for p in layer:
                    all_points.append([p[0], p[1], z_val])
        else:
            if hasattr(g, 'top_vertex1'):
                all_points.append(g.top_vertex1)
            if hasattr(g, 'top_vertex2'):
                all_points.append(g.top_vertex2)

    all_points = np.array(all_points)
    if all_points.size == 0:
        print("❗ 无有效坐标数据")
        return

    # 原始 min/max 值
    raw_mins = np.min(all_points, axis=0)
    raw_maxs = np.max(all_points, axis=0)

    # ✅ 起点更靠近几何体（原 -5 改为 -1）
    x0 = int(np.floor(raw_mins[0]) )
    y0 = int(np.floor(raw_mins[1]) )
    z0 = int(np.floor(raw_mins[2]) - 1)

    # ✅ 终点为 max - 5，Z 再缩成 2/3
    x1 = int(np.ceil((raw_maxs[0] ) / tick_interval) * tick_interval)
    y1 = int(np.ceil((raw_maxs[1]) / tick_interval) * tick_interval)
    z1 = int(np.ceil((raw_maxs[2] )/ (tick_interval)) * tick_interval)

    def draw_axis(p_start, p_end, label, direction, base_value, label_direction):
        mlab.plot3d([p_start[0], p_end[0]],
                    [p_start[1], p_end[1]],
                    [p_start[2], p_end[2]],
                    color=axis_color, tube_radius=None, line_width=line_width)

        # # ❌ 不再绘制 PTZ 文字标签
        # label_pos = np.array(p_end) + label_offset * np.array(label_direction)
        # mlab.text3d(label_pos[0], label_pos[1], label_pos[2],
        #             label, color=axis_color, scale=0.5)

        length = np.linalg.norm(np.array(p_end) - np.array(p_start))
        num_ticks = int(length // tick_interval)
        for i in range(num_ticks + 1):
            tick_pos = np.array(p_start) + direction * i * tick_interval

            ortho = np.cross(direction, [0, 0, 1])
            if np.allclose(ortho, 0):
                ortho = np.cross(direction, [0, 1, 0])
            ortho = ortho / np.linalg.norm(ortho) * tick_length / 2

            p1_, p2_ = tick_pos - ortho, tick_pos + ortho
            mlab.plot3d([p1_[0], p2_[0]],
                        [p1_[1], p2_[1]],
                        [p1_[2], p2_[2]],
                        color=tick_color, tube_radius=None, line_width=0.3)

            if i != 0:
                mlab.text3d(tick_pos[0], tick_pos[1], tick_pos[2],
                            f"{base_value + i * tick_interval}", color=tick_color, scale=0.5)

    # # ✅ 绘制 P 轴（X轴）- 标签向上
    # draw_axis((x0, y0, z0), (x1, y0, z0), 'P', np.array([1, 0, 0]), base_value=x0, label_direction=[-2, 0, 2])
    # # ✅ 绘制 T 轴（Y轴）- 标签向下
    # draw_axis((x0, y0, z0), (x0, y1, z0), 'T', np.array([0, 1, 0]), base_value=y0, label_direction=[0, -2, 2])
    # # ✅ 绘制 Z 轴（Z轴）- 标签向右
    # draw_axis((x0, y0, z0), (x0, y0, z1), 'Z', np.array([0, 0, 1]), base_value=z0, label_direction=[0, 2, -2])
    # ✅ 绘制 P 轴（X轴）- 标签向上
    draw_axis((x0, y0, z0), (x1, y0, z0), 'P', np.array([1, 0, 0]), base_value=x0, label_direction=[-2, 0, 2])
    # ✅ 绘制 T 轴（Y轴）- 标签向下
    draw_axis((x0, y0, z0), (x0, y1, z0), 'T', np.array([0, 1, 0]), base_value=y0, label_direction=[0, -2, 2])
    # ✅ 绘制 Z 轴（Z轴）- 标签向右
    draw_axis((x0, y0, z0), (x0, y0, z1), 'Z', np.array([0, 0, 1]), base_value=z0, label_direction=[0, 2, -2])





def generate_geometric_shapes_show(Geometrics, pos_now):
    # 坐标系是和二维图片相反的
    """
    -------->y
    |
    |
    x
    """

    for g in Geometrics:
        if g.geometric_type == 1:
            data = g.hierarchical_point_set
            # 提取 z, x, y
            z_values = np.array([d[0] for d in data])
            xy_layers = np.array([d[1] for d in data])  # 形状: (层数, 8, 2)
            num_layers, num_vertices, _ = xy_layers.shape

            # 构建 3D 点云
            X = xy_layers[:, :, 0].flatten()
            Y = xy_layers[:, :, 1].flatten()
            Z = np.repeat(z_values, num_vertices)

            # 生成侧面网格 faces = (N, 3)
            faces = []
            for i in range(num_layers - 1):
                bottom_start = i * num_vertices
                top_start = (i + 1) * num_vertices
                for j in range(num_vertices):
                    next_j = (j + 1) % num_vertices
                    faces.append([bottom_start + j, bottom_start + next_j, top_start + j])
                    faces.append([top_start + j, bottom_start + next_j, top_start + next_j])

            faces = np.array(faces)

            if len(faces) > 0:
                # 绘制侧面
                mlab.triangular_mesh(X, Y, Z, faces, color=(222/255, 88/255, 43/255), opacity=0.6)


            draw_custom_face(xy_layers[0], z_values[0], color=(222/255, 88/255, 43/255))  # 底面
            draw_custom_face(xy_layers[-1], z_values[-1], color=(222/255, 88/255, 43/255))  # 顶面


        elif g.geometric_type in [4, 5]:
            x = [g.top_vertex1[0], g.top_vertex2[0]]
            y = [g.top_vertex1[1], g.top_vertex2[1]]
            z = [g.top_vertex1[2], g.top_vertex2[2]]
            mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.1)

        # 绘制中心点和索引标签
        c_x, c_y, c_z = g.get_centroid()
        mlab.points3d(c_x, c_y, c_z, color=(1, 1, 1), scale_factor=0.3)
        mlab.text3d(c_x, c_y, c_z, str(g.g_index + 1), color=(1, 1, 1), scale=1, orient_to_camera=True)

    set_white_background()
    draw_auto_coordinate_axes_from_geometries(Geometrics)

    # 绘制当前点
    mlab.points3d(pos_now[0], pos_now[1], pos_now[2], color=(0, 0, 0), scale_factor=0.3)
    mlab.show()


def generate_geometric_shapes_show_path(Geometrics, pos_now, path_data):
    # 坐标系是和二维图片相反的
    """
    -------->y
    |
    |
    x
    """

    for g in Geometrics:
        if g.geometric_type == 1:
            data = g.hierarchical_point_set
            # 提取 z, x, y
            z_values = np.array([d[0] for d in data])
            xy_layers = np.array([d[1] for d in data])  # 形状: (层数, 8, 2)
            num_layers, num_vertices, _ = xy_layers.shape

            # 构建 3D 点云
            X = xy_layers[:, :, 0].flatten()
            Y = xy_layers[:, :, 1].flatten()
            Z = np.repeat(z_values, num_vertices)

            # 生成侧面网格 faces = (N, 3)
            faces = []
            for i in range(num_layers - 1):
                bottom_start = i * num_vertices
                top_start = (i + 1) * num_vertices
                for j in range(num_vertices):
                    next_j = (j + 1) % num_vertices
                    faces.append([bottom_start + j, bottom_start + next_j, top_start + j])
                    faces.append([top_start + j, bottom_start + next_j, top_start + next_j])

            faces = np.array(faces)

            if len(faces) > 0:
                # 绘制侧面
                mlab.triangular_mesh(X, Y, Z, faces, color=(222/255, 88/255, 43/255), opacity=0.6)

            draw_custom_face(xy_layers[0], z_values[0], color=(222/255, 88/255, 43/255))  # 底面
            draw_custom_face(xy_layers[-1], z_values[-1], color=(222/255, 88/255, 43/255))  # 顶面


        elif g.geometric_type in [4, 5]:
            x = [g.top_vertex1[0], g.top_vertex2[0]]
            y = [g.top_vertex1[1], g.top_vertex2[1]]
            z = [g.top_vertex1[2], g.top_vertex2[2]]
            mlab.plot3d(x, y, z, color=(1, 0, 0), tube_radius=0.1)

        # 绘制中心点和索引标签
        c_x, c_y, c_z = g.get_centroid()
        mlab.points3d(c_x, c_y, c_z, color=(1, 1, 1), scale_factor=0.5)
        mlab.text3d(c_x, c_y, c_z, str(g.g_index + 1), color=(1, 1, 1), scale=1, orient_to_camera=True)

    # 绘制当前点
    mlab.points3d(pos_now[0], pos_now[1], pos_now[2], color=(0, 0, 0), scale_factor=1)

    points = path_data['coords']

    points = np.array(points)  # 转换为 NumPy 数组
    points = np.vstack([points, points[0]])
    # 分离 x, y, z 坐标
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # 绘制点
    mlab.points3d(x, y, z, color=(58/255, 186/255, 109/255), scale_factor=0.8,opacity=0.8)

    # 绘制连线
    mlab.plot3d(x, y, z, color=(58/255, 186/255, 109/255), tube_radius=0.3,opacity=0.8)
    mlab.points3d(pos_now[0], pos_now[1], pos_now[2], color=(58/255, 186/255, 109/255), scale_factor=1)

    set_white_background()
    draw_auto_coordinate_axes_from_geometries(Geometrics)

    # 显示场景
    mlab.show()


# 目标框生成楔形体（三种顶:横向、竖向、点）
def box_to_geometric(pos_now, box, proportion, fov_w, fov_h, g_index, z_max, viewing_angle_dict, shrink=0):
    # viewing_angle_x = viewing_angle_dict['x']
    # viewing_angle_y = viewing_angle_dict['y']
    geometric = Geometric(g_index, pos_now, box, proportion, fov_w, fov_h, z_max, viewing_angle_dict, shrink)
    return geometric


def is_restricted(new_box, existing_boxes, proportion):
    x1_new, y1_new, x2_new, y2_new = new_box

    width_new = x2_new - x1_new
    height_new = y2_new - y1_new
    area_new = width_new * height_new

    # 基本合法性检查
    if width_new < 2 or height_new < 2:
        return True
    if area_new < 10 or area_new > 200 * 300:
        return True
    if area_new > canvas_width * canvas_height * proportion:
        return True

    # 高宽比控制
    aspect_ratio = height_new / width_new
    if aspect_ratio > 5 or aspect_ratio < 0.3:
        return True

    # 检查是否嵌套或严重重叠
    for box in existing_boxes:
        x1, y1, x2, y2 = box
        width_existing = x2 - x1
        height_existing = y2 - y1
        area_existing = width_existing * height_existing

        # 嵌套检查（完全包含）
        if (x1_new > x1 and y1_new > y1 and x2_new < x2 and y2_new < y2) or \
           (x1 > x1_new and y1 > y1_new and x2 < x2_new and y2 < y2_new):
            return True

        # 重叠面积检查
        inter_x1 = max(x1, x1_new)
        inter_y1 = max(y1, y1_new)
        inter_x2 = min(x2, x2_new)
        inter_y2 = min(y2, y2_new)

        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        inter_area = inter_width * inter_height

        if inter_area > 0:
            overlap_ratio_new = inter_area / area_new
            overlap_ratio_existing = inter_area / area_existing
            # 超过80%视为严重重叠，不允许
            if overlap_ratio_new > 0.8 or overlap_ratio_existing > 0.8:
                return True

    return False


def DEC2HEX_doc(x):
    # 十六进制转化为十进制
    x = int(str(int(x * 10)), 16)
    return x


def scale_box(box, scale_factor, img_width, img_height):  # 放大后，框的中心点不变（对于边界附近的框，自动限制放大比例）
    x1, y1, x2, y2 = box

    # 计算当前框的中心点和宽高
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    # 计算当前框到画面边界的最大允许缩放比例
    max_scale_x = min(
        (cx) / (width / 2),  # 左边界的限制
        (img_width - cx) / (width / 2)  # 右边界的限制
    )
    max_scale_y = min(
        (cy) / (height / 2),  # 上边界的限制
        (img_height - cy) / (height / 2)  # 下边界的限制
    )

    # 取最小的缩放比例，确保框不会超出画面
    max_scale = min(max_scale_x, max_scale_y)

    # 如果目标缩放比例大于最大允许缩放比例，则使用最大允许缩放比例
    if scale_factor > max_scale:
        scale_factor = max_scale

    # 计算放大后的宽高
    new_width = width * scale_factor
    new_height = height * scale_factor

    # 计算新的左上角和右下角坐标
    new_x1 = max(0, cx - new_width / 2)
    new_y1 = max(0, cy - new_height / 2)
    new_x2 = min(img_width, cx + new_width / 2)
    new_y2 = min(img_height, cy + new_height / 2)

    # 返回新的框
    return [new_x1, new_y1, new_x2, new_y2]


# def get_all_Boundary_point(boxes,pos_now,proportion,fov_w,fov_h):
#     geometrics = []
#     i = 0
#     r_p = []
#     for box in boxes:
#         n_p = proportion * 1.21
#         n_box = scale_box(box, 1.1, fov_w, fov_h)
#         # 把框等比放大两倍，再把要求比例也放大到两倍
#         if n_p > 0.7:
#             n_p = 0.7
#
#         g = box_to_geometric(pos_now, n_box, proportion, fov_w, fov_h, i,0.09,0.09)
#         i = i + 1
#         geometrics.append(g)
#     for g in geometrics:
#         if g.geometric_type != 0:
#             ps = g.get_all_p()
#             for p in ps:
#                 r_p.append(p)
#
#     return r_p

def base_box_ptz(pos_now, box, proportion, fov_w, fov_h, z_max, viewing_angle_dict):
    # 目标框宽
    box_w = box[2] - box[0]
    # 目标框高
    box_h = box[3] - box[1]
    # 目标框像素面积
    box_area = box_w * box_h
    # 视野像素面积
    fov_area = fov_w * fov_h
    # 视野的高宽比
    fov_aspect_ratio = fov_h / fov_w
    # 目标框高宽比
    box_aspect_ratio = box_h / box_w
    now_p = box_area / fov_area
    b_c_x = (box[2] + box[0]) / 2
    b_c_y = (box[3] + box[1]) / 2
    pt = absoute_coordinate_test_invert(pos_now, b_c_x, b_c_y)
    if now_p >= proportion:
        return None
    else:
        # 最大的视野面积
        fov_max_area = box_area / proportion
        fov_max_w = (fov_max_area / fov_aspect_ratio) ** 0.5
        fov_max_h = fov_max_w * fov_aspect_ratio
        if fov_max_h < box_h or fov_max_w < box_w:
            # 框太瘦高，达到面积比例，但是超出视野了，那就最小视野和最大视野一致（都是视野），变为一条直线
            if box_aspect_ratio > fov_aspect_ratio:

                # box的高作为最小视野的高
                fov_min_h = box_h
                fov_min_w = fov_min_h / fov_aspect_ratio

                z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                if z2 > z_max or z2 < 0:
                    z2 = z_max
                return [pt[0], pt[1], z2]
            elif box_aspect_ratio < fov_aspect_ratio:
                # box的宽作为最小视野的宽
                fov_min_w = box_w
                fov_min_h = fov_min_w * fov_aspect_ratio
                z2 = translate_Z(fov_min_w, fov_min_h, viewing_angle_dict)
                if z2 > z_max or z2 < 0:
                    z2 = z_max
                return [pt[0], pt[1], z2]
        else:
            z2 = translate_Z(fov_max_w, fov_max_h, viewing_angle_dict)
            if z2 > z_max or z2 < 0:
                z2 = z_max
            return [pt[0], pt[1], z2]


def snapshot_base(boxes, pos_now, proportion, scale_factor, shrink, fov_w, fov_h, z_max, viewing_angle_dict):
    """

        :param boxes:所有检测框
        :param pos_now: ptz起点(当前ptz)
        :param proportion: 最小的放大比例
        :param fov_w: 当前视野的像素宽(画布宽)
        :param fov_h: 当前视野的像素长(画布长)
        :return:
        """
    print("所有的框:", boxes)
    f_ps = []
    for box in boxes:
        n_p = scale_factor * scale_factor
        n_box = scale_box(box, scale_factor, fov_w, fov_h)
        # 把框等比放大两倍，再把要求比例也放大到两倍
        if n_p > 0.7:
            n_p = 0.7
        ptz = base_box_ptz(pos_now, n_box, proportion, fov_w, fov_h, z_max, viewing_angle_dict)
        if ptz != None:
            f_ps.append(ptz)
    return f_ps


def boxs_to_gs(boxes, pos_now, proportion, scale_factor, shrink, fov_w, fov_h, z_max, viewing_angle_dict):
    """
    :param boxes:所有检测框
    :param pos_now: ptz起点(当前ptz)
    :param proportion: 最小的放大比例
    :param fov_w: 当前视野的像素宽(画布宽)
    :param fov_h: 当前视野的像素长(画布长)
    :return:
    """
    print("所有的框:", boxes)
    geometrics = []
    i = 0

    for box in boxes:
        n_p = scale_factor * scale_factor
        n_box = scale_box(box, scale_factor, fov_w, fov_h)
        # 把框等比放大两倍，再把要求比例也放大到两倍
        if n_p > 0.7:
            n_p = 0.7

        # n_p = proportion
        # n_box = box

        # print("框转几何体:",i,n_box)
        g = box_to_geometric(pos_now, n_box, proportion, fov_w, fov_h, i, z_max, viewing_angle_dict, shrink)
        i = i + 1
        geometrics.append(g)
    return geometrics


def snapshot_gsoa(boxes, pos_now, proportion, scale_factor, shrink, fov_w, fov_h, z_max, viewing_angle_dict):
    """
    :param boxes:所有检测框
    :param pos_now: ptz起点(当前ptz)
    :param proportion: 最小的放大比例
    :param fov_w: 当前视野的像素宽(画布宽)
    :param fov_h: 当前视野的像素长(画布长)
    :return:
    """
    print("所有的框:", boxes)
    geometrics = []
    i = 0

    for box in boxes:
        n_p = scale_factor * scale_factor
        n_box = scale_box(box, scale_factor, fov_w, fov_h)
        # 把框等比放大两倍，再把要求比例也放大到两倍
        if n_p > 0.7:
            n_p = 0.7

        # n_p = proportion
        # n_box = box

        # print("框转几何体:",i,n_box)
        g = box_to_geometric(pos_now, n_box, proportion, fov_w, fov_h, i, z_max, viewing_angle_dict, shrink)
        i = i + 1
        geometrics.append(g)
    # 起点
    first_geometric = {
        "centroid": [
            pos_now[0],  # x
            pos_now[1],  # y
            pos_now[2]  # z
        ]
    }
    # 生成其他几何体的 JSON 数据
    targets = [first_geometric]  # 第一个几何体
    for g in geometrics:
        if g.geometric_type == 1 or g.geometric_type == 4 or g.geometric_type == 5:
            gosa_json = json.loads(g.get_gosa_input_json_cs())  # 调用 get_gosa_input_json 函数
            targets.append(gosa_json)
    t_l = len(targets)
    if t_l > 1:
        # 组合成最终的 JSON 结构
        final_json = {
            "targets": targets
        }

        # 将最终的 JSON 结构转换为字符串
        final_json_str = json.dumps(final_json, )
        # 发送 POST 请求到 API
        api_url = "http://172.30.249.71:8888/api/gsoa"  # 本机的wsl2
        # api_url = "http://10.156.2.57:8888/api/gsoa"
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
            print("接口返回json:", path_data)
            print("路线长度（包含起点）:", len(path), ",目标个数（包含起点）:", t_l)
            # 利用起点重新排序，并把起点删除
            new_path = generate_new_path(path, pos_now)

            # 用另一个进程来显示几何体
            p = multiprocessing.Process(target=generate_geometric_shapes_show, args=(geometrics, pos_now,))
            # 启动进程
            p.start()
            # 用另一个进程来显示路线
            p2 = multiprocessing.Process(target=generate_geometric_shapes_show_path,
                                         args=(geometrics, pos_now, path_data))
            # 启动进程
            p2.start()

            # 判断路线中的访问点是否真正的访问到了所有的几何体
            if check_path_access(new_path, geometrics):
                print("所有的几何体都被访问到啦")
            else:
                print("有几何体都被漏啦！！！！！！！！！！！")

            return new_path
        else:
            # 如果请求失败，抛出异常或返回错误信息
            raise Exception(f"API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}")


    else:
        return []

def g_gs_str(geometrics, pos_now):
    # 起点
    first_geometric = {
        "centroid": [
            pos_now[0],  # x
            pos_now[1],  # y
            pos_now[2]  # z
        ]
    }
    # 生成其他几何体的 JSON 数据
    targets = [first_geometric]  # 第一个几何体
    for g in geometrics:
        if g.geometric_type == 1 or g.geometric_type == 4 or g.geometric_type == 5:
            gosa_json = json.loads(g.get_gosa_input_json_cs())  # 调用 get_gosa_input_json 函数
            targets.append(gosa_json)
    t_l = len(targets)
    # 组合成最终的 JSON 结构
    final_json = {
            "targets": targets
    }

    # 将最终的 JSON 结构转换为字符串ls
    final_json_str = json.dumps(final_json, )
    return final_json_str


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
            # 用另一个进程来显示路线
            p2 = multiprocessing.Process(target=generate_geometric_shapes_show_path,
                                         args=(geometrics, pos_now, path_data))
            # 启动进程
            p2.start()

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
def check_path_access(new_path, geometrics):
    """
    检查 new_path 中的点是否访问了 geometrics 中的所有几何体
    :param new_path: 需要检查的三维点路径 [list of [x, y, z]]
    :param geometrics: 几何体数组 [list of Geometric objects]
    :return: 如果所有几何体都被访问到，返回 True；否则返回 False
    """
    r = True
    # 遍历每个几何体
    for geometric in geometrics:
        visited = False  # 标志是否访问了当前几何体

        # 遍历路径中的每个点
        for point in new_path:
            if point[0] == geometric.centroid[0] and point[1] == geometric.centroid[1] and point[2] == \
                    geometric.centroid[2]:
                visited = True  # 如果点在几何体内部，标记为已访问
                break  # 跳出当前点的检查，继续检查下一个几何体
            if geometric.geometric_type == 1:
                if geometric.is_point_inside(point):
                    visited = True  # 如果点在几何体内部，标记为已访问
                    break  # 跳出当前点的检查，继续检查下一个几何体

        if not visited:
            # 如果有任何一个几何体没有被访问到，则返回 False
            print(geometric.g_index, "几何体被漏了")
            r = False

    # 如果所有几何体都被访问到，返回 True
    return r


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


# ============ 1. 量化区间遍历函数 =============
def quantized_range(low, high, step=0.1):
    """
    在 [low, high] 范围内，以 step=0.1 为单位，在浮点数网格上取值。
    例如当 low=0.1234, high=0.567 时, 结果生成 0.2, 0.3, 0.4, 0.5.
    """
    # 找到不小于 low 的最小 0.1 倍数
    start = math.ceil(low / step) * step
    # 找到不大于 high 的最大 0.1 倍数
    end = math.floor(high / step) * step
    # 逐步产出
    x = start
    while x <= end + 1e-9:  # 考虑浮点误差做一点冗余
        # 可以根据需要保留一位小数，也可直接返回浮点数
        yield round(x, 1)
        x += step


# ============ 2. 2D 多边形内判定(射线法) =============
def point_in_polygon_2d(px, py, polygon2d):
    """
    判断 (px, py) 是否在 polygon2d 内。
    polygon2d: [(x1, y1), (x2, y2), ..., (xn, yn)]
    """
    cnt = 0
    n = len(polygon2d)
    for i in range(n):
        x1, y1 = polygon2d[i]
        x2, y2 = polygon2d[(i + 1) % n]
        # 仅在 py 落在 y1,y2 之间时，判断是否与射线相交
        if (y1 <= py < y2) or (y2 <= py < y1):
            # 计算与 y=py 相交时的 x 坐标
            intersect_x = x1 + (py - y1) * (x2 - x1) / (y2 - y1 + 1e-15)
            if intersect_x > px:
                cnt += 1
    return (cnt % 2) == 1


# ============ 3. 按 0.1 网格化对 2D 多边形采样 =============
def sample_polygon_2d(z, polygon2d, step=0.1):
    """
    在 polygon2d 的外包矩形内，以 0.1(或 step) 为网格间隔对 (x, y) 采样。
    返回所有落在多边形内的 (x, y, z_量化)。
    """
    if not polygon2d:
        return []
    xs = [p[0] for p in polygon2d]
    ys = [p[1] for p in polygon2d]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    # 把 z 也量化到 0.1 网格
    z_quant = round(z / step) * step

    sampled_points = []
    # 依次在 [min_x, max_x], [min_y, max_y] 的范围上取 0.1 步长
    for qx in quantized_range(min_x, max_x, step):
        for qy in quantized_range(min_y, max_y, step):
            if point_in_polygon_2d(qx, qy, polygon2d):
                sampled_points.append((qx, qy, z_quant))
    return sampled_points


def polygon_from_8pts(h_ps):
    """
    给定本层 8 个顶点(2D坐标,不含z), 返回多边形顶点序列 polygon2d。
    h_ps 如 [ (x1,y1), (x2,y2), ..., (x8,y8) ]。
    需要保证顶点按顺序围成一个闭合多边形。
    """
    return [(p[0], p[1]) for p in h_ps]


# ============ 4. 生成几何体内“可行点”集合(按 0.1 网格) =============
def generate_feasible_points_for_one_geometry(geom, step=0.1):
    """
    对单个几何体 geom，其 hierarchical_point_set = [
      [z1, [8顶点2D]], [z2, [8顶点2D]], ...
    ]
    逐层进行 0.1 网格化采样。若仅想对每个层面做离散(不插值)，则直接对每层 polygon 采样即可。
    返回所有采样点列表 [(x, y, z), ...]。
    """
    all_pts = []
    for z_layer, eight_pts_2d in geom.hierarchical_point_set:
        polygon2d = polygon_from_8pts(eight_pts_2d)
        pts_layer = sample_polygon_2d(z_layer, polygon2d, step=step)
        all_pts.extend(pts_layer)
    return all_pts


# ============ 5. 构造全局点集 & 每点覆盖的几何体集合 =============
def point_in_geom(p, geom, step=0.1):
    """
    简化版判断 3D点 p=(x,y,z) 是否属于几何体 geom。
    做法：在 geom.hierarchical_point_set 找到“离 p.z 最近”的层，取其 2D 多边形做射线法判断 (x,y)。
    如果需要严格考虑层与层之间插值，可自行添加插值逻辑。
    """
    x, y, z = p
    min_dist = float('inf')
    nearest_poly = None
    for z_layer, eight_pts_2d in geom.hierarchical_point_set:
        dist_ = abs(z_layer - z)
        if dist_ < min_dist:
            min_dist = dist_
            nearest_poly = polygon_from_8pts(eight_pts_2d)
    # 判断 (x,y) 是否在 nearest_poly 内
    return point_in_polygon_2d(x, y, nearest_poly)


def build_global_point_set(geom_list, step=0.1):
    """
    将所有几何体的可行采样点合并成一个全局点集 (global_points)。
    再计算每个点能覆盖哪些几何体 (cover_sets)。
    返回:
      global_points: List[(x,y,z)]
      cover_sets   : List[ set_of_geom_ids ]
    """
    # 1) 分别生成每个几何体自己的候选点集合
    geom_points_list = []
    for g in geom_list:
        pts = generate_feasible_points_for_one_geometry(g, step=step)
        geom_points_list.append(pts)

    # 2) 合并所有点(可能有重复坐标)
    tmp_all = []
    for pts in geom_points_list:
        tmp_all.extend(pts)

    # 3) 去重(可选)，用字典/集合聚合
    unique_map = {}
    for p in tmp_all:
        # 用 (x,y,z) 保留一位小数作为 key
        key = (round(p[0], 1), round(p[1], 1), round(p[2], 1))
        unique_map[key] = key
    global_points = list(unique_map.values())

    # 4) 计算覆盖集合
    cover_sets = []
    for p in global_points:
        covers = set()
        for i, g in enumerate(geom_list):
            if point_in_geom(p, g, step):
                covers.add(i)
        cover_sets.append(covers)
    return global_points, cover_sets


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

def get_path_time_cost(pts, closed):
    total_cost = 0.0
    for i in range(1, len(pts)):
        total_cost += travel_time(pts[i - 1], pts[i])
    if closed and len(pts) > 1:
        total_cost += travel_time(pts[-1], pts[0])
    return total_cost


class State:
    __slots__ = ("current_idx", "covered_set", "cost_so_far", "path")

    def __init__(self, current_idx, covered_set, cost_so_far, path):
        self.current_idx = current_idx
        self.covered_set = covered_set  # frozenset
        self.cost_so_far = cost_so_far
        self.path = path  # [节点idx列表]

    def __lt__(self, other):
        return self.cost_so_far < other.cost_so_far


def branch_and_bound_tspn(global_points, cover_sets, start_point, all_geoms):
    print("开始分支限界")
    """
    TSPN 分支限界:
      - 状态包含 (current_idx, covered_set, cost_so_far, path)
      - covered_set 是已覆盖几何体的集合
      - 当 covered_set == all_geoms 时，再回起点计算总代价，若更优则更新
    """
    n = len(global_points)
    # 将起点也加入 global_points，索引为 n
    extended_global_points = global_points + [start_point]
    extended_cover_sets = cover_sets + [set()]  # 起点不覆盖任何几何体
    idx_start = n  # 起点索引

    best_cost = float('inf')
    best_path = None

    # 初始状态
    init_state = State(idx_start, frozenset(), 0.0, [idx_start])
    pq = []
    heapq.heappush(pq, init_state)

    visited = {}  # 用于剪枝: (current_idx, covered_set) -> minimal_cost

    # 初始化进度条
    total_nodes = len(global_points) * len(global_points)  # 估计最大节点数，实际更少
    processed_nodes = 0

    while pq:
        top = heapq.heappop(pq)
        curr_idx = top.current_idx
        covered = top.covered_set
        cost_sf = top.cost_so_far
        path_ = top.path

        # 如果当前代价 >= 已知最优解, 可剪枝
        if cost_sf >= best_cost:
            continue

        # 若已覆盖所有几何体, 回起点
        if covered == all_geoms:
            cost_back = travel_time(extended_global_points[curr_idx], start_point)
            total_cost = cost_sf + cost_back
            if total_cost < best_cost:
                best_cost = total_cost
                best_path = path_ + [idx_start]
            continue

        # 尝试走向其它点
        for nxt_idx in range(len(extended_global_points)):
            if nxt_idx == curr_idx:
                continue
            # 计算从 curr_idx -> nxt_idx 的开销
            c = travel_time(extended_global_points[curr_idx], extended_global_points[nxt_idx])
            new_cost = cost_sf + c
            if new_cost >= best_cost:
                continue

            # 更新覆盖
            new_cover = set(covered).union(extended_cover_sets[nxt_idx])
            new_cover_fs = frozenset(new_cover)

            # 剪枝: 如果之前访问过 (nxt_idx, new_cover_fs) 且 cost 更小，则无需再扩展
            old_best = visited.get((nxt_idx, new_cover_fs), float('inf'))
            if new_cost < old_best:
                visited[(nxt_idx, new_cover_fs)] = new_cost
                new_path = path_ + [nxt_idx]
                heapq.heappush(pq, State(nxt_idx, new_cover_fs, new_cost, new_path))

        # 更新进度条
        processed_nodes += 1
        progress = (processed_nodes / total_nodes) * 100
        print(f"进度: {progress:.2f}% ({processed_nodes}/{total_nodes} 节点)")

    return best_path, best_cost


# ============ 8. 主函数封装 =============
def solve_3d_tspn(geom_list, start_point, step=0.1):
    """
    主流程:
      1) 对所有几何体做 0.1 网格化采样, 得到全局点集 global_points 以及每点覆盖集合 cover_sets
      2) 分支限界搜索最优路径
      3) 返回 (最优访问点的坐标序列, 代价)
    """
    # 1) 建立全局点集
    global_points, cover_sets = build_global_point_set(geom_list, step=step)
    all_geoms = set(range(len(geom_list)))

    # 2) 分支限界 TSPN
    best_path, best_cost = branch_and_bound_tspn(global_points, cover_sets, start_point, all_geoms)

    # 3) 还原坐标序列
    #   global_points 的索引范围是 [0..n-1], 起点索引是 n
    route_in_coords = []
    if best_path is not None:
        n = len(global_points)
        for idx in best_path:
            if idx < n:
                route_in_coords.append(global_points[idx])
            else:
                route_in_coords.append(start_point)

    return route_in_coords, best_cost
def generate_single_point_polyhedron_string(f_name,pos_now):
    x, y, z = [f"{v:.4f}" for v in pos_now]

    result = []
    result.append(f"{f_name}               l = 3")
    result.append("")
    result.append("=" * 91)
    result.append("=" * 91)
    result.append(f"S0:")
    result.append(f"\tq_c:\t{x}\t{y}\t{z}")
    result.append(f"\tub:\t{x}\t{y}\t{z}")
    result.append(f"\tlb:\t{x}\t{y}\t{z}")
    result.append("")
    result.append("\t" + "=" * 82)
    result.append(f"\tQ1:")
    result.append(f"\t\tShape: Polyhedra")
    result.append(f"\t\tq_c:\t{x}\t{y}\t{z}")
    result.append(f"\t\tub:\t{x}\t{y}\t{z}")
    result.append(f"\t\tlb:\t{x}\t{y}\t{z}")
    result.append(f"\t\tA:")
    result.extend([
        f"\t\t\t1.0\t0.0\t0.0",
        f"\t\t\t-1.0\t0.0\t0.0",
        f"\t\t\t0.0\t1.0\t0.0",
        f"\t\t\t0.0\t-1.0\t0.0",
        f"\t\t\t0.0\t0.0\t1.0",
        f"\t\t\t0.0\t0.0\t-1.0"
    ])
    result.append(f"\t\tb:")
    result.extend([
        f"\t\t\t{x}",
        f"\t\t\t-{x}",
        f"\t\t\t{y}",
        f"\t\t\t-{y}",
        f"\t\t\t{z}",
        f"\t\t\t-{z}"
    ])
    result.append("")
    result.append("=" * 91)
    result.append("=" * 91)

    return "\n".join(result)

def compute_std_and_mean(data):
    mean_val = np.mean(data)
    std_dev = np.std(data)
    rsd = std_dev / mean_val * 100
    return rsd,mean_val

import csv

def evaluate_one_sample(length_lists, time_lists):
    all_lengths = [l for sublist in length_lists for l in sublist]
    L_ref = min(all_lengths)

    per_algo_stats = []
    for lengths, times in zip(length_lists, time_lists):
        best = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        avg_time = sum(times) / len(times)

        pdb = (best - L_ref) / L_ref * 100
        pdm = (avg_len - L_ref) / L_ref * 100

        per_algo_stats.append((
            round(pdb, 4),
            round(pdm, 4),
            round(avg_time, 4)
        ))
    return L_ref, per_algo_stats

def evaluate_one_sample_v2(sample_id, length_lists, time_lists):
    """
    参数：
        sample_id: str 样本编号
        length_lists: list of list，每个算法的路径长度列表
        time_lists: list of list，每个算法的运行时间列表
    返回：
        可直接作为 append_result() 输入的值
    """
    # 全部路径中找 Lref
    all_lengths = [l for sublist in length_lists for l in sublist]
    L_ref = min(all_lengths)

    # 逐个算法计算 PDB, PDM, 平均时间
    stats = []
    for lengths, times in zip(length_lists, time_lists):
        best = min(lengths)
        avg_len = sum(lengths) / len(lengths)
        avg_time = sum(times) / len(times)

        pdb = (best - L_ref) / L_ref * 100
        pdm = (avg_len - L_ref) / L_ref * 100

        stats.append((pdb, pdm, avg_time))

    # 拆出三组结果
    (base_pdb, base_pdm, base_time), \
    (imp_pdb, imp_pdm, imp_time), \
    (gro_pdb, gro_pdm, gro_time) = stats

    return (
        sample_id, L_ref,
        base_pdb, base_pdm, base_time,
        imp_pdb, imp_pdm, imp_time,
        gro_pdb, gro_pdm, gro_time
    )



def append_result(
    sample_id, Lref,
    base_pdb, base_pdm,base_std_l, base_std_m, base_best, base_avg, base_time,
    imp_pdb, imp_pdm,imp_std_l, imp_std_m, imp_best, imp_avg, imp_time,
    gro_pdb, gro_pdm,gro_std_l, gro_std_m, gro_best, gro_avg, gro_time,
    filename='results.csv'
):
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            sample_id, Lref,
            f"{base_pdb:.2f}", f"{base_pdm:.2f}",f"{base_std_l:.2f}", f"{base_std_m:.2f}", f"{base_best:.2f}", f"{base_avg:.2f}", f"{base_time:.2f}",
            f"{imp_pdb:.2f}", f"{imp_pdm:.2f}",f"{imp_std_l:.2f}", f"{imp_std_m:.2f}", f"{imp_best:.2f}", f"{imp_avg:.2f}", f"{imp_time:.2f}",
            f"{gro_pdb:.2f}", f"{gro_pdm:.2f}",f"{gro_std_l:.2f}", f"{gro_std_m:.2f}", f"{gro_best:.2f}", f"{gro_avg:.2f}", f"{gro_time:.2f}"
        ])

if __name__ == '__main__':
            # random.seed(21341)  # 42 是种子值，你可以换成任意整数
            random.seed(32432)  # 42 是种子值，你可以换成任意整数
            # random.seed(98)  # 42 是种子值，你可以换成任意整数

            # for b_n in [30]:
            #     for i in range(1, 2):
            f_name = f"test"
            print("==================================================================")
            print(f_name)
            best_cost = float('inf')
            best_path = None
            # 记录开始时间
            start_time = time.time()

            proportion = 0.06
            n = 1
            s = 0
            # 画布大小
            canvas_width = 2560
            canvas_height = 1440
            # 起点
            pos_now = [20.2, 43.1, 1.0]
            # 生成随机的几何框个数
            num_boxes = 20
            # boxes = [[384, 1300, 807, 1435]]
            boxes = []
            # boxes = [[0, 0, 150, 200]]
            # boxes = [[1939, 5, 2020, 229], [1681, 546, 1912, 712], [1533, 967, 1610, 1022], [335, 599, 727, 746], [801, 587, 980, 797], [845, 127, 1106, 216], [2287, 618, 2415, 657], [754, 627, 959, 746], [570, 602, 775, 804], [1232, 1105, 1538, 1263]]
            # 生成不嵌套的 box 列表，确保 x2 > x1 且 y2 > y1
            while len(boxes) < num_boxes:
                x1 = random.randint(0, canvas_width)
                x2 = random.randint(0, canvas_width)
                y1 = random.randint(0, canvas_height)
                y2 = random.randint(0, canvas_height)

                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)

                # 可选：限制框的最小宽高，防止生成细长或几乎为点的框
                if (x2 - x1) < 30 or (y2 - y1) < 30:
                    continue  # 太小就重新生成

                new_box = [x1, y1, x2, y2]
                if not is_restricted(new_box, boxes, proportion):
                    boxes.append(new_box)

            # boxes = [[1200, 700, 1380, 850]]
            # 创建画布
            image = Image.new("RGB", (canvas_width, canvas_height), "white")
            draw = ImageDraw.Draw(image)
            # 设置字体（可以选择系统中存在的字体）
            try:
                font = ImageFont.truetype("arial.ttf", 10)  # 尝试加载系统中的字体
            except IOError:
                font = ImageFont.load_default()  # 如果无法加载，使用默认字体
            # boxes.append([1533, 967, 1610, 1022])
            # boxes.append([2287, 618, 2415, 657])
            # 生成随机的[x1, y1, x2, y2]
            i = 0
            for box in boxes:

                # 将坐标转换为画布尺寸
                x1_canvas = box[0]
                y1_canvas = box[1]
                x2_canvas = box[2]
                y2_canvas = box[3]
                n_b = [x1_canvas, y1_canvas, x2_canvas, y2_canvas]

                # g = box_to_geometric(pos_now,n_b,proportion,canvas_width,canvas_height,i,0)
                # geometrics.append(g)
                # 为每个框生成随机颜色
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

                # 绘制矩形框
                draw.rectangle(n_b, outline=color,width=3)

                # 在矩形框中心写上编号
                center_x = (x1_canvas + x2_canvas) // 2
                center_y = (y1_canvas + y2_canvas) // 2
                draw.text((center_x, center_y), str(i+1), fill=color, font=font)
                i = i + 1
            # 打印所有的[x1, y1, x2, y2]数组
            # boxes = [boxes[10]]
            print(boxes)
            str_boxs = str(boxes)
            with open(f"{f_name}-boxs.txt", "w", encoding="utf-8") as f:
                f.write(str_boxs)
            # 保存图像
            # image.save(f"{f_name}.png")
            image.show()

            viewing_angle_dict = {'x': 51.2400016784668, 'y': 30.190000534057617}

            start_time_z = time.perf_counter_ns()
            geometrics = boxs_to_gs(boxes, pos_now, proportion, n, s, canvas_width, canvas_height, 22.5, viewing_angle_dict)
            end_time_z = time.perf_counter_ns()
            u_t_z = end_time_z - start_time_z



            print("建模完成，总用时:" + str(u_t_z / 1000000000) + "-秒:")
            # 用另一个进程来显示几何体
            p = multiprocessing.Process(target=generate_geometric_shapes_show, args=(geometrics, pos_now,))
            # 启动进程
            p.start()

            # 起点
            first_geometric = {
                "centroid": [
                    pos_now[0],  # x
                    pos_now[1],  # y
                    pos_now[2]  # z
                ]
            }
            # 生成其他几何体的 JSON 数据
            targets = [first_geometric]  # 第一个几何体
            for g in geometrics:
                if g.geometric_type == 1 or g.geometric_type == 4 or g.geometric_type == 5:
                    gosa_json = json.loads(g.get_gosa_input_json_cs())  # 调用 get_gosa_input_json 函数
                    targets.append(gosa_json)
            t_l = len(targets)
            if t_l > 1:
                # 组合成最终的 JSON 结构
                final_json = {
                    "targets": targets
                }

                # 将最终的 JSON 结构转换为字符串
                final_json_str = json.dumps(final_json, )
                # 发送 POST 请求到 API
                api_url = "http://172.30.249.71:8888/api/gsoa"  # 本机的wsl2
                # api_url = "http://10.156.2.57:8888/api/gsoa"
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
                    print("接口返回json:", path_data)
                    print("路线长度（包含起点）:", len(path), ",目标个数（包含起点）:", t_l)
                    # 利用起点重新排序，并把起点删除
                    new_path = generate_new_path(path, pos_now)

                    # # 用另一个进程来显示几何体
                    # p = multiprocessing.Process(target=generate_geometric_shapes_show, args=(geometrics, pos_now,))
                    # # 启动进程
                    # p.start()
                    # 用另一个进程来显示路线
                    p2 = multiprocessing.Process(target=generate_geometric_shapes_show_path,
                                                 args=(geometrics, pos_now, path_data))
                    # 启动进程
                    p2.start()
                    #
                    # # 判断路线中的访问点是否真正的访问到了所有的几何体
                    # if check_path_access(new_path, geometrics):
                    #     print("所有的几何体都被访问到啦")
                    # else:
                    #     print("有几何体都被漏啦！！！！！！！！！！！")


                else:
                    # 如果请求失败，抛出异常或返回错误信息
                    raise Exception(f"API 请求失败，状态码: {response.status_code}, 错误信息: {response.text}")

            # gs_str = g_gs_str(geometrics, pos_now)
            # gsoa_j_g_url = "http://10.156.2.57:8888/api/gsoa"
            # j_g_ps, j_g_cost, j_g_t = snapshot_gsoa_gs(10, gs_str, gsoa_j_g_url)



















