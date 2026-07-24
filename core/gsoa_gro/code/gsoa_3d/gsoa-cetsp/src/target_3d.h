#ifndef __TARGET3D_H__
#define __TARGET3D_H__


#include "coords_3d.h"

namespace gsoa {

   struct SNeuron;

   struct STarget {
      const int label;
      //横切面
      const CorssSectionVector corssSectionVector;
      const double max_z;
      const double min_z;

      //质心
      const Coords coords;
      int stepWinnerSelected;
      SNeuron *selectedWinner;
      const double radius;

      STarget(const int id,  const CorssSectionVector &corssSectionVector,const double max_z,const double min_z,const Coords &c)
         : label(id),   corssSectionVector(corssSectionVector), max_z(max_z),min_z(min_z),stepWinnerSelected(-1), selectedWinner(nullptr),coords(c),radius(0) {}

      bool isPointInPolygon(const Coords& p, const CoordsVector& polygon) const {
         int count = 0;
         int M = polygon.size();
         for (int i = 0; i < M; ++i) {
            Coords p1 = polygon[i];
            Coords p2 = polygon[(i + 1) % M]; // 取下一个点，形成一条边

            // 1. 跳过水平边，避免重复计算交点
            if (p1.y == p2.y) continue;

            // 2. 让 p1 在下，p2 在上，保证一致性
            if (p1.y > p2.y) std::swap(p1, p2);

            // 3. 判断点的 y 是否在 (p1.y, p2.y] 之间
            if (p.y > p1.y && p.y <= p2.y) {
               // 计算交点的 x 坐标
               double intersectX = p1.x + (p.y - p1.y) / (p2.y - p1.y) * (p2.x - p1.x);
               // 4. 处理射线穿过顶点的情况
               if (p.y == p2.y && p2.y > polygon[(i + 2) % M].y) {
                  continue; // 忽略顶点
               }
               // 5. 交点在 p 右侧，计入交点数
               if (intersectX > p.x) {
                  count++;
               }
            }
         }
         return count % 2 == 1; // 奇数个交点 -> 在多边形内部
      }

      bool isPointInside(const Coords& point) const {
         int n = corssSectionVector.size();
         if (n < 2) return false; // 至少需要两个横截面才能形成柱体

         // 1. 检查点的 z 坐标是否在范围内
         if (point.z < min_z || point.z > max_z) {
            return false;
         }

         // 2. 二分查找找到最近的两个横截面
         int low = 0, high = n - 1;
         while (low < high - 1) { // 找到 `crossSections[low]` 和 `crossSections[high]` 紧邻围住 `point.z`
            int mid = (low + high) / 2;
            if (corssSectionVector[mid].z > point.z) {
               high = mid;
            } else {
               low = mid;
            }
         }

         // 3. 获取上下两个横截面
         const CorssSection& lower = corssSectionVector[low];
         const CorssSection& upper = corssSectionVector[high];

         // 4. 计算插值横截面
         double t = (point.z - lower.z) / (upper.z - lower.z);
         CoordsVector interpolatedPoints;
         int M = lower.cv.size();

         for (int i = 0; i < M; ++i) {
            // 由于点是顺时针存储的，直接按索引进行线性插值
            Coords p1 = lower.cv[i];
            Coords p2 = upper.cv[i];
            Coords interpolated = {
               p1.x + t * (p2.x - p1.x),
               p1.y + t * (p2.y - p1.y),
               point.z
           };
            interpolatedPoints.push_back(interpolated);
         }

         // 5. 使用射线法判断点是否在插值横截面内部
         return isPointInPolygon(point, interpolatedPoints);
      }


      //找到几何体中距离p最近的点
      Coords findIntersection(const Coords &A) const {
         // 如果没有横截面，返回几何体质心
         if (corssSectionVector.empty()) {
            return coords;
         }

         // 如果 A 在几何体内部，直接返回 A
         if (isPointInside(A)) {
            return A;
         }

         // 设 B 为几何体的质心
         Coords B = coords;

         // 确保 p_in 在几何体内部
         Coords p_in = B;

         Coords p_out = A;  // p_out 在几何体外


         // 计算 A 到 B 的初始距离
         double D = std::sqrt((p_out.x - p_in.x) * (p_out.x - p_in.x) +
                              (p_out.y - p_in.y) * (p_out.y - p_in.y) +
                              (p_out.z - p_in.z) * (p_out.z - p_in.z));

         // 设置最小步长为 AB 线段长度的 7%
         double min_step = std::max(1e-3, 0.04 * D); // 防止过小 0.5 0.25 0.125 0.0625 0.03125 0.015625
                                                        //循环次数      1   2     3     4      5       6
         // 设置最大迭代次数，避免无限循环
         int max_iterations = 100;
         int iterations = 0;

         // 二分查找 A -> B 的交点
         while (std::sqrt((p_out.x - p_in.x) * (p_out.x - p_in.x) +
                         (p_out.y - p_in.y) * (p_out.y - p_in.y) +
                         (p_out.z - p_in.z) * (p_out.z - p_in.z)) > min_step &&
                iterations < max_iterations) {
            Coords mid = {
               (p_in.x + p_out.x) / 2.0,
               (p_in.y + p_out.y) / 2.0,
               (p_in.z + p_out.z) / 2.0
           };

            if (isPointInside(mid)) {
               p_in = mid;  // 继续向外部搜索
            } else {
               p_out = mid;  // 继续向内部搜索
            }

            iterations++;
                }

         return p_in;  // 返回靠近边界的内部点
      }


   };

   typedef std::vector<STarget*> TargetPtrVector;

} // end namespace gsoa
// 结束头文件保护宏
#endif  // __TARGET3d_H__




