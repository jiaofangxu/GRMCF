#include <iostream>
#include <vector>
#include <cmath>
#include <limits>
#include <chrono>  // 用于计时

// 定义三维点结构
struct Point {
    double x, y, z;
    Point() : x(0), y(0), z(0) {}
    Point(double x, double y, double z) : x(x), y(y), z(z) {}

    // 向量加法
    Point operator+(const Point& p) const {
        return Point(x + p.x, y + p.y, z + p.z);
    }

    // 向量减法
    Point operator-(const Point& p) const {
        return Point(x - p.x, y - p.y, z - p.z);
    }

    // 向量乘法
    Point operator*(double t) const {
        return Point(x * t, y * t, z * t);
    }

    // 点积
    double dot(const Point& p) const {
        return x * p.x + y * p.y + z * p.z;
    }

    // 计算向量的长度
    double length() const {
        return std::sqrt(x * x + y * y + z * z);
    }

    // 向量的单位化
    Point normalize() const {
        double len = length();
        return Point(x / len, y / len, z / len);
    }

    // 计算两点之间的距离
    double distance(const Point& p) const {
        return std::sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (z - p.z) * (z - p.z));
    }
};

// 定义平面结构
struct Plane {
    double A, B, C, D; // 平面方程: Ax + By + Cz = D

    // 判断一个点是否在平面上（Ax + By + Cz <= D）
    bool isPointOnPlane(const Point& p) const {
        return A * p.x + B * p.y + C * p.z <= D;
    }
};

// 判断线段与平面是否相交
bool doesSegmentIntersectPlane(const Point& p1, const Point& p2, const Plane& plane, Point& intersection) {
    Point dir = p2 - p1;
    double denom = plane.A * dir.x + plane.B * dir.y + plane.C * dir.z;

    // 如果线段与平面平行
    if (std::abs(denom) < 1e-6) {
        return false;
    }

    double t = (plane.D - plane.A * p1.x - plane.B * p1.y - plane.C * p1.z) / denom;

    // 计算交点
    if (t < 0 || t > 1) {
        return false; // 交点不在段线段上
    }

    intersection = p1 + dir * t;
    return true;
}

// 计算线段到平面的最短距离
double distanceToPlane(const Point& p1, const Point& p2, const Plane& plane, Point& nearestPoint) {
    Point dir = p2 - p1;
    double denom = plane.A * dir.x + plane.B * dir.y + plane.C * dir.z;

    // 如果线段与平面平行
    if (std::abs(denom) < 1e-6) {
        // 返回线段端点到平面的距离
        double dist1 = std::abs(plane.A * p1.x + plane.B * p1.y + plane.C * p1.z - plane.D) /
                       std::sqrt(plane.A * plane.A + plane.B * plane.B + plane.C * plane.C);
        double dist2 = std::abs(plane.A * p2.x + plane.B * p2.y + plane.C * p2.z - plane.D) /
                       std::sqrt(plane.A * plane.A + plane.B * plane.B + plane.C * plane.C);

        // 选择更小的距离
        if (dist1 < dist2) {
            nearestPoint = p1;
            return dist1;
        } else {
            nearestPoint = p2;
            return dist2;
        }
    }

    double t = (plane.D - plane.A * p1.x - plane.B * p1.y - plane.C * p1.z) / denom;

    // 如果线段和该平面相交
    if (t >= 0 && t <= 1) {
        nearestPoint = p1 + dir * t;
        return 0;
    }

    // 否则计算最近点
    double dist1 = std::abs(plane.A * p1.x + plane.B * p1.y + plane.C * p1.z - plane.D) /
                   std::sqrt(plane.A * plane.A + plane.B * plane.B + plane.C * plane.C);
    double dist2 = std::abs(plane.A * p2.x + plane.B * p2.y + plane.C * p2.z - plane.D) /
                   std::sqrt(plane.A * plane.A + plane.B * plane.B + plane.C * plane.C);

    if (dist1 < dist2) {
        nearestPoint = p1;
        return dist1;
    } else {
        nearestPoint = p2;
        return dist2;
    }
}

// 主函数，计算线段到多面体的最短距离
double segmentToConvexPolyhedronDistance(const Point& p1, const Point& p2, const std::vector<Plane>& polyhedronPlanes, Point& closestPoint) {
    double minDist = std::numeric_limits<double>::infinity();

    for (const auto& plane : polyhedronPlanes) {
        Point nearestPoint;
        double dist = distanceToPlane(p1, p2, plane, nearestPoint);

        if (dist < minDist) {
            minDist = dist;
            closestPoint = nearestPoint;
        }
    }

    return minDist;
}

int main() {
    // 定义一个简单的凸多面体（例如立方体的6个面）
    std::vector<Plane> polyhedronPlanes = {
            {2.240000000000002 ,-36.959999999999994, 27.959999999999997, -144.51199999999983},  // x <= 1
            {12.320000000000002, 1.4000000000000001, 17.039999999999978, 657.38}, // x >= -1
            {-3.6400000000000023, 37.800000000000004, 26.540000000000006, 450.612},  // y <= 1
            {-10.920000000000002, -2.240000000000012, 13.540000000000012, -343.11600000000016}, // y >= -1
            {-0.0, -0.0, -58.47999999999998, -210.52799999999993},  // z <= 1
    };

    // 定义线段的两个端点
    Point p1(0, 0, 2), p2(0, 0, -2);

    Point closestPoint;

    // 开始计时
    auto start = std::chrono::high_resolution_clock::now();

    // 调用计算函数
    double dist = segmentToConvexPolyhedronDistance(p1, p2, polyhedronPlanes, closestPoint);

    // 结束计时
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "The shortest distance: " << dist << std::endl;
    std::cout << "Closest point on the polyhedron: (" << closestPoint.x << ", " << closestPoint.y << ", " << closestPoint.z << ")" << std::endl;
    std::cout << "Time taken: " << duration.count() << " microseconds" << std::endl;

    return 0;
}
