
#include <cmath>
#include <algorithm>
#include "route_path_utils.h"
#include <cmath> // 包含 pow 函数
/// - function -----------------------------------------------------------------
double get_path_length(const CoordsVector &pts, bool closed)
{
   double len = 0.;
   for (int i = 1; i < pts.size(); ++i) {
      len += sqrt(pts[i-1].squared_distance(pts[i]));
   }
   if (closed and pts.size() > 1) {
      len += sqrt(pts.back().squared_distance(pts.front()));
   }
   return len;
}


double T_x(double x) {
    x = std::abs(x);
    if (x < 1e-12) {
        return 0.0;
    }
    return 0.106 * std::pow(x, 0.555) + 0.219;
}

double T_y(double y) {
    y = std::abs(y);
    if (y < 1e-12) {
        return 0.0;
    }
    return 0.060 * std::pow(y, 0.753) + 0.233;
}

double T_z(double z) {
    z = std::abs(z);
    if (z < 1e-12) {
        return 0.0;
    }
    return 0.034 * std::pow(z, 1.371) + 0.382;
}

double stay_time(double dx, double dy, double dz) {
    if (std::abs(dx) < 1e-12 &&
        std::abs(dy) < 1e-12 &&
        std::abs(dz) < 1e-12) {
        return 0.0;
    }

    return 0.323 + 0.015 * std::abs(dz);
}
// 定义一个内联函数计算时间代价
double cost(const Coords &a, const Coords &b) {
	double dx = b.x - a.x;
	double dy = b.y - a.y;
	double dz = b.z - a.z;

	return std::max({T_x(dx), T_y(dy), T_z(dz)}) + stay_time(dx, dy, dz);
}

double cost_ts(const Coords_ts &a, const Coords_ts &b) {
	double dx = b.x - a.x;
	double dy = b.y - a.y;
	double dz = b.z - a.z;

	return std::max({T_x(dx), T_y(dy), T_z(dz)}) + stay_time(dx, dy, dz);
}

double get_path_time_cost(const CoordsVector &pts, bool closed)
{
	double total_cost = 0.0;

	for (int i = 1; i < pts.size(); ++i) {
		total_cost += cost(pts[i - 1], pts[i]);
	}

	if (closed && pts.size() > 1) {
		total_cost += cost(pts.back(), pts.front());
	}

	return total_cost;
}

double get_path_time_cost_ts(const Coords_tsVector &pts, bool closed)
{
	double total_cost = 0.0;

	for (int i = 1; i < pts.size(); ++i) {
		total_cost += cost_ts(pts[i - 1], pts[i]);
	}

	if (closed && pts.size() > 1) {
		total_cost += cost_ts(pts.back(), pts.front());
	}

	return total_cost;
}


#define dist(i, j) sqrt(path[i].squared_distance(path[j]))
/// - function -----------------------------------------------------------------
void two_opt(CoordsVector &path) 
{
   const int N = path.size();
   int counter = 0;
   double mchange;
   do {
      mchange = 0.0;
      int mi = -1;
      int mj = -1;
      for (int i = 1; i < N; ++i) {
	 for (int j = i + 1; j < N-1; ++j) {
	    double change = 
	       dist(i-1, j) + dist(i, j+1) - dist(i-1, i) - dist(j, j+1);
	    if (mchange > change) {
	       mchange = change;
	       mi = i; mj = j;
	    }
	    counter += 1;
	 }
      }
      if (mi > 0 and mj > 0) {
	 CoordsVector newPath;
	 for (int i = 0; i < mi; ++i) {
	    newPath.push_back(path[i]);
	 }
	 for (int i = mj; i >= mi; --i) {
	    newPath.push_back(path[i]);
	 }
	 for (int i = mj+1; i < N; ++i) {
	    newPath.push_back(path[i]);
	 }
	 path = newPath;
      }
   } while (mchange < -1e-5);
}


#define cost_ts(i, j) Coords_ts::time_cost_ts(path[i], path[j])
/// - function -----------------------------------------------------------------
void two_opt_cost(Coords_tsVector &path)
{
	const int N = path.size();
	int counter = 0;
	double mchange;
	do {
		mchange = 0.0;
		int mi = -1;
		int mj = -1;
		for (int i = 1; i < N; ++i) {
			for (int j = i + 1; j < N-1; ++j) {
				double change =
				   cost_ts(i-1, j) + cost_ts(i, j+1) - cost_ts(i-1, i) - cost_ts(j, j+1);
				if (mchange > change) {
					mchange = change;
					mi = i; mj = j;
				}
				counter += 1;
			}
		}
		if (mi > 0 and mj > 0) {
			std::reverse(path.begin() + mi, path.begin() + mj + 1);
		}
	} while (mchange < -1e-5);
}



