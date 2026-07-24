/*
 * File name: coords.h
 * Date:      2013/10/13 09:23
 * Author:    Jan Faigl
 */

#ifndef __COORDS_H__
#define __COORDS_H__

#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <unordered_set>
/// ----------------------------------------------------------------------------
/// @brief Coords
/// ----------------------------------------------------------------------------
struct Coords {
   double x;
   double y;
   double z;


   Coords(const Coords &c) : x(c.x), y(c.y), z(c.z) {}
   Coords() {}
   Coords(double x, double y, double z) : x(x), y(y), z(z) {}
   Coords& operator=(const Coords &c) {
      if (this != &c) {
         x = c.x;
         y = c.y;
         z = c.z;
      }
      return *this;
   }
   bool operator==(const Coords& other) const {
      return x == other.x && y == other.y && z == other.z;
   }

   inline static double T_x(double x) {
    x = std::abs(x);
    if (x < 1e-12) {
        return 0.0;
    }
    return 0.106 * std::pow(x, 0.555) + 0.219;
}

inline static double T_y(double y) {
    y = std::abs(y);
    if (y < 1e-12) {
        return 0.0;
    }
    return 0.060 * std::pow(y, 0.753) + 0.233;
}

inline static double T_z(double z) {
    z = std::abs(z);
    if (z < 1e-12) {
        return 0.0;
    }
    return 0.034 * std::pow(z, 1.371) + 0.382;
}

inline static double stay_time(double dx, double dy, double dz) {
    if (std::abs(dx) < 1e-12 &&
        std::abs(dy) < 1e-12 &&
        std::abs(dz) < 1e-12) {
        return 0.0;
    }

    return 0.323 + 0.015 * std::abs(dz);
}

   inline static double time_cost(const Coords &c1, const Coords &c2) {
      double dx = c1.x - c2.x;
      double dy = c1.y - c2.y;
      double dz = c1.z - c2.z;

      return std::max({T_x(dx), T_y(dy), T_z(dz)}) + stay_time(dx, dy, dz);
   }
   inline double squared_distance(const Coords &c) const {
      return squared_distance(*this, c);
   }

   inline static double squared_distance(const Coords &c1, const Coords &c2) {
      double dx = c1.x - c2.x;
      double dy = c1.y - c2.y;
      double dz = c1.z - c2.z;
      return dx*dx + dy*dy + dz*dz;
   }

   inline double time_cost(const Coords &c) const {
      return time_cost(*this, c);
   }






};
struct Coords_ts
{
   double x;
   double y;
   double z;
   std::vector<int> ts;
   Coords_ts(const Coords_ts& c)
    : x(c.x), y(c.y), z(c.z), ts(c.ts) {}
   Coords_ts() {}
   Coords_ts(double x, double y, double z, const std::vector<int>& ts)
    : x(x), y(y), z(z), ts(ts) {}


   inline static double T_x(double x) {
    x = std::abs(x);
    if (x < 1e-12) {
        return 0.0;
    }
    return 0.106 * std::pow(x, 0.555) + 0.219;
}

inline static double T_y(double y) {
    y = std::abs(y);
    if (y < 1e-12) {
        return 0.0;
    }
    return 0.060 * std::pow(y, 0.753) + 0.233;
}

inline static double T_z(double z) {
    z = std::abs(z);
    if (z < 1e-12) {
        return 0.0;
    }
    return 0.034 * std::pow(z, 1.371) + 0.382;
}

inline static double stay_time(double dx, double dy, double dz) {
    if (std::abs(dx) < 1e-12 &&
        std::abs(dy) < 1e-12 &&
        std::abs(dz) < 1e-12) {
        return 0.0;
    }

    return 0.323 + 0.015 * std::abs(dz);
}

   inline static double time_cost_ts(const Coords_ts &c1, const Coords_ts &c2) {
      double dx = c1.x - c2.x;
      double dy = c1.y - c2.y;
      double dz = c1.z - c2.z;

      return std::max({T_x(dx), T_y(dy), T_z(dz)}) + stay_time(dx, dy, dz);
   }
};


typedef std::vector<Coords> CoordsVector;
typedef std::vector<CoordsVector> CoordsVectorVector;
typedef std::vector<Coords_ts> Coords_tsVector;
std::ostream& operator<<(std::ostream &os, const Coords &pt);

std::istream& operator>>(std::istream &is, Coords &pt);

std::istream& operator>>(std::istream &is, CoordsVector &pts);





struct CorssSection {
   double z;


   CoordsVector cv;

   CorssSection(CorssSection &c) : z(c.z), cv(c.cv){}
   CorssSection(const CorssSection &c) : z(c.z), cv(c.cv) {}
   CorssSection() {}
   CorssSection(double z, CoordsVector cv) : z(z), cv(cv) {}
   CorssSection& operator=(const CorssSection &c) {
      if (this != &c) {
         z = c.z;
         cv = c.cv;
      }
      return *this;
   }

};

typedef std::vector<CorssSection> CorssSectionVector;
typedef std::vector<CorssSectionVector> CorssSectionVectorVector;









#endif

/* end of coords.h */
