/*
 * File name: coords.cc
 * Date:      2014/01/16 16:54
 * Author:    Jan Faigl
 */

#include <cmath>
#include <cstdint>
#include "coords_3d.h"
//重写 对本类使用<<和>>符号的逻辑
/// - public method ------------------------------------------------------------
std::ostream& operator<<(std::ostream &os, const Coords &pt)
{
   return os << pt.x << ", " << pt.y << ", " << pt.z;
}

/// - public method ------------------------------------------------------------
std::istream& operator>>(std::istream &is, Coords &pt)
{
   std::string hexString;
   is >> hexString;
   uint64_t hexValue = std::stoull(hexString, nullptr, 16);
   pt.x = *reinterpret_cast<double*>(&hexValue);

   is >> hexString;
   hexValue = std::stoull(hexString, nullptr, 16);
   pt.y = *reinterpret_cast<double*>(&hexValue);

   is >> hexString;
   hexValue = std::stoull(hexString, nullptr, 16);
   pt.z = *reinterpret_cast<double*>(&hexValue);
   return is;
}

/// - public method ------------------------------------------------------------
std::istream& operator>>(std::istream &is, CoordsVector &pts)
{
   Coords pt;
   while (is >> pt) {
      pts.push_back(pt);
   }
   return is;
}

/* end of coords.cc */