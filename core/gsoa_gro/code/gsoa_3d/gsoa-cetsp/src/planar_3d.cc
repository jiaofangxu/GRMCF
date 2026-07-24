#include <cmath>
#include <cstdint>
#include "planar_3d.h"
//重写 对本类使用<<和>>符号的逻辑
/// - public method ------------------------------------------------------------
std::ostream& operator<<(std::ostream &os, const Planar &pt)
{
   return os << pt.a << ", " << pt.b << ", " << pt.c << ", " << pt.d;
}

/// - public method ------------------------------------------------------------
std::istream& operator>>(std::istream &is, Planar &pt)
{
   std::string hexString;
   for (auto *field : {&pt.a, &pt.b, &pt.c, &pt.d}) {
      if (is >> hexString) {
         uint64_t hexValue = std::stoull(hexString, nullptr, 16);
         *field = *reinterpret_cast<double*>(&hexValue);
      } else {
         return is;
      }
   }
   return is;
}

/// - public method ------------------------------------------------------------
std::istream& operator>>(std::istream &is, PlanarVector &pts)
{
   Planar pt;
   while (is >> pt) {
      pts.push_back(pt);
   }
   return is;
}

/* end of planar_3d.cc */