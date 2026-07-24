#ifndef __COORDS3D_H__
#define __COORDS3D_H__

#include <vector>
#include <iostream>
#include <algorithm>

/// ----------------------------------------------------------------------------
/// @brief Planar
/// ----------------------------------------------------------------------------
struct Planar {
   double a;
   double b;
   double c;
   double d;

   Planar(Planar &c) : a(c.a), b(c.b), c(c.c),d(c.d) {}
   Planar(const Planar &c) : a(c.a), b(c.b), c(c.c),d(c.d) {}
   Planar() {}
   Planar(double a, double b, double c, double d) : a(a), b(b), c(c), d(d) {}
   Planar& operator=(const Planar &c) {
      if (this != &c) {
         a = c.a;
         b = c.b;
         this->c = c.c;
         d = c.d;
      }
      return *this;
   }

};

typedef std::vector<Planar> PlanarVector;
typedef std::vector<PlanarVector> PlanarVectorVector;

std::ostream& operator<<(std::ostream &os, const Planar &pt);

std::istream& operator>>(std::istream &is, Planar &pt);

std::istream& operator>>(std::istream &is, PlanarVector &pts);

// 结束头文件保护宏
#endif  //__COORDS3D_H__
