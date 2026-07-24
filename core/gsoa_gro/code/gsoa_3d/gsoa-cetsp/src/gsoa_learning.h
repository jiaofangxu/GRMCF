/*
 * File name: gsoa_learning.h
 * Date:      2016/12/09 22:53
 * Author:    Jan Faigl
 */

#ifndef __GSOA_LEARNING_H__
#define __GSOA_LEARNING_H__

#include <string> 

#include <crl/config.h>

namespace gsoa {

   struct Schema {
      const double GAIN_DECREASING_RATE; //收益递减比率
      //适应时候的邻近关系
      const double NEIGHBORHOOD_FACTOR; //邻近关系

      const double MIN_GAIN; //最小收益

      const double COMM_RADIUS;
      const double COMM_RADIUS2;
      const double COMM_RADIUS_SMALLER;//更小半径

      static crl::CConfig &getConfig(crl::CConfig &config);
      Schema(crl::CConfig &config);
      ~Schema();

      void updateExp(int n, int step);

      double mi;
      double G; //variable 变量
      int d; //number neighborhood neurons 邻近神经元数量

      double *exps;
      int expN;
      int explen;

   };

   extern Schema *schema;

} // end namespace gsoa

#endif

/* end of gsoa_learning.h */
