
#ifndef __GSOA3D_H__
#define __GSOA3D_H__

#include <crl/config.h>
#include <crl/alg/algorithm.h>

#include <crl/gui/shape.h>
#include <unordered_set>
#include "coords_3d.h"
#include "target_3d.h"
#include "gsoa_ring.h"

namespace gsoa {

   class CGSOA : public crl::CAlgorithm {
      typedef crl::CAlgorithm Base;
      typedef std::vector<int> IntVector;
      public:

      static crl::CConfig &getConfig(crl::CConfig &config);

      CGSOA(crl::CConfig &config);
      ~CGSOA();

      std::string getVersion(void);
      std::string getRevision(void);

      CoordsVector api_run(TargetPtrVector targets);
       // 梯度计算函数：对一个 cost 函数在点 B 上进行数值梯度估计
       Coords_ts compute_numerical_gradient(std::function<double(const Coords_ts&)> func,
                                            const Coords_ts& B,
                                            double h,
                                            const std::vector<int>& ts_l);

       // 单个方向归一化
       void normalize(Coords_ts& dir);

       // 判断某点是否在所有指定邻域中
       bool inside_all_regions(const Coords_ts& pt,
                               const TargetPtrVector& targets,
                               const std::vector<int>& ts_l);

       // 对路径中某点 B 进行基于梯度方向的局部优化
       void optimize_B(const Coords_ts& A,
                       Coords_ts& B,
                       const Coords_ts& C,
                       TargetPtrVector targets);

       // 对整个路径进行迭代式的局部爬山优化
       void gro(Coords_tsVector& path,
                          TargetPtrVector targets);


      void visualize(const Coords_tsVector & path);
      void solve(void);
      void print_path_with_neighborhoods(const Coords_tsVector& path,
                                   const TargetPtrVector& targets);
      protected:
      void load(void);
      void initialize(void);
      void after_init(void); 

      void iterate(int iter);
      double refine(int step, double errorMax);

      void save(void);
      void release(void);

      void defineResultLog(void);
      void fillResultRecord(int trial);


      private:
      void drawPath(void);
      void drawRing(int step);
      void savePic(int step, bool detail = false, const std::string &dir_suffix = "");

      void getSolution(int step, Coords_tsVector &solution) const;


      private:
      const int DEPOT_IDX; 
      const bool VARIABLE_RADIUS;
      const bool SAVE_RESULTS;
      const bool SAVE_SETTINGS;
      const bool SAVE_INFO;

      const bool DRAW_RING_ITER;
      const bool DRAW_RING_ENABLE;
      const bool DRAW_RING_NODES;
      const bool DRAW_TOUR_REPRESENTED_BY_RING;
      const bool SAVE_PIC;

      const bool IN_NEIGH_WAYPOINT;

      crl::gui::CShape shapeTargets;
      crl::gui::CShape shapeNeurons;
      crl::gui::CShape shapePath;
      crl::gui::CShape shapeRing;
      crl::gui::CShape shapeCommRadius;
      crl::gui::CShape shapePathNodes;
      crl::gui::CShape shapeDepot;
      crl::gui::CShape shapeTourRepresentedByRing;

      std::string method;

      IntVector permutation;
      TargetPtrVector targets;
      Coords_tsVector finalSolution;

      CRing *ring;
   };


   // CoordsVector api_func(CGSOA &gsoa,TargetPtrVector targets);


} // end name gsoa

#endif


