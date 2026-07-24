/*
 * File name: gsoa_ring.h
 * Date:      2016/12/09 08:18
 * Author:    Jan Faigl
 */

#ifndef __GSOA_RING_H__
#define __GSOA_RING_H__

#include <vector>

#include "coords_3d.h"
#include "target_3d.h"
#include "gsoa_neuron.h"

namespace gsoa {

   typedef std::vector<int> IntVector;

   /// ----------------------------------------------------------------------------
   /// @brief 
   /// ----------------------------------------------------------------------------
   struct SWinnerSelection {
      bool hasWinner;
      STarget *target;
      SNeuron *winner; // if 0, use the newPoint
      SNeuron *preWinner;
      Coords newPoint; 
      Coords alternateGoal;
      int targetOnTour;
      // std::unordered_set<int> targetOnTour; // 当前神经元关联的目标点标签（多个）
  //    // 在末尾增加一个目标点标签
      // void addTargetOnTour(int target) {
         // targetOnTour.insert(target);
      // }

//      // 删除所有与指定目标点标签相同的元素
      // void removeTargetOnTour(int target) {
         // targetOnTour.erase(target);
      // }



   };

   /// ----------------------------------------------------------------------------
   /// @brief 
   /// ----------------------------------------------------------------------------
   class CRing {
      const double RADIUS_DECREASE;
      const bool IN_NEIGH_WAYPOINT;
      public:
      CRing(TargetPtrVector *targets, const double radiusDecrease = 0.1, const bool in_neigh_waypoint = false);
      ~CRing();

      int size(void) const { return m; }
      const SNeuron* begin(void) const { return start; }

      void initialize_neurons(const Coords &pt);
      void deallocate_neurons(void);
      void update_targets(TargetPtrVector *n_targets);

      SNeuron* insertNeuron(SNeuron* cur, SNeuron* neuron);
      void removeNeuron(SNeuron *neuron);

      SWinnerSelection* selectWinner(int step, STarget* target, double &errorToGoal);
      SWinnerSelection* selectWinner_3d(int step, STarget* target, double &errorToGoal);
      void adapt(const int step);
      SNeuron* adapt_3d(const int step);
      void regenerate(int step);


      Coords_tsVector& get_ring_path(int step, Coords_tsVector &path,TargetPtrVector ts);
      IntVector& get_ring_route(int step, IntVector &route) const;
      private:
      TargetPtrVector *targets;
      SNeuron *start;
      int m;
      SWinnerSelection winner;
   };

} // end namespace gsoa

#endif

/* end of gsoa_ring.h */
