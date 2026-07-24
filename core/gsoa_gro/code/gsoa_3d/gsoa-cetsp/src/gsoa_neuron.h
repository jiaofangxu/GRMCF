//神经元

#ifndef __GSOA_NEURON_H__
#define __GSOA_NEURON_H__

#include "coords_3d.h"
#include <unordered_set>
namespace gsoa
{
   struct SNeuron
   {
      SNeuron* prev; //前一个神经元
      SNeuron* next; //后一个神经元
      Coords coords; //当前神经元的坐标
      Coords alternateGoal; //代替的目标访问点
      int targetOnTourStep;//当前神经元关联目标点的时间步
      // int targetOnTour; //当前神经元关联的目标点标签
      std::unordered_set<int>  targetOnTour; // 当前神经元关联的目标点标签（多个）

      char s;//当前神经元在线段上的位置标记
      Coords x; //目标中心 到 线段的最近点(投影点)
      double dist2; //squared distance  改成时间代价

      SNeuron()
      {
         clear();
      }

      SNeuron(const Coords &pt) : coords(pt)
      {
         clear();
      } 

      ~SNeuron() {}

      void setPoint(const Coords &pt) 
      {
         coords = pt;
         clear();
      }

      // void clear(void)
      // {
      //    targetOnTour = -1;
      //    targetOnTourStep = -1;
      // }
      void clear(void)
      {
         targetOnTour.clear();
         targetOnTourStep = -1;
      }

      void adapt(const Coords &pt, const double beta)
      {
         coords.x += beta * (pt.x - coords.x);
         coords.y += beta * (pt.y - coords.y);
         coords.z += beta * (pt.z - coords.z);
      }

      inline bool isInhibited(int step) const { return targetOnTourStep >= step; }
      // 在末尾增加一个目标点标签
      void addTargetOnTour(int target) {
         targetOnTour.insert(target);
      }

      // 删除所有与指定目标点标签相同的元素
      void removeTargetOnTour(int target) {
         targetOnTour.erase(target); // 删除元素 target;
      }
   };
}// end namespace gsoa

#endif

/* end of gsoa_neuron.h */
