/*
 * File name: gsoa_ring.cc
 * Date:      2016/12/09 21:47
 * Author:    Jan Faigl
 */

#include <limits>

#include <boost/foreach.hpp>

#include <crl/logging.h>
#include <unordered_set> // 包含 unordered_set 的头文件
#include "simple_intersection.h"

#include "gsoa_neuron.h"
#include "gsoa_learning.h"
#include "gsoa_ring.h"

#define foreach BOOST_FOREACH

using namespace gsoa;

typedef geom::CIntersection<Coords> Intersection;
static const double NEURON_COORDS_IDENTITY2 = 1e-5*1e-5;

using crl::logger;

/// - constructor --------------------------------------------------------------
CRing::CRing(TargetPtrVector *targets, const double radiusDecrease, bool in_neigh_waypoint)
   : RADIUS_DECREASE(radiusDecrease), IN_NEIGH_WAYPOINT(in_neigh_waypoint), targets(targets)
{
   start = 0;
   m = 0;
}

/// - destructor ---------------------------------------------------------------
CRing::~CRing()
{
   deallocate_neurons();
}

/// - public method ------------------------------------------------------------
void CRing::initialize_neurons(const Coords &pt)
{
   deallocate_neurons();
   start = new SNeuron(pt);
   start->prev = start;
   start->next = start;
   m = 1;
}
void CRing::update_targets(TargetPtrVector *n_targets)
{
	targets = n_targets;
}
/// - public method ------------------------------------------------------------
void CRing::deallocate_neurons(void)
{
   SNeuron *cur = start;
   for (int i = 0; i < m; ++i) {
      SNeuron *n = cur->next;
      delete cur;
      cur = n;
   }
   m = 0;
   start = 0;
}

/// - public method ------------------------------------------------------------
SNeuron* CRing::insertNeuron(SNeuron* cur, SNeuron* neuron)
{  // insert neuron after the cur
   neuron->prev = cur;
   neuron->next = cur->next;
   cur->next->prev = neuron;
   cur->next = neuron;
   m += 1;
   return neuron;
}

/// - public method ------------------------------------------------------------
void CRing::removeNeuron(SNeuron *neuron)
{
   neuron->prev->next = neuron->next;
   neuron->next->prev = neuron->prev;
   if (start == neuron) {
      start = neuron->next;
      if (m == 1) {
	 start = 0;
      }
   }
   delete neuron;
   m -= 1;
}




/// - public method ------------------------------------------------------------
SWinnerSelection* CRing::selectWinner(int step, STarget* target, double &errorToGoal)
{
   double error = std::numeric_limits<double>::max();

   winner.hasWinner = false;
   winner.target = target;
   winner.preWinner = winner.winner= 0;
   //质心
   winner.alternateGoal = target->coords;
   const double RADIUS = target->radius;
   const double RADIUS_SMALLER = RADIUS - RADIUS_DECREASE;
   const double RADIUS2 = RADIUS * RADIUS;
   double bestLength = std::numeric_limits<double>::max();

   SNeuron *cur = start;
   for (int i = 0; i < m; ++i) {
     //计算 目标的质心 到 当前线段的距离
   	const double d2 = Intersection::point_segment_squared_distance(target->coords, cur->coords, cur->next->coords, cur->s, cur->x);

	if (d2 >= RADIUS2) { //如果在邻域外面
	    const double td = sqrt(d2);
	    const Coords alternateGoal(
		  target->coords.x + (cur->x.x - target->coords.x) * (RADIUS_SMALLER / td),
		  target->coords.y + (cur->x.y - target->coords.y) * (RADIUS_SMALLER / td),0
		  );
	    const double dist2 = alternateGoal.squared_distance(cur->x);
	    if (dist2 < bestLength) { //select the shortest one
	       if (cur->s == 's') { //new point inside the segment
		  winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
	       } else {
		  winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
	       }
	       bestLength = error = dist2;
	       winner.hasWinner = true;
	       winner.alternateGoal = alternateGoal;
	    } //end new winner
	 } else {
	    if (IN_NEIGH_WAYPOINT and m > 2) {
	       SNeuron *prev = cur->isInhibited(step) ? cur : 0;
	       SNeuron *next = cur->next->isInhibited(step) ? cur->next : 0;
	       if (prev and next) {
		  Coords x;
		  char s;
		  const double d2 = Intersection::point_segment_squared_distance(target->coords, prev->alternateGoal, next->alternateGoal, s, x);
		  const double td = sqrt(d2);
		  Coords ag(
			target->coords.x + (x.x - target->coords.x) * (RADIUS_SMALLER / td),
			target->coords.y + (x.y - target->coords.y) * (RADIUS_SMALLER / td),0
			);
		  winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
		  winner.hasWinner = true;
		  winner.alternateGoal = ag; 
		  bestLength = error = ag.squared_distance(x);
	       } else {
		  if (cur->s == 's') { //new point inside the segment
		     winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
		  } else {
		     winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
		  }
		  winner.hasWinner = true;
		  winner.alternateGoal = cur->x;
		  bestLength = error = 0.0;
	       }
	    } else { 
	       // the x point is within the communication radius
	       // mark it as the alternateGoal goal and also as the winner
	       if (cur->s == 's') { //new point inside the segment
		  winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
	       } else {
		  winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
	       }
	       winner.hasWinner = true;
	       winner.alternateGoal = cur->x;
	       bestLength = error = 0.0;
	    }
	    break; // stop the search 
	 }
	 cur = cur->next;
      } //end all neurons

   if (winner.hasWinner) {
   	      winner.targetOnTour = target->label;
//   	winner.addTargetOnTour(target->label);
      errorToGoal = sqrt(error);
   }
   return &winner;
}



/// - public method ------------------------------------------------------------
SWinnerSelection* CRing::selectWinner_3d(int step, STarget* target, double &errorToGoal)
{
   double error = std::numeric_limits<double>::max();

   winner.hasWinner = false;
   winner.target = target;
   winner.preWinner = winner.winner= 0;
   winner.alternateGoal = target->coords;

//   const double RADIUS = target->radius;
//   const double RADIUS_SMALLER = RADIUS - RADIUS_DECREASE;
//   const double RADIUS2 = RADIUS * RADIUS;

   double bestLength = std::numeric_limits<double>::max();
   double bestRingAddition = 0.0;

      SNeuron *cur = start;
   for (int i = 0; i < m; ++i) {
        //
	 const double d2 = Intersection::point_segment_squared_distance(target->coords, cur->coords, cur->next->coords, cur->s, cur->x);
	 if (!target->isPointInside(cur->x)) { //最近点是否在多面体内部
	 	//最近点不在多面体内部
	    const double td = sqrt(d2);
	 	//生成一个靠近多面体表面的点
        const Coords alternateGoal = target->findIntersection(cur->x);

//	    const Coords alternateGoal(
//		  target->coords.x + (cur->x.x - target->coords.x) * (RADIUS_SMALLER / td),
//		  target->coords.y + (cur->x.y - target->coords.y) * (RADIUS_SMALLER / td),
//		  target->coords.z + (cur->x.z - target->coords.z) * (RADIUS_SMALLER / td)
//		  );
        //欧式距离
	    const double dist2 = alternateGoal.squared_distance(cur->x);
        //时间代价
//	    const double dist2 = alternateGoal.time_cost(cur->x);
	    if (dist2 < bestLength) { //select the shortest one
	       if (cur->s == 's') { //new point inside the segment 线段中的新点
		  winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
	       } else {
		  winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
	       }
	       bestLength = error = dist2;
	       winner.hasWinner = true;
	       winner.alternateGoal = alternateGoal;
	    } //end new winner
	 } else {
	    if (IN_NEIGH_WAYPOINT and m > 2) {
	       SNeuron *prev = cur->isInhibited(step) ? cur : 0;
	       SNeuron *next = cur->next->isInhibited(step) ? cur->next : 0;
	       if (prev and next) {
			  Coords x;
			  char s;
			  const double d2 = Intersection::point_segment_squared_distance(target->coords, prev->alternateGoal, next->alternateGoal, s, x);
			  const double td = sqrt(d2);

		      const Coords ag = target->findIntersection(x);

	       	  const double dist2 = ag.squared_distance(x);
	       	//时间代价
	       	//	    const double dist2 = ag.time_cost(x);


	       	  if (dist2 < bestLength) { //select the shortest one
	       	  	if (s == 's') { //new point inside the segment
	       			winner.newPoint = x; winner.preWinner = cur; winner.winner = 0; //use new point
	       		} else {
	       			winner.preWinner = 0; winner.winner = s == 'a' ?  prev : next; //use winner
	       		}
	       	  	bestLength = error = dist2;
	       		winner.hasWinner = true;
	       		winner.alternateGoal = ag;
	       	  } //end new winner


	       } else {
		  		if (cur->s == 's') { //new point inside the segment
		     		winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
		  		} else {
		     		winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
		  		}
		  		winner.hasWinner = true;
		  		winner.alternateGoal = cur->x;
		  		bestLength = error = 0.0;
	       	    break; // stop the search
	       	}
	    } else {
	       // the x point is within the communication radius
	       // mark it as the alternateGoal goal and also as the winner
	       if (cur->s == 's') { //new point inside the segment
		  	winner.newPoint = cur->x; winner.preWinner = cur; winner.winner = 0; //use new point
	       } else {
		  	winner.preWinner = 0; winner.winner = cur->s == 'a' ?  cur : cur->next; //use winner
	       }
	       winner.hasWinner = true;
	       winner.alternateGoal = cur->x;
	       bestLength = error = 0.0;
	    	break; // stop the search
	    }

	 }
	 cur = cur->next;
   } //end all neurons

   if (winner.hasWinner) {
      winner.targetOnTour = target->label;
//   	  winner.addTargetOnTour(target->label);
      errorToGoal = error;
   }
   return &winner;
}









/// - public method ------------------------------------------------------------
void CRing::adapt(const int step)
{
   ASSERT_ARGUMENT(winner.hasWinner, "Cannot adapt without winner");
   //Update the network 
   if (winner.winner) { // it is a regular winner
      SNeuron *wNeuron = winner.winner;
      if (winner.winner == start) { // it is the first neuron
	  wNeuron = new SNeuron(winner.winner->coords);
	 insertNeuron(start, wNeuron);
      } else if (winner.winner->isInhibited(step)) { //also create a new one
        //之前选中过
	 	if (winner.winner->targetOnTour.find( winner.targetOnTour)!= winner.winner->targetOnTour.end()) {
	    	// avoid replication of the winner as it has been already selected in this epoch

	 	} else {
	    	wNeuron = new SNeuron(winner.winner->coords);
	    	insertNeuron(winner.winner, wNeuron);
	 	}
      }
      winner.winner = wNeuron;
   } else {//处理胜利节点为新点的情况
      ASSERT_ARGUMENT(winner.preWinner, "newPoint needs preWinner");
      SNeuron *wNeuron = new SNeuron(winner.newPoint);
      insertNeuron(winner.preWinner, wNeuron);
      winner.winner = wNeuron;
   }
   // 更新胜利节点的目标信息
   winner.winner->alternateGoal = winner.alternateGoal;
   winner.winner->addTargetOnTour(winner.targetOnTour);
   winner.winner->targetOnTourStep = step;

   { // uninhibit previous winner associated to the target in the current learning epoch
     //清除当前目标的前一个获胜节点(局部代码块，作用域)
      STarget *target = winner.target;
      //选择的胜利节点步骤 和 本步 相等
      if (target->stepWinnerSelected == step) {
	 	//target has been already associated in this epoch
	 	if (target->selectedWinner and target->selectedWinner != winner.winner) { //但是选择的胜利节点 不是本胜利节点(之前的)
	    	//new winner has been selected, clear te previous association 新的获胜者已经选出，清除之前的关联
	    	target->selectedWinner->targetOnTourStep = -1; //uninhibit the tour 取消抑制
	    	//maybe we can consider to delete the neuron right in the current epoch 也许我们可以考虑在当前时代删除神经元
	 	}
      }
      //更新目标的当前胜利节点为新的胜利节点
      target->stepWinnerSelected = step;
      target->selectedWinner = winner.winner;
   }
	//开始适应神经元网络
   SNeuron *neuron = winner.winner; //当前胜利节点
   SNeuron *neuronBackward = neuron->prev; //前驱
   SNeuron *neuronForward = neuron->next; //后继
   //确定适应的范围。在胜利节点附近的节点
   double dd = 1.0;
   //衰减系数
   const int d = m / schema->NEIGHBORHOOD_FACTOR;
   const int t = d < (m / 2) ? d : m / 2;
   const int TO = t < schema->explen ? t : schema->explen; // up to MIN_GAIN
   //适应胜利节点和邻域神经元
   const Coords& target = neuron->alternateGoal;
   const SNeuron *ringStart = start;
   if (neuron->coords.squared_distance(target) > 0.0) {
     //胜利节点 neuron 按学习率 mi 调整其坐标，使其更接近目标点
      const double mi = schema->mi;
      neuron->adapt(target, mi);
      //遍历邻域内的神经元（neuronBackward 和 neuronForward），按衰减系数 b 调整其坐标。调整的范围由 TO 决定。
      for (int i = 0; i < TO; i++) {
	 	const double b = schema->exps[i];
	 	if (neuronBackward != neuron) {
	    	neuronBackward->adapt(target, b);
	    	neuronBackward = neuronBackward->prev;
	 	}
	 	if (neuronForward != neuron) {
	    	neuronForward->adapt(target, b);
	    	neuronForward = neuronForward->next;
	 	}
	 	dd += 1.0;
      } //end neighborhood
   } //end conditional adapt
//   return winner.winner
}





/// - public method ------------------------------------------------------------
SNeuron* CRing::adapt_3d(const int step)
{
   ASSERT_ARGUMENT(winner.hasWinner, "Cannot adapt without winner");
   //Update the network
   if (winner.winner) { // it is a regular winner
      SNeuron *wNeuron = winner.winner;
      if (winner.winner == start) { // it is the first neuron
	 		wNeuron = new SNeuron(winner.winner->coords);
	 		insertNeuron(start, wNeuron);
      } else if (winner.winner->isInhibited(step)) { //also create a new one
      	//之前选中过
      	if (winner.winner->targetOnTour.find( winner.targetOnTour)!= winner.winner->targetOnTour.end()) {
	    	// avoid replication of the winner as it has been already selected in this epoch
//      		std::cout << "winner.winner->targetOnTour 含有 winner.targetOnTour" << std::endl;  // 换行
      		winner.winner->removeTargetOnTour(winner.targetOnTour);
      		if(winner.winner->targetOnTour.empty()){
      			//把邻域之前对应的节点解除对应(还有可能有其他的关联点，所以不要直接解除，要判断一下)
      			winner.winner->targetOnTourStep =-1;
      		}
      		wNeuron = new SNeuron(winner.winner->coords);
      		insertNeuron(winner.winner, wNeuron);
	 	} else {
	    	wNeuron = new SNeuron(winner.winner->coords);
	    	insertNeuron(winner.winner, wNeuron);
	 	}
      }
      winner.winner = wNeuron;
   } else {
      ASSERT_ARGUMENT(winner.preWinner, "newPoint needs preWinner");
      SNeuron *wNeuron = new SNeuron(winner.newPoint);
      insertNeuron(winner.preWinner, wNeuron);
      winner.winner = wNeuron;
   }
   winner.winner->alternateGoal = winner.alternateGoal;
   winner.winner->addTargetOnTour(winner.targetOnTour);
   winner.winner->targetOnTourStep = step;

   { // uninhibit previous winner associated to the target in the current learning epoch
      STarget *target = winner.target;
      if (target->stepWinnerSelected == step) {
	 	//target has been already associated in this epoch
	 	if (target->selectedWinner and target->selectedWinner != winner.winner) {
	    	//new winner has been selected, clear te previous association
//	    	target->selectedWinner->targetOnTourStep = -1; //uninhibit the tour
	    	//maybe we can consider to delete the neuron right in the current epoch
	 		target->selectedWinner->removeTargetOnTour(target->label);
	 		if(target->selectedWinner->targetOnTour.empty()){
	 			//把邻域之前对应的节点解除对应(还有可能有其他的关联点，所以不要直接接触，要判断一下)
	 			target->selectedWinner->targetOnTourStep =-1;
	 		}
	 	}
      }
      target->stepWinnerSelected = step;
      target->selectedWinner = winner.winner;
   }

   SNeuron *neuron = winner.winner;
   SNeuron *neuronBackward = neuron->prev;
   SNeuron *neuronForward = neuron->next;
   double dd = 1.0;
   const int d = m / schema->NEIGHBORHOOD_FACTOR;
   const int t = d < (m / 2) ? d : m / 2;
   const int TO = t < schema->explen ? t : schema->explen; // up to MIN_GAIN
   const Coords& target = neuron->alternateGoal;
   const SNeuron *ringStart = start;
   if (neuron->coords.squared_distance(target) > 0.0) {
      const double mi = schema->mi;
      neuron->adapt(target, mi);
      for (int i = 0; i < TO; i++) {
	 const double b = schema->exps[i];
	 if (neuronBackward != neuron) {
	    neuronBackward->adapt(target, b);
	    neuronBackward = neuronBackward->prev;
	 }
	 if (neuronForward != neuron) {
	    neuronForward->adapt(target, b);
	    neuronForward = neuronForward->next;
	 }
	 dd += 1.0;
      } //end neighborhood
   } //end conditional adapt
  return winner.winner;
}





/// - public method ------------------------------------------------------------
void CRing::regenerate(int step)
{
   SNeuron *del[m];
   int c = 0;
   SNeuron *cur = start->next;
   for (int i = 0; i < m; ++i) {
      if (not cur->isInhibited(step)) {
	 		del[c++] = cur;
      }
      cur = cur->next;
   }
   for (int i = 0; i < c; ++i) {
      removeNeuron(del[i]);
   }
}

/// - public method ------------------------------------------------------------
//CoordsVector& CRing::get_ring_path(int step, CoordsVector &path) const
//{
//   path.clear();
//   SNeuron *cur = start;
//   IntVector is_v_targets;
//   for (int i = 0; i < m; ++i) {
////      if (cur->isInhibited(step) and cur->targetOnTourStep >= 0 and cur == (*targets)[cur->targetOnTour]->selectedWinner) {
//      if (cur->isInhibited(step) and cur->targetOnTourStep >= 0) {
//	 path.push_back(cur->alternateGoal);
//      }
//      cur = cur->next;
//   }
//
//   return path;
//}

Coords_tsVector& CRing::get_ring_path(int step, Coords_tsVector &path,TargetPtrVector ts)
{
	path.clear();
	if (!start) {
		return path; // 如果链表为空，直接返回空的 path
	}

	// 统计所有 targetOnTour 的值
//	std::unordered_set<int> targetSet; // 用于存储所有唯一的 targetOnTour 值
	SNeuron *cur = start;

	for (int i = 0; i < m; ++i) {
		if (cur->isInhibited(step) && cur->targetOnTourStep >= 0) {
			std::vector<int> toCheck ;

			for (size_t j = 0; j < ts.size(); ++j) {
				if (ts[j]->selectedWinner->alternateGoal == cur->alternateGoal){
					toCheck.push_back(j);
				}
			}

//            Coords_ts ct(cur->alternateGoal.x,cur->alternateGoal.y,cur->alternateGoal.z,cur->targetOnTour);
			if (!toCheck.empty()){
				Coords_ts ct(cur->alternateGoal.x,cur->alternateGoal.y,cur->alternateGoal.z,toCheck);
				path.push_back(ct); // 将 alternateGoal 添加到 path 中
			}


//			std::cout << "(" << cur->alternateGoal.x << ", "
//						  << cur->alternateGoal.y << ", "
//						  << cur->alternateGoal.z << "):";
//			// 打印 cur->targetOnTour 中的所有目标
//			std::cout << "[";
//			bool first = true;
//			for (int target : cur->targetOnTour) {
//				if (target >= 0) { // 确保 target 是有效的
//					if (!first) std::cout << ", ";
//					std::cout << target;
//					first = false;
//				}
//			}
//			std::cout << "]" << std::endl;  // 换行


			// 将 cur->targetOnTour 中的所有值添加到 targetSet 中
//			for (int target : cur->targetOnTour) {
//				if (target >= 0) { // 确保 target 是有效的
//					targetSet.insert(target);
//				}
//			}
		}
		cur = cur->next;
		if (!cur) break; // 如果链表不是严格的环形链表，终止循环
	}

//	// 检查是否覆盖了所有目标点
//	int targetSize = targets->size(); // 假设 targets 是一个指向 vector 的指针
//	bool allTargetsCovered = true;
//
//	for (int i = 0; i < targetSize; ++i) {
//		if (targetSet.find(i) == targetSet.end()) {
//			allTargetsCovered = false; // 如果某个目标点未被覆盖，设置为 false
//			break;
//		}
//	}
//
//	// 输出统计结果（可以根据需要调整）
//	if (allTargetsCovered) {
//		std::cout << "CRing::get_ring_path: All targets are covered by targetOnTour." << std::endl;
//	} else {
//		std::cout << "CRing::get_ring_path: Not all targets are covered by targetOnTour." << std::endl;
//	}

	return path;
}



/// - public method ------------------------------------------------------------
IntVector& CRing::get_ring_route(int step, IntVector &route) const
{
   route.clear();
   SNeuron *cur = start;
   for (int i = 0; i < m; ++i) {
//      if (cur->isInhibited(step) and cur->targetOnTourStep >= 0 and cur == (*targets)[cur->targetOnTour]->selectedWinner) {
      if (cur->isInhibited(step) and cur->targetOnTourStep >= 0 ) {
      	for (int target : cur->targetOnTour) {
      		route.push_back(target); // 逐个插入
      	}
      }
      cur = cur->next;
   }
   return route;
}

/* end of gsoa_ring.cc */
