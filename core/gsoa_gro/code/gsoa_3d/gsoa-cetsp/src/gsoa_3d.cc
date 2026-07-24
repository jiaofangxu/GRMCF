/*
 * File name: gsoa.cc
 * Date:      2016/12/07 08:32
 * Author:    Jan Faigl
 */

#include <limits>

#include <boost/foreach.hpp>

#include <crl/random.h>
#include <crl/logging.h>
#include <crl/assert.h>
#include <crl/file_utils.h>

#include <crl/gui/shape.h>
#include <crl/gui/shapes.h>
#include <random>

#include "gsoa_3d.h"
#include "coords_3d.h"

#include "gsoa_learning.h"

#include "route_path_utils.h"

#include "canvasview_coords.h"
#include "canvasview_gsoa.h"

#define foreach BOOST_FOREACH

using namespace gsoa;

typedef std::vector<int> IntVector;

using namespace crl;
using namespace crl::gui;





//void printTargets(const std::vector<gsoa::STarget*>& targets) {
//   for (const auto* target : targets) {
//      if (!target) continue; // 防止空指针
//
//      // 打印目标的基本信息
//      std::cout << "Target ID: " << target->label << "\n";
//      std::cout << "Centroid: (" << target->coords.x << ", "
//                << target->coords.y << ", " << target->coords.z << ")\n";
//
//      // 打印每个面的参数
//      std::cout << "Faces:\n";
//      for (size_t i = 0; i < target->planarVector.size(); ++i) {
//         const auto& face = target->planarVector[i];
//         std::cout << "  Face " << i + 1 << ": A=" << face.a
//                   << " B=" << face.b
//                   << " C=" << face.c
//                   << " D=" << face.d << "\n";
//      }
//
//      std::cout << "-----------------------------------\n";
//   }
//}


/// ----------------------------------------------------------------------------
static void createPermutation(int number, IntVector &permutation) 
{
   permutation.clear();
   for (int i = 0; i < number; i++) {
      permutation.push_back(i);
   }
}

/// ----------------------------------------------------------------------------
static void permute(IntVector &permutation) 
{
   int k, tmp;
   // crl::CRandom::randomize();
   for (int i = permutation.size(); i > 0; --i) {
      k = crl::CRandom::random() % i;
      tmp = permutation[i - 1];
      permutation[i - 1] = permutation[k];
      permutation[k] = tmp;
   }
}

/// - static method ------------------------------------------------------------
crl::CConfig& CGSOA::getConfig(crl::CConfig &config) 
{
   // basic properties not included in the crl algorithm
   Base::getConfig(config);
   config.add<bool>("save-info", "disable/enable save info", true);
   config.add<bool>("save-settings", "disable/enable save settings", true);
   config.add<int>("port", "set api port", 8000);

   config.add<std::string>("result-path", "file name for the final found path (ring) as sequence of points",
         "path");
   config.add<std::string>("pic-dir", "relative directory in result directory to store pictures from each iteration");
   config.add<std::string>("pic-ext",
         "extension of pic, eps, png, pdf, svg (supported by particular gui renderer rendered",
         "png");
   config.add<bool>("save-pic", "enable/disable saving pictures (after each refine)");
   // 
   // GSOA - learning paramters 
   Schema::getConfig(config);
   config.add<bool>("2opt-post", "enable 2opt postprocessing of the found path", false);
   config.add<double>("radius-decrease", "Distance to decreased the radius to ensure the waypoint is in the neighbourhood", 0.1);
   config.add<bool>("in-neigh-waypoint", "enable/disable determining in neighborhood waypoint", false);
   //
   // Problem specification
   config.add<std::string>("problem", "Problem file");
   config.add<int>("depot-idx", "If >= 0, the particular goal is considered as the depot with the radius 0", -1);
   config.add<bool>("variable-radius", "If enabled, the input file is considered as x y radius ", false);
   config.add<double>("communication-radius", "Radius within other sensors can be read, disabled if <= 0", 0);
   config.add<std::string>("method", "Specify method in the result log", "gsoa");
   //
   // Gui properties
   config.add<std::string>("draw-shape-targets", "Shape of the target", Shape::CITY());
   config.add<std::string>("draw-shape-neurons", "Shape of the neurons", Shape::NEURON());

   config.add<std::string>("draw-shape-path", "Shape of the path", Shape::RED_LINE());
   config.add<std::string>("draw-shape-ring", "Shape of the ring", Shape::GREEN_BOLD_LINE());
   config.add<std::string>("draw-shape-communication-radius", "Shape of the communication radius for highlight coverage", Shape::POLYGON_FILL());
   config.add<std::string>("draw-shape-path-nodes", "Shape of the path nodes", Shape::MAP_VERTEX());
   config.add<std::string>("draw-shape-depot", "Shape of the depot node", Shape::DEPOT());
   config.add<std::string>("draw-shape-tour-represented-by-ring", "Shape of the tour represented by the ring", Shape::RED_LINE());

   config.add<bool>("draw-ring-iter", "enable/disable drawing ring at each iteration", false);
   config.add<bool>("draw-ring", "Enable/Disable drawing ring in the final shoot", true);
   config.add<bool>("draw-ring-nodes", "Enable/disable drawing ring nodes", true);
   config.add<bool>("draw-tour-represented-by-ring", "Enable/disable drawing tour represented by the ring", false);
   config.add<bool>("draw-path", "Enable/Disable drawing ring in the final shoot", true);
   config.add<bool>("draw-path-nodes", "enable/disable drawing path vertices(nodes)", true);
   config.add<bool>("draw-communication-radius", "enable/disable drawing radius of the selected targets", false);
   config.add<bool>("draw-depot", "enable/disable drawing depot", false);
   config.add<bool>("disable-area-radius", "enable/disable computing area from the radius", false);
   config.add<double>("add-x", "add the value to the coords for computing the canvas area", 0.0);
   config.add<double>("add-y", "add the value to the coords for computing the canvas area", 0.0);
   return config;
}

/// - constructor --------------------------------------------------------------
CGSOA::CGSOA(crl::CConfig &config) : Base(config, "TRIAL"),

   DEPOT_IDX(config.get<int>("depot-idx")),//仓库的索引位置
   VARIABLE_RADIUS(config.get<bool>("variable-radius")),//是否使用可变半径
   SAVE_RESULTS(config.get<bool>("save-results")),//是否保存结果
   SAVE_SETTINGS(config.get<bool>("save-settings")),//是否保存设置
   SAVE_INFO(config.get<bool>("save-info")),//是否保存信息
   DRAW_RING_ITER(config.get<bool>("draw-ring-iter")),//绘画的配置
   DRAW_RING_ENABLE(config.get<bool>("draw-ring")),
   DRAW_RING_NODES(config.get<bool>("draw-ring-nodes")),
   DRAW_TOUR_REPRESENTED_BY_RING(config.get<bool>("draw-tour-represented-by-ring")),
   SAVE_PIC(config.get<bool>("save-pic")),//是否保存图片
   IN_NEIGH_WAYPOINT(config.get<bool>("in-neigh-waypoint"))//是否启用邻近路径点模式
{
  //读取配置文件的配置，填充到自己的成员变量中
   shapeTargets.setShape(config.get<std::string>("draw-shape-targets"));
   shapeNeurons.setShape(config.get<std::string>("draw-shape-neurons"));
   shapePath.setShape(config.get<std::string>("draw-shape-path"));
   shapeRing.setShape(config.get<std::string>("draw-shape-ring"));
   shapeCommRadius.setShape(config.get<std::string>("draw-shape-communication-radius"));
   shapePathNodes.setShape(config.get<std::string>("draw-shape-path-nodes"));
   shapeDepot.setShape(config.get<std::string>("draw-shape-depot"));
   shapeTourRepresentedByRing.setShape(config.get<std::string>("draw-shape-tour-represented-by-ring"));

   method = config.get<std::string>("method");

   if (!schema) {
      schema = new Schema(config);
   }

   const std::string fname = config.get<std::string>("problem");
   std::ifstream in(fname);

//   //读取多面体数据
//   std::string line;
//   while (true) {
//      Coords cd;
//      PlanarVector planarVec;
//
//      // 读取质心
//      if (!std::getline(in, line) || line.empty()) break;
//      std::istringstream centroidStream(line);
//      if (!(centroidStream >> cd)) break;
//
//      // 读取面的参数
//      planarVec.clear();
//      while (std::getline(in, line)) {
//         if (line.empty()) break; // 遇到空行，结束当前多面体
//         std::istringstream planarStream(line);
//         Planar pt;
//         if (!(planarStream >> pt)) break;
//         planarVec.push_back(pt);
//      }
//      // 生成目标
//      if (!planarVec.empty()) {
//         // 如果有面的参数，生成带面的目标
//         targets.push_back(new gsoa::STarget(targets.size(), planarVec, cd));
//      } else {
//         // 如果没有面，生成仅包含点的目标
//         targets.push_back(new gsoa::STarget(targets.size(), PlanarVector(), cd));
//      }
//   }
//
//   if (name.size() == 0) {
//      std::string n = getBasename(fname);
//      size_t i = n.rfind(".txt");
//      if (i != std::string::npos) {
//         name = n.erase(i, 4);
//      }
//   }

//   printTargets(targets);
   //半径减小这个的用法 用不到，暂时不删除
   ring = new CRing(&targets, config.get<double>("radius-decrease"), IN_NEIGH_WAYPOINT);
}


/// - destructor ---------------------------------------------------------------
CGSOA::~CGSOA()
{
   delete ring;
   foreach(STarget *target, targets) {
      delete target;
   }
}

/// - public method ------------------------------------------------------------
std::string CGSOA::getVersion(void) 
{
   return "GSOA 3d-TSPN 1.0";
}

CoordsVector CGSOA:: api_run(TargetPtrVector new_targets){
   foreach(STarget *target, targets) {
      delete target;
   }
   targets = new_targets;
   ring->update_targets(&new_targets);
   DEBUG("Single iteration enabled " << iter);
   tLoad.reset().start();
   load();
   tLoad.stop();
   tInit.reset().start();
   initialize();
   tInit.stop();
   after_init();
   tSolve.reset().start();
   iterate(iter);
   tSolve.stop();
   tRelease.reset().start();
   release();
   tRelease.stop();
   CoordsVector r;
   for (Coords_ts f : finalSolution){
      r.push_back(Coords(f.x,f.y,f.z));
   }

   return r;
}

Coords_ts CGSOA::compute_numerical_gradient(std::function<double(const Coords_ts&)> func,
                                            const Coords_ts& B, double h,
                                            const std::vector<int>& ts_l) {
    double dx = (func(Coords_ts(B.x + h, B.y, B.z, ts_l)) - func(Coords_ts(B.x - h, B.y, B.z, ts_l))) / (2 * h);
    double dy = (func(Coords_ts(B.x, B.y + h, B.z, ts_l)) - func(Coords_ts(B.x, B.y - h, B.z, ts_l))) / (2 * h);
    double dz = (func(Coords_ts(B.x, B.y, B.z + h, ts_l)) - func(Coords_ts(B.x, B.y, B.z - h, ts_l))) / (2 * h);
    return Coords_ts(dx, dy, dz, ts_l); // 梯度方向保留原标签
}

void CGSOA::normalize(Coords_ts& dir) {
    double norm = std::sqrt(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    if (norm > 1e-12) {
        dir.x /= norm;
        dir.y /= norm;
        dir.z /= norm;
    }
}

bool CGSOA::inside_all_regions(const Coords_ts& pt,
                               const TargetPtrVector& targets,
                               const std::vector<int>& ts_l) {
    for (int val : ts_l) {
        Coords c(pt.x, pt.y, pt.z); // 转为 Coords 判断
        if (!targets[val]->isPointInside(c)) return false;
    }
    return true;
}

void CGSOA::optimize_B(const Coords_ts& A, Coords_ts& B, const Coords_ts& C,
                       TargetPtrVector targets) {
    int max_iter = 50;
    double h = 1e-4;
    int stall_limit = 8;
    int stall_count = 0;

    std::vector<int> ts_l = B.ts;

    auto total_cost = [&](const Coords_ts& b) {
        return cost_ts(A, b) + cost_ts(b, C);
    };

    for (int iter = 0; iter < max_iter; ++iter) {
        double base_cost = total_cost(B);

        Coords_ts grad_AB = compute_numerical_gradient(
            [&](const Coords_ts& b) { return cost_ts(A, b); }, B, h, ts_l);
        Coords_ts grad_BC = compute_numerical_gradient(
            [&](const Coords_ts& b) { return cost_ts(b, C); }, B, h, ts_l);

        Coords_ts dir_AB(-grad_AB.x, -grad_AB.y, -grad_AB.z, ts_l);
        Coords_ts dir_BC(-grad_BC.x, -grad_BC.y, -grad_BC.z, ts_l);

        normalize(dir_AB);
        normalize(dir_BC);

        double dot_product = dir_AB.x * dir_BC.x + dir_AB.y * dir_BC.y + dir_AB.z * dir_BC.z;
        if (dot_product < -0.99) return; // 方向完全相反，跳出

        // 构造两个半球的重叠方向集合
        std::vector<Coords_ts> directions;
        std::random_device rd;
        std::mt19937 rng(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        int num_directions = 30;
        for (int i = 0; i < num_directions; ++i) {
            double x = dist(rng);
            double y = dist(rng);
            double z = dist(rng);
            double norm = std::sqrt(x * x + y * y + z * z);
            if (norm < 1e-6) continue;

            x /= norm; y /= norm; z /= norm;

            // 与两个方向都成锐角（即 dot > 0）
            double dot1 = x * dir_AB.x + y * dir_AB.y + z * dir_AB.z;
            double dot2 = x * dir_BC.x + y * dir_BC.y + z * dir_BC.z;

            if (dot1 > 0 && dot2 > 0) {
                directions.emplace_back(x, y, z, ts_l);
            }
        }

        bool improved = false;
        for (const Coords_ts& dir : directions) {
            double step_list[] = {1.0, 0.5, 0.25, 0.1, 0.02};
            for (double step : step_list) {
                Coords_ts newB(
                    B.x + step * dir.x,
                    B.y + step * dir.y,
                    B.z + step * dir.z,
                    ts_l
                );

                double new_cost = total_cost(newB);
                if (new_cost < base_cost && inside_all_regions(newB, targets, ts_l)) {
                    B = newB;
                    improved = true;
                    stall_count = 0;  // 有改进就清零
                    break;
                }
            }
            if (improved) break;
        }

        if (!improved) {
            stall_count++;
            if (stall_count >= stall_limit) break;  // 多轮失败，退出
        }
    }
}


void CGSOA::gro(Coords_tsVector &path, TargetPtrVector targets) {
    const int N = path.size();
    if (N < 3) return;

    for (int i = 0; i < N; ++i) {
        int f = (i - 1 + N) % N;
        int a = (i + 1) % N;
       std::vector<int> n_ts = path[i].ts;
       if (!(n_ts.size() == 1 && n_ts[0] == 0)) {
          optimize_B(path[f], path[i], path[a], targets);
       }
    }
}


void CGSOA::print_path_with_neighborhoods(const Coords_tsVector& path,
                                   const TargetPtrVector& targets) {
   int N = path.size();
   for (int i = 0; i < N; ++i) {
      std::vector<int>  ts_l = path[i].ts;

      std::cout << "路径点 " << i << " : " << path[i].x << "  " << path[i].y << "  " << path[i].z << " 属于邻域 [";
      for (int t : ts_l) {
         std::cout << t;
         std::cout << ", ";
      }
      std::cout << "]\n";
   }
}
















/// - public method ------------------------------------------------------------
std::string CGSOA::getRevision(void) 
{
   return "$Id: gsoa_3d.cc 241 2025-1-10 21:44:59Z jf $";
}

/// - public method ------------------------------------------------------------
void CGSOA::visualize(const Coords_tsVector & path)//可视化
{
   load();
   finalSolution = path;
   drawPath();
   saveCanvas();
}

/// - public method ------------------------------------------------------------
void CGSOA::solve(void) 
{
   crl::CRandom::randomize();
   Base::solve();
}

/// - protected method ---------------------------------------------------------
void CGSOA::load(void)
{
   DEBUG("GSOA::load -- done");
}

/// - protected method ---------------------------------------------------------
void CGSOA::initialize(void)
{
   permutation.clear();
   foreach(const STarget *target, targets) {
      permutation.push_back(target->label);
   }
}

/// - protected method ---------------------------------------------------------
void CGSOA::after_init(void) 
{
   //  tLoad.append(loadTimer);
   //  tInit.append(initTimer);
}



/// - protected method ---------------------------------------------------------
void CGSOA::iterate(int iter) 
{
   crl::CRandom::randomize();
   //清空结果数组
   finalSolution.clear();
   //记录最佳的结果 步数
   int finalBestSolutionStep;

   TargetPtrVector allTargets; 

   ring->initialize_neurons(targets[0]->coords);
   foreach(STarget *target, targets) {
      target->selectedWinner = 0;
      target->stepWinnerSelected = -1;
   }
   schema->G = 10;
   schema->mi = config.get<double>("learning-rate");
   //thresholds for the termination conditions 终止条件阈值
   //error代表当前胜利节点 到 邻域的距离
   const double MAX_ERROR = config.get<double>("termination-error");
   //终止最大步数
   const int MAX_STEPS = config.get<int>("termination-max-steps");
   //终止变更次数
   const bool TERM_CHANGE = config.get<bool>("termination-change");
   if (canvas) {
      *canvas << canvas::CLEAR << "path" << "path";
   }
   double error = 2 * MAX_ERROR;
   int step = 0;
   Coords_tsVector solution;
   Coords_tsVector bestSolution;
   int bestSolutionStep = -1;
   double bestSolutionLength = std::numeric_limits<double>::max();
   const bool BEST_SOLUTION = config.get<bool>("best-solution");
   bool term = false;
   IntVector routes[2];
   int routeCur = 0;
   int routePrev = 1;
   while (!((error < MAX_ERROR)) && (step < MAX_STEPS) && not term) { //perform adaptation step
     //执行一次适配，更新误差
      error = refine(step, error);
      DEBUG("Step: " << step << " G: " << schema->G << " error: " << error);
      //是否需要记录最佳结果
      if (BEST_SOLUTION) {
         getSolution(step, solution); //collect solution
//         const double len = get_path_length(solution);
         const double len = get_path_time_cost_ts(solution);
         if (len < bestSolutionLength) {
            bestSolution = solution;
            bestSolutionStep = step;
            bestSolutionLength = len;
         }
      }
      //环是否改变
      if (TERM_CHANGE) {
         ring->get_ring_route(step, routes[routeCur]);
         if (routes[routeCur].size() == routes[routePrev].size()) {
            term = true;
            for (int i = 0; i < routes[routeCur].size(); ++i) {
               if (routes[routeCur][i] != routes[routePrev][i]) {
                  term = false;
                  break;
               }
            }
         }
         routePrev = (routePrev + 1)%2;
         routeCur = (routeCur + 1)%2;
      }
      schema->G = schema->G * (1 - schema->GAIN_DECREASING_RATE * (step + 1));
      step++;
   } //end step loop
   tSolve.stop();
   double length;
   if (BEST_SOLUTION) {
      finalSolution = bestSolution;
      length = bestSolutionLength;
      finalBestSolutionStep = bestSolutionStep;
   } else {
      getSolution(step - 1, finalSolution); //collect solution
//      length = get_path_length(finalSolution);
      length = get_path_time_cost_ts(finalSolution);
      finalBestSolutionStep = step - 1;
   }
   DEBUG("config.get<bool>(2opt-post): " << config.get<bool>("2opt-post"));
   if (config.get<bool>("2opt-post")) {
      two_opt_cost(finalSolution);

//      double twoOptLength = get_path_length(finalSolution);
      double twoOptLength = get_path_time_cost_ts(finalSolution);
      DEBUG("cost: " << length << " after 2opt: " << twoOptLength);
      length = twoOptLength;

      gro(finalSolution, targets);
      double hill_climbing_Length = get_path_time_cost_ts(finalSolution);

      length = hill_climbing_Length;

      DEBUG("cost: " << twoOptLength << " after hill_climbing: " << hill_climbing_Length);

      print_path_with_neighborhoods(finalSolution,targets);
   }
   fillResultRecord(iter);
   resultLog
      << length // 
      << step
      << finalBestSolutionStep
      << crl::result::endrec;
   DEBUG("Best solution with the cost: " << bestSolutionLength << " found in: " << bestSolutionStep << " steps");
}

/// - protected method ---------------------------------------------------------
double CGSOA::refine(int step, double errorMax)
{
   double errorToGoal = errorMax;
   double error = 0.0;

   std::vector<bool> targetVisited(targets.size(), false);  // 初始化为 false
   //打乱邻域顺序
   permute(permutation);
   //更新学习规则中与目标数量和当前步数相关的期望值
   schema->updateExp(targets.size(), step);

   for (IntVector::iterator i = permutation.begin(); i != permutation.end(); i++) {
     if (!targetVisited[*i]){
        STarget *target = targets[*i];
//        SNeuron *prevWinner = target->stepWinnerSelected == step - 1 ? target->selectedWinner : 0;

        SWinnerSelection* winner = ring->selectWinner_3d(step, target, errorToGoal);
        if (winner and winner->hasWinner) {
           targetVisited[*i] = true;
           SNeuron* win = ring->adapt_3d(step);
           win->addTargetOnTour(*i);
           // 检查当前点是否在其他邻域内
           for (size_t j = 0; j < targets.size(); ++j) {
             if(*i != j){
                 // 如果这个目标尚未被访问且目标的邻域包含了 win->alternateGoal
                 if (!targetVisited[j]) {
                   if (targets[j]->isPointInside(win->alternateGoal)){ //访问点是否在当前邻域内部
                      // 如果包含，跳过当前目标，直接进入下一个循环
                      targetVisited[j] = true;  // 标记该邻域已访问
                      targets[j]->stepWinnerSelected = step;
                      targets[j]->selectedWinner = win;
                      win->addTargetOnTour(targets[j]->label);
                   }
                 }else{
                    if (targets[j]->isPointInside(win->alternateGoal) && targets[j]->selectedWinner != win && targets[j]->stepWinnerSelected==step ){//这个邻域是在本循环之前选到的，
                       if ((targets[j]->selectedWinner != nullptr)){
                          targets[j]->selectedWinner->removeTargetOnTour(targets[j]->label);
                          if(targets[j]->selectedWinner->targetOnTour.empty()){
                             //把邻域之前对应的节点解除对应(还有可能有其他的关联点，所以不要直接接触，要判断一下)
                             targets[j]->selectedWinner->targetOnTourStep =-1;
                          }
                       }
                       targets[j]->selectedWinner = win;
                       win->addTargetOnTour(targets[j]->label);
                    }
                 }
              }
           }
           if (error < errorToGoal) {
              error = errorToGoal; //update error
           }
        }
     }


   } //end permutation of all targets
   ring->regenerate(step);


   return error; // return largest error to city
}

/// - protected method ---------------------------------------------------------
void CGSOA::save(void) 
{
//   std::string dir;
//   updateResultRecordTimes(); //update timers as load and initilization is outside class
//   if (SAVE_SETTINGS) {
//      saveSettings(getOutputIterPath(config.get<std::string>("settings"), dir));
//   }
//   if (SAVE_INFO) {
//      saveInfo(getOutputIterPath(config.get<std::string>("info"), dir));
//   }
//   if (SAVE_RESULTS) {
//      std::string file = getOutputIterPath(config.get<std::string>("result-path"), dir);
//      assert_io(createDirectory(dir), "Cannot create file in the path'" + file + "'");
//
//      const int i = 0;
//      std::stringstream ss;
//      ss << file << "-" << std::setw(2) << std::setfill('0') << i << ".txt";
//      std::ofstream ofs(ss.str());
//      assert_io(not ofs.fail(), "Cannot create path '" + ss.str() + "'");
//      ofs << std::setprecision(14);
//      foreach(const Coords_st &pt, finalSolution) {
//         ofs << pt.x << " " << pt.y << " "<<pt.z << std::endl;
//      }
//      assert_io(not ofs.fail(), "Error occur during path saving");
//      ofs.close();
//   }
//   if (canvas) { // map must be set
//      *canvas << canvas::CLEAR << "ring";
//      if (config.get<bool>("draw-path")) {
//         drawPath();
//      } else if (DRAW_RING_ENABLE) {
//         drawRing(-1);
//      }
//      saveCanvas();
//   }
}

/// - protected method ---------------------------------------------------------
void CGSOA::release(void) 
{
}

/// - protected method ---------------------------------------------------------
void CGSOA::defineResultLog(void) 
{
   static bool resultLogInitialized = false;
   if (!resultLogInitialized) {
      resultLog << result::newcol << "NAME";
      resultLog << result::newcol << "METHOD";
      resultLog << result::newcol << "TRIAL";
      resultLog << result::newcol << "RTIME";
      resultLog << result::newcol << "CTIME";
      resultLog << result::newcol << "UTIME";
      resultLog << result::newcol << "LENGTH"; 
      resultLog << result::newcol << "STEPS";
      resultLog << result::newcol << "SOLUTION_STEP";
      resultLogInitialized = true;
   }
}

/// - protected method ---------------------------------------------------------
void CGSOA::fillResultRecord(int trial) 
{
   resultLog << result::newrec << name << method << trial;
   long t[3] = {0l, 0l, 0l};
   tLoad.addTime(t);
   tInit.addTime(t);
   tSolve.addTime(t);
   tSave.addTime(t);
   resultLog << t[0] << t[1] << t[2];
}

/// - private method -----------------------------------------------------------
void CGSOA::drawPath(void)
{
//   if (canvas) {
//      *canvas
//         << canvas::CLEAR << "path" << "path"
//         << CShape(config.get<std::string>("draw-shape-path"))
//         << canvas::LINESTRING
//         << finalSolution
//         << finalSolution.front()
//         << canvas::END;
//
//      if (config.get<bool>("draw-path-nodes")) {
//         *canvas << canvas::POINT << shapePathNodes << finalSolution;
//      } //end draw-path-nodes
//   } //end if canvas
}

/// - private method -----------------------------------------------------------
void CGSOA::drawRing(int step)
{
//   if (canvas) {
//      *canvas << canvas::CLEAR << "ring" << "ring";
//      if (DRAW_RING_ENABLE) {
//         *canvas
//            << canvas::LINESTRING << shapeRing
//            << ring << ring->begin()
//            << canvas::END;
//      } //end ring
//      if (DRAW_RING_NODES) {
//         *canvas << canvas::POINT << shapeNeurons << ring;
//      }
//      if (DRAW_TOUR_REPRESENTED_BY_RING) {
//	 CoordsVector path;
//	 double r;
//	 ring->get_ring_path(step, path);
//	 if (path.size() > 1) {
//	    *canvas
//	       << shapeTourRepresentedByRing << canvas::LINESTRING
//	       << path;
//	    *canvas << canvas::END;
//	 }
//      }
//   } //end canvas
}

/// - private method -----------------------------------------------------------
void CGSOA::savePic(int step, bool detail, const std::string &dir_suffix)
{
   static int lastStep = step;
   static int i = 0;
   if (lastStep != step) {
      i = 0;
   }
   if (canvas) {
      canvas->redraw();
      std::string dir;
      std::string file = getOutputIterPath(config.get<std::string>("pic-dir") + dir_suffix, dir);
      assert_io(createDirectory(file), "Cannot create file in path '" + file + "'");
      std::stringstream ss;
      ss << file << "/" << "iter-" << std::setw(3) << std::setfill('0') << step;
      ss << "-" << std::setw(4) << std::setfill('0') << i;

      std::string suffixes(config.get<std::string>("pic-ext"));
      if (!suffixes.empty()) {
	 std::string::size_type cur = 0;
	 std::string::size_type next;
	 do {
	    next = suffixes.find(',', cur);
	    const std::string &ext = suffixes.substr(cur, next - cur);
	    if (!ext.empty()) {
	       assert_io(canvas->save(ss.str() + "." + ext), "Cannot create output canvas file '" + file + "'");
	    }
	    cur = next + 1;
	 } while (next != std::string::npos);
      } else {
	 ss << "." << config.get<std::string>("pic-ext");
	 assert_io(canvas->save(ss.str()), "Cannot create output canvas file '" + ss.str() + "'");
      }
   }
   lastStep = step;
   i++;
}

/// - private method -----------------------------------------------------------
void CGSOA::getSolution(int step, Coords_tsVector &solution) const
{
   ring->get_ring_path(step, solution,targets);
}

//CoordsVector api_func(CGSOA &gsoa,TargetPtrVector targets){
//   return gsoa.api_run(targets);
//}
/* end of gsoa.cc */
