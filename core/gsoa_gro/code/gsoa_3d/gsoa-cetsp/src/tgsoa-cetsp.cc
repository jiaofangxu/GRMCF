/*
 * File name: tgsoa-cetsp.cc
 * Date:      2016/12/07 08:33
 * Author:    Jan Faigl
 */
#include <boost/program_options.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include <chrono>
#include <crl/config.h>
#include <crl/logging.h>
#include <crl/perf_timer.h>
#include <crl/boost_args_config.h>

#include <crl/gui/guifactory.h>
#include <crl/gui/win_adjust_size.h>
#include <time.h>
#include "gsoa_3d.h"
#include "target_3d.h"
#include "coords_3d.h"
#include "crow.h"


using crl::logger;
using namespace gsoa;

namespace po = boost::program_options;
namespace fs = boost::filesystem;

const std::string GSOA_VERSION = "0.4";

typedef crl::gui::CCanvasBase Canvas;

/// ----------------------------------------------------------------------------
/// Program options variables
/// ----------------------------------------------------------------------------
std::string guiType = "none";

crl::CConfig guiConfig;
crl::CConfig gsoaConfig;
std::string canvasOutput = "";
std::string solutionFile = "";

/// ----------------------------------------------------------------------------
/// Global variable
/// ----------------------------------------------------------------------------
crl::gui::CGui* g = 0;
#define GUI(x)  if(gui) { x;}

/// ----------------------------------------------------------------------------
bool parseArgs(int argc, char* argv[])
{
    bool ret = true;
    std::string configFile;
    std::string guiConfigFile;
    std::string loggerCfg = "";

    po::options_description desc("General options");
    desc.add_options()
        ("help,h", "produce help message")
        //读取函数名+.cfg的配置文件
        ("config,c", po::value<std::string>(&configFile)->default_value(std::string(argv[0]) + ".cfg"),
         "configuration file")
        ("logger-config,l", po::value<std::string>(&loggerCfg)->default_value(loggerCfg),
         "logger configuration file")
        ("config-gui", po::value<std::string>(&guiConfigFile)->default_value(std::string(argv[0]) + "-gui.cfg"),
         "dedicated gui configuration file")
        ("solution-file", po::value<std::string>(&solutionFile)->default_value(""));
    try
    {
        po::options_description guiOptions("Gui options");
        crl::gui::CGuiFactory::getConfig(guiConfig);
        crl::gui::CWinAdjustSize::getConfig(guiConfig);
        guiConfig.add<double>("gui-add-x",
                              "add the given value to the loaded goals x coord to determine the canvas size and transformation",
                              0);
        guiConfig.add<double>("gui-add-y",
                              "add the given value to the loaded goals y coord to determine the canvas size and transformation",
                              0);
        boost_args_add_options(guiConfig, "", guiOptions);
        guiOptions.add_options()
            ("canvas-output", po::value<std::string>(&canvasOutput), "result canvas outputfile");

        po::options_description gsoaOptions("GSOA options");
        boost_args_add_options(CGSOA::getConfig(gsoaConfig), "", gsoaOptions);

        po::options_description cmdline_options;
        cmdline_options.add(desc).add(guiOptions).add(gsoaOptions);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, cmdline_options), vm);
        po::notify(vm);

        std::ifstream ifs(configFile.c_str());
        store(parse_config_file(ifs, cmdline_options), vm);
        po::notify(vm);
        ifs.close();
        ifs.open(guiConfigFile.c_str());
        store(parse_config_file(ifs, cmdline_options), vm);
        po::notify(vm);
        ifs.close();

        if (vm.count("help"))
        {
            std::cerr << std::endl;
            std::cerr << "GSOA solver ver. " << GSOA_VERSION << std::endl;
            std::cerr << cmdline_options << std::endl;
            ret = false;
        }
        if (
            ret &&
            loggerCfg != "" &&
            fs::exists(fs::path(loggerCfg))
        )
        {
            crl::initLogger("gsoa", loggerCfg.c_str());
        }
        else
        {
            crl::initLogger("gsoa");
        }
        const std::string problemFile = gsoaConfig.get<std::string>("problem");
        if (!fs::exists(fs::path(problemFile)))
        {
            ERROR("Problem file '" + problemFile + "' does not exists");
            ret = false;
        }
    }
    catch (std::exception& e)
    {
        std::cerr << std::endl;
        std::cerr << "Error in parsing arguments: " << e.what() << std::endl;
        ret = false;
    }
    return ret;
}


/// - main ---------------------------------------------------------------------
// int main(int argc, char *argv[])
// {
// 	//int argc, char *argv[]这两个参数是空的
//     Canvas *canvas = 0;
//     int ret = -1;
//
//
//
//
//    if (parseArgs(argc, argv)) {
//       INFO("Start Logging");
//       try {
// 	 CoordsVector pts;
// 	 //
// 	 // if (gsoaConfig.get<bool>("variable-radius")) {
// 	 //    double x, y, r;
// 	 //    std::ifstream in(gsoaConfig.get<std::string>("problem").c_str());
// 	 //    while (in >> x >> y >> r) {
// 	 //       pts.push_back(Coords(x - r, y - r,0));
// 	 //       pts.push_back(Coords(x + r, y + r,0));
// 	 //    }
// 	 // } else {
// 	 //    crl::CPerfTimer t("Load problem time real:");
// 	 //    double r = gsoaConfig.get<double>("communication-radius");
// 	 //    double x, y;
// 	 //    std::ifstream in(gsoaConfig.get<std::string>("problem").c_str());
// 	 //    while (in >> x >> y) {
// 	 //       pts.push_back(Coords(x - r, y - r,0);
// 	 //       pts.push_back(Coords(x + r, y + r,0));
// 	 //    }
// 	 // }
// 	 crl::gui::CWinAdjustSize::adjust(pts, guiConfig);
// 	 if ((g = crl::gui::CGuiFactory::createGui(guiConfig)) != 0) {
// 	    INFO("Start gui " + guiConfig.get<std::string>("gui"));
// 	    canvas = new Canvas(*g);
// 	 }
// 	 CGSOA gsoa(gsoaConfig);
// 	 gsoa.setCanvas(canvas);
// 	 {
// 	    crl::CPerfTimer t("Total solve time: ");
// 	    if (solutionFile.empty()) {
// 	       gsoa.solve();
// 	    } else {
// 	       CoordsVector pts;
// 	       Coords pt;
// 	       std::ifstream in(solutionFile.c_str());
// 	       while (in >> pt.x >> pt.y >> pt.z) {
// 		  pts.push_back(pt);
// 	       }
// 	       gsoa.visualize(pts);
// 	    }
// 	 }
// 	 INFO("End Logging");
// 	 if (canvas) {
// 	    if (canvasOutput.size()) {
// 	       canvas->save(canvasOutput);
// 	    }
// 	    if (!guiConfig.get<bool>("nowait")) {
// 	       INFO("click to exit");
// 	       canvas->click();
// 	    }
// 	    delete canvas;
// 	    delete g;
// 	 }
//       } catch (crl::exception &e) {
// 	 ERROR("Exception " << e.what() << "!");
//       } catch (std::exception &e) {
// 	 ERROR("Runtime error " << e.what() << "!");
//       }
//       ret = EXIT_SUCCESS;
//    }
//    crl::shutdownLogger();
//    return ret;
// }

//
// STarget* create_target_from_json(const int label, const crow::json::rvalue& item)
// {
//     // 确保 "centroid" 是数组，并且大小为 3
//     if (item["centroid"].t() != crow::json::type::List || item["centroid"].size() != 3)
//     {
//         throw std::invalid_argument("Invalid centroid data.");
//     }
//
//     uint64_t hex_x = std::stoull(item["centroid"][0].s(), nullptr, 16);
//     uint64_t hex_y = std::stoull(item["centroid"][1].s(), nullptr, 16);
//     uint64_t hex_z = std::stoull(item["centroid"][2].s(), nullptr, 16);
//     // 解析质心
//     Coords centroid(
//         *reinterpret_cast<double*>(&hex_x),
//         *reinterpret_cast<double*>(&hex_y),
//         *reinterpret_cast<double*>(&hex_z)
//     );
//
//     // 解析约束（面数据）
//     PlanarVector planarVector;
//     if (item.has("constraints"))
//     {
//         if (item["constraints"].t() == crow::json::type::List)
//         {
//             // 如果 "constraints" 是数组
//             if (item["constraints"].size() > 0)
//             {
//                 for (const auto& constraint : item["constraints"])
//                 {
//                     if (constraint.t() == crow::json::type::List && constraint.size() == 4)
//                     {
//                         uint64_t hex_a = std::stoull(constraint[0].s(), nullptr, 16);
//                         uint64_t hex_b = std::stoull(constraint[1].s(), nullptr, 16);
//                         uint64_t hex_c = std::stoull(constraint[2].s(), nullptr, 16);
//                         uint64_t hex_d = std::stoull(constraint[3].s(), nullptr, 16);
//                         planarVector.push_back(Planar(
//                             *reinterpret_cast<double*>(&hex_a),
//                             *reinterpret_cast<double*>(&hex_b),
//                             *reinterpret_cast<double*>(&hex_c),
//                             *reinterpret_cast<double*>(&hex_d)
//                         ));
//                     }
//                     else
//                     {
//                         uint64_t hex_a = std::stoull(constraint[0].s(), nullptr, 16);
//                         // 如果约束面不完整，按照需要的逻辑补充
//                         planarVector.push_back(Planar(
//                             *reinterpret_cast<double*>(&hex_a),
//                             constraint.size() > 1 ? constraint[1].d() : 0.0,
//                             constraint.size() > 2 ? constraint[2].d() : 0.0,
//                             constraint.size() > 3 ? constraint[3].d() : 0.0
//                         ));
//                     }
//                 }
//             }
//             else
//             {
//             }
//         }
//         else
//         {
//         }
//     }
//     else
//     {
//     }
//
//     // 创建并返回目标对象
//     return new STarget(label, planarVector, centroid);
// }

STarget* create_target_from_json_cs(const int label, const crow::json::rvalue& item)
{
    // 确保 "centroid" 是数组，并且大小为 3
    if (item["centroid"].t() != crow::json::type::List || item["centroid"].size() != 3)
    {
        throw std::invalid_argument("Invalid centroid data.");
    }

    // 解析质心
    Coords centroid(
        item["centroid"][0].d(),
        item["centroid"][1].d(),
        item["centroid"][2].d()

    );
    double max_z;
    double min_z;
    // 解析约束（横截面数据）
    CorssSectionVector corssSectionVector;
    if (item.has("cross_sections"))
    {
        if (item["cross_sections"].t() == crow::json::type::List)
        {
            // 如果 "cross_section" 是数组
            if (item["cross_sections"].size() > 0)
            {
                min_z = item["cross_sections"][0][0].d();
                max_z = item["cross_sections"][item["cross_sections"].size() - 1][0].d();

                for (const auto& cross_section : item["cross_sections"])
                {

                    double z = cross_section[0].d();
                    CoordsVector coordsVector;
                    if (cross_section[1].t() == crow::json::type::List)
                    {
                        for (const auto& coords : cross_section[1])
                        {
                            coordsVector.push_back(Coords(coords[0].d(),coords[1].d(),z));
                        }
                    }
                    corssSectionVector.push_back(CorssSection(z,coordsVector));



                }
            }
            else
            {
            }
        }
        else
        {
        }
    }
    else
    {
    }

    // 创建并返回目标对象
    return new STarget(label,corssSectionVector, max_z,min_z, centroid);
}




int main(int argc, char* argv[])
{
    crow::SimpleApp app;


    if (parseArgs(argc, argv))
    {
        INFO("Start Logging");
        try
        {
            CGSOA gsoa(gsoaConfig);

            // 定义 HTTP 路由
            app.route_dynamic("/api/gsoa")
               .methods("POST"_method)([&gsoa](const crow::request& req)
               {

                   // 从请求中加载 JSON 数据
                   auto json_data = crow::json::load(req.body);

                   if (!json_data)
                   {
                       return crow::response(400, "Invalid JSON");
                   }

                   // 解析目标数据
                   TargetPtrVector targets;

                   try
                   {
                       // 循环解析 targets 数组
                       if (json_data["targets"].t() == crow::json::type::List)
                       {
                           for (const auto& item : json_data["targets"])
                           {
                               // 从 JSON 创建 STarget 对象
                               targets.push_back(create_target_from_json_cs(targets.size(), item));
                           }
                       }
                       else
                       {
                           return crow::response(400, "Invalid targets array");
                       }
                   }
                   catch (const std::exception& e)
                   {
                       return crow::response(400, std::string("Error parsing JSON: ") + e.what());
                   }


                   // 记录开始时间

                   struct timespec start, end;
                   clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);

                   // 调用原始函数（api_func）
                   CoordsVector coords = gsoa.api_run(targets);

                   // 记录结束时间
                   clock_gettime(CLOCK_THREAD_CPUTIME_ID, &end);
                   long sec_diff = end.tv_sec - start.tv_sec;
                    long nsec_diff = end.tv_nsec - start.tv_nsec;
                    double cpu_time_ms = sec_diff * 1000.0 + nsec_diff / 1e6;

                   // 打印执行时间
                    std::cout << "Function execution time: " << cpu_time_ms << " ms" << std::endl;
                   // 构造 JSON 响应
                   crow::json::wvalue result;
                   std::vector<std::vector<double>> coords_list; // 使用 vector<vector<double>> 代替 crow::json::wvalue

                   // 遍历 coords 并构建 JSON
                   for (const auto& coord : coords)
                   {
                       coords_list.push_back({coord.x, coord.y, coord.z}); // 直接存入数值数组
                   }

                   // 赋值给 JSON
                   result["coords"] = std::move(coords_list);
                   result["t"] = cpu_time_ms;


                   return crow::response(result);
               });
            // 启动 HTTP 服务器
            app.port(gsoaConfig.get<int>("port")).multithreaded().run();
        }
        catch (crl::exception& e)
        {
            ERROR("Exception " << e.what() << "!");
        } catch (std::exception& e)
        {
            ERROR("Runtime error " << e.what() << "!");
        }
    }
    return 0;
}

/* end of tgsoa-cetsp.cc */
