/*
 * File name: route_path_utils.h
 * Date:      2016/12/10 18:12
 * Author:    Jan Faigl
 */

#ifndef __ROUTE_PATH_UTILS_H__
#define __ROUTE_PATH_UTILS_H__

#include "coords_3d.h"

/// ----------------------------------------------------------------------------
/// @brief get_path_length
/// 
/// @param pts 
/// 
/// @return 
/// ----------------------------------------------------------------------------
double get_path_length(const CoordsVector &pts, bool closed = true);


double T_x(double x) ;

double T_y(double y) ;

double T_z(double z) ;

double stay_time(double dx, double dy, double dz) ;
double cost(const Coords &a, const Coords &b);
double cost_ts(const Coords_ts &a, const Coords_ts &b);
double get_path_time_cost(const CoordsVector &pts, bool closed = true);
double get_path_time_cost_ts(const Coords_tsVector &pts, bool closed = true);





/// ----------------------------------------------------------------------------
/// @brief two_opt
/// 
/// @param path 
/// ----------------------------------------------------------------------------
void two_opt(CoordsVector &path);

void two_opt_cost(Coords_tsVector &path);

#endif

/* end of route_path_utils.h */
