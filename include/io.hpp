/***************************************************************************************
 * Project: openMPSPH
 * File: io.hpp
 * Description:
 *     This program is a Smoothed Particle Hydrodynamics (SPH) solver
 *     developed for simulating dam break flow.
 * Author:      Kan Liu, Master.
 * Affiliation: School of Aerospace Engineering. Beijing Institue of Technology.
 * Email:       lkbhg@outlook.com
 * GitHub:      https://github.com/lkbhg
 *
 * Created:     October 2025
 * Last Updated: October 2025
 *
 * Language:    C++17
 * Parallelism: OpenMP
 * Build Tool:  CMake
 *
 * -------------------------------------------------------------------------------------
 * Copyright (c) 2025 Kan Liu
 * This software is distributed under the AGPL License.
 * If you use this software in academic work, please cite:
 * Kan Liu, "openMPSPH: A C++ OpenMP framework for dam break", GitHub Repository, 2025.
 * -------------------------------------------------------------------------------------
 ***************************************************************************************/
#ifndef __IO__H
#define __IO__H

#include <string>

// 判断路径是否为绝对路径
bool is_absolute_path(const std::string &path);

// 相对路径转完整路径
std::string get_full_path(const std::string &base_path, const std::string &filename);

#endif
