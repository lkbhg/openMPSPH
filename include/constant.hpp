/***************************************************************************************
 * Project: openMPSPH
 * File: constant.hpp
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
#ifndef __CONSTANT__H
#define __CONSTANT__H

#include <string>

using real = double;
using un = int;
// using un = uint32_t;
using index_t = std::size_t;

// system
inline constexpr un dimension = 2;       // Number of Spatial Dimension
inline constexpr un MaxParticleNum = 5000;       // Maximum number of particles
inline constexpr un MaxParticlePair = 20 * MaxParticleNum; // Maximum number of particle pairs
inline constexpr un DensityRei = 20;           // Timesteps for density reinitialization
inline constexpr un NTsave = 500;        // Timesteps to save data

// simulation
inline const std::string results_path = "../data/";
inline constexpr real PI = 3.14159265358979;
inline constexpr real Gravity = -9.8;
inline constexpr real RefRho = 1000;
inline constexpr real Mu = 1.0E-3;
inline constexpr real Alpha = 0.01;
inline constexpr real SmoothingLength = 1.0; 

inline constexpr real SoundVelocity = 50.0;
inline constexpr real ds = 0.01;
inline constexpr real Dt = 0.00004; // suggest dt = ds / (SoundVelocity * 5) 内置了dt计算过程，除非特殊需求无需手动更改。
inline constexpr real StartTime = 0.0;
inline constexpr real EndTime = 2.0;
inline constexpr real WallLength = 1.0;
inline constexpr real WallHeight = 0.5;
inline constexpr real LiquidLength = 0.3;
inline constexpr real LiquidHeight = 0.2;

#endif
