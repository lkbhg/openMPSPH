/***************************************************************************************
 * Project: openMPSPH
 * File: main.cpp
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
 * ------------------------------------------------------------------------------------
 ***************************************************************************************/


#include <omp.h>
#include <iostream>
#include <cstring>
#include <cstdlib>
#include "solver.hpp"

int main(int argc, char *argv[])
{
    int nthreads = 1; // 0 表示使用默认线程数，但不一定为1

    // 命令行参数
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "-thread") == 0 && i + 1 < argc)
        {
            nthreads = std::atoi(argv[i + 1]);
            i++; // 跳过下一个参数
        }
    }

    if (nthreads > 0)
    {
        omp_set_num_threads(nthreads);
        std::cout << "OpenMP threads set to: " << nthreads << std::endl;
    }
    else
    {
        std::cout << "Using default OpenMP threads." << std::endl;
    }

    solver s(nthreads);

    return 0;
}




