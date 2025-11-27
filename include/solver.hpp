/***************************************************************************************
 * Project: openMPSPH
 * File: solver.hpp
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

#ifndef __SOLVER__H
#define __SOLVER__H

#include <math.h>
#include <vector>
#include <array>

#include "io.hpp"
#include "constant.hpp"

class solver
{

public:
	solver(un num_threads);
	~solver();
	int solve();

private:
	void ParticleGeneration();

	void TimeIntegral();

	void Equation();

	std::string make_output_path(real tt) const;

	void output_file(real tt);
	void output_file(real tt, bool A);

	void density_correction();

	void InitDomain();

	void Search();

	un NumThreads;
	std::string base_path = "../data/";

	// variable
	un NumLiquidParticle, TotalParticleNum;
	real hs, kh, dt, eta, TotalTimeStep;

	std::vector<std::vector<real>> position;
	std::vector<std::vector<real>> velocity;
	std::vector<std::vector<real>> acceleration;

	std::vector<real> rho;
	std::vector<real> mass;
	std::vector<real> pressure;
	std::vector<real> drho;
	std::vector<real> w;

	// leapfrog
	std::vector<real> rhonBuff;
	std::vector<std::vector<real>> velocityBuff;

	// cache
	std::vector<real> rhot;
	std::vector<real> wv;

	// 初始化
	un ParticleIndex, NumXaxis, NumYaxis;
	real dx, dy;

	real Xmin, Xmax, Ymin, Ymax;
	un CellNumX, CellNumY;
	un MaxParticleOneCell = 20;

	// ------------------- 线程缓冲 -------------------
	// 每个线程的 buffer
	struct ThreadBuffer
	{
		un NumParticlePair;
		std::vector<std::vector<std::vector<un>>> cell;
		std::vector<std::array<un, 2>> pair_local;
		std::vector<std::array<real, 2>> dwdx_local;
		std::vector<real> drho_local;
		std::vector<std::vector<real>> acceleration_local;

		std::vector<real> rhot, w, wv;

		void init(un ParticleOneCell, un MaxCellX, un MaxCellY)
		{
			cell.resize(ParticleOneCell);
			for (un i = 0; i < ParticleOneCell; ++i)
			{
				cell[i].resize(MaxCellX);
				for (un j = 0; j < MaxCellX; ++j)
				{
					cell[i][j].assign(MaxCellY, 0);
				}
			}
			pair_local.clear();
			pair_local.resize(MaxParticlePair); // 预分配，避免频繁扩容

			dwdx_local.clear();
			dwdx_local.resize(MaxParticlePair);

			drho_local.clear();
			drho_local.resize(MaxParticleNum);

			rhot.clear();
			rhot.resize(MaxParticleNum);

			w.clear();
			w.resize(MaxParticleNum);

			wv.clear();
			wv.resize(MaxParticleNum);

			acceleration_local.assign(dimension, std::vector<real>(MaxParticleNum, 0.0));
		}
		void clear()
		{
			NumParticlePair = 0;
			for (auto &layer : cell)
				for (auto &row : layer)
					std::fill(row.begin(), row.end(), 0);

			std::fill(pair_local.begin(), pair_local.end(), std::array<un, 2>{0, 0});
			std::fill(dwdx_local.begin(), dwdx_local.end(), std::array<real, 2>{0, 0});

			std::fill(drho_local.begin(), drho_local.end(), 0.0);
			std::fill(rhot.begin(), rhot.end(), 0.0);
			std::fill(w.begin(), w.end(), 0.0);
			std::fill(wv.begin(), wv.end(), 0.0);

			for (auto &vec : acceleration_local)
				std::fill(vec.begin(), vec.end(), 0.0);
		}
	};

	std::vector<ThreadBuffer> thread_buffers;
	// 初始化线程缓冲
	void InitThreadBuffers(un MaxParticleOneCell, un MaxCellX, un MaxCellY);

	inline void continuous(un &i, un &j, solver::ThreadBuffer &buf, size_t &k);

	inline void pressure_grad(un &i, un &j, solver::ThreadBuffer &buf, size_t &k);

	inline void viscous_artifical(un &i, un &j, solver::ThreadBuffer &buf, size_t &k);

	inline void viscous_laminar(un &i, un &j, solver::ThreadBuffer &buf, size_t &k);

	inline void Equation_of_State(index_t &i);

	inline void gravity(index_t &i);

	inline void kernel(real r, real *dx, real h, real *w, real *dwdx);

	void reduction();
};


#endif
