/***************************************************************************************
 * Project: openMPSPH
 * File: solver.cpp
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

#include <cstring>
#include <string>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <errno.h>
#include <omp.h>
#include <time.h>

#include "solver.hpp"
#include "constant.hpp"
#include "io.hpp"

solver::solver(un num_threads) : NumThreads(num_threads),
                                 base_path(results_path),
                                 position(dimension, std::vector<real>(MaxParticleNum, 0)),
                                 velocity(dimension, std::vector<real>(MaxParticleNum, 0)),
                                 acceleration(dimension, std::vector<real>(MaxParticleNum, 0)),
                                 rho(MaxParticleNum, 0),
                                 mass(MaxParticleNum, 0),
                                 pressure(MaxParticleNum, 0),
                                 drho(MaxParticleNum, 0),
                                 w(MaxParticlePair, 0),
                                 rhot(MaxParticleNum, 0),
                                 wv(MaxParticleNum, 0),
                                 rhonBuff(MaxParticleNum, 0),
                                 velocityBuff(dimension, std::vector<real>(MaxParticleNum, 0))

{

    if (num_threads > 0)
    {
        omp_set_num_threads(num_threads);
        // std::cout << "OpenMP threads set to: " << NumThreads << std::endl;
    }
    else
    {
        std::cout << "Number of threads is less than 1." << std::endl;
        std::cout << "Using default 1 threads." << std::endl;
    }

    dt = (Dt < (ds / (SoundVelocity * 5))) ? Dt : (ds / (SoundVelocity * 5));

    ParticleIndex = 0;
    NumXaxis = std::round(LiquidLength / ds);
    NumYaxis = std::round(LiquidHeight / ds);
    dx = LiquidLength / NumXaxis;
    dy = LiquidHeight / NumYaxis;

    hs = SmoothingLength * ds;
    kh = 2.0 * hs;
    eta = 0.01 * hs * hs;

    ParticleGeneration();

    InitDomain();

    InitThreadBuffers(MaxParticleOneCell, CellNumX, CellNumY);

    solve();
}

solver::~solver()
{
}

void solver::ParticleGeneration()
{
    // ------------------------
    // 液体粒子
    // ------------------------
    un numLiquid = NumXaxis * NumYaxis;

#pragma omp parallel for collapse(2)
    for (int i = 1; i <= NumXaxis; i++)
    {
        for (int j = 1; j <= NumYaxis; j++)
        {
            un local = (i - 1) * NumYaxis + (j - 1); // 从0开始
            un idx = local;

            position[0][idx] = i * dx - dx / 2;
            position[1][idx] = j * dy - dy / 2;
            pressure[idx] = 0;
            rho[idx] = RefRho;
            mass[idx] = rho[idx] * dx * dy;
        }
    }
    NumLiquidParticle = numLiquid;

    NumXaxis = round(WallLength / ds);
    NumYaxis = round(WallHeight / ds);
    dx = WallLength / NumXaxis;
    dy = WallHeight / NumYaxis;

    // ------------------------
    // 左边界
    // i ∈ [-2,0], 共3列; j ∈ [1, NumYaxis], 共 NumYaxis 行
    // ------------------------
    un numLeft = 3 * NumYaxis;
    un leftOffset = NumLiquidParticle;

#pragma omp parallel for collapse(2)
    for (int i = -2; i <= 0; i++)
    {
        for (int j = 1; j <= NumYaxis; j++)
        {
            un local = (i + 2) * NumYaxis + (j - 1);
            un idx = leftOffset + local;

            position[0][idx] = i * dx - dx / 2;
            position[1][idx] = j * dy - dy / 2;
            pressure[idx] = 0;
            rho[idx] = RefRho;
            mass[idx] = rho[idx] * dx * dy;
        }
    }

    // ------------------------
    // 底部边界
    // i ∈ [-2, NumXaxis+3], 共 NumXaxis+6 列; j ∈ [-2,0], 共3行
    // ------------------------
    un numBottom = (NumXaxis + 6) * 3;
    un bottomOffset = leftOffset + numLeft;

#pragma omp parallel for collapse(2)
    for (int i = -2; i <= NumXaxis + 3; i++)
    {
        for (int j = -2; j <= 0; j++)
        {
            un local = (i + 2) * 3 + (j + 2);
            un idx = bottomOffset + local;

            position[0][idx] = i * dx - dx / 2;
            position[1][idx] = j * dy - dy / 2;
            pressure[idx] = 0;
            rho[idx] = RefRho;
            mass[idx] = rho[idx] * dx * dy;
        }
    }

    // ------------------------
    // 右边界
    // i ∈ [NumXaxis+1, NumXaxis+3], 共3列; j ∈ [1, NumYaxis], 共 NumYaxis 行
    // ------------------------
    un numRight = 3 * NumYaxis;
    un rightOffset = bottomOffset + numBottom;

#pragma omp parallel for collapse(2)
    for (int i = NumXaxis + 1; i <= NumXaxis + 3; i++)
    {
        for (int j = 1; j <= NumYaxis; j++)
        {
            un local = (i - (NumXaxis + 1)) * NumYaxis + (j - 1);
            un idx = rightOffset + local;

            position[0][idx] = i * dx - dx / 2;
            position[1][idx] = j * dy - dy / 2;
            pressure[idx] = 0;
            rho[idx] = RefRho;
            mass[idx] = rho[idx] * dx * dy;
        }
    }

    // ------------------------
    // 顶部边界
    // i ∈ [-2, NumXaxis+3], 共 NumXaxis+6 列; j ∈ [NumYaxis+1, NumYaxis+3], 共3行
    // ------------------------
    un numTop = (NumXaxis + 6) * 3;
    un topOffset = rightOffset + numRight;

#pragma omp parallel for collapse(2)
    for (int i = -2; i <= NumXaxis + 3; i++)
    {
        for (int j = NumYaxis + 1; j <= NumYaxis + 3; j++)
        {
            un local = (i + 2) * 3 + (j - (NumYaxis + 1));
            un idx = topOffset + local;

            position[0][idx] = i * dx - dx / 2;
            position[1][idx] = j * dy - dy / 2;
            pressure[idx] = 0;
            rho[idx] = RefRho;
            mass[idx] = rho[idx] * dx * dy;
        }
    }

    // 总粒子数
    TotalParticleNum = NumLiquidParticle + numLeft + numBottom + numRight + numTop;
}

void solver::InitDomain()
{
    // ------------------ 确定搜索空间 ------------------
    Xmin = position[0][0];
    Xmax = position[0][0];
    Ymin = position[1][0];
    Ymax = position[1][0];

    for (un i = 0; i < TotalParticleNum; i++)
    {
        Xmin = std::min(Xmin, position[0][i]);
        Xmax = std::max(Xmax, position[0][i]);
        Ymin = std::min(Ymin, position[1][i]);
        Ymax = std::max(Ymax, position[1][i]);
    }

    Xmin -= 1.5 * kh;
    Xmax += 1.5 * kh;
    Ymin -= 1.5 * kh;
    Ymax += 1.5 * kh;
    CellNumX = static_cast<un>(std::ceil((Xmax - Xmin) / kh)) + 2; // 边界处理
    CellNumY = static_cast<un>(std::ceil((Ymax - Ymin) / kh)) + 2; // 边界处理
}

void solver::InitThreadBuffers(un MaxParticleOneCell, un MaxCellX, un MaxCellY)
{
    int nthreads = NumThreads;

    thread_buffers.resize(nthreads);

#pragma omp parallel for
    for (int i = 0; i < nthreads; i++)
    {
        thread_buffers[i].init(MaxParticleOneCell + 1, MaxCellX, MaxCellY); // 最后一位用于计数
    }
}

int solver::solve()
{
    using clock = std::chrono::high_resolution_clock;
    using std::chrono::duration;
    using std::chrono::duration_cast;
    using std::chrono::seconds;

    auto start = clock::now();

    if (EndTime <= StartTime)
    {
        std::cerr << "StartTime >= EndTime.\n";
        return 0;
    }

    TotalTimeStep = static_cast<un>(std::round((EndTime - StartTime) / dt));

    std::cout << "Fluid particle: " << NumLiquidParticle << '\n'
              << "wall particle: " << (TotalParticleNum - NumLiquidParticle) << '\n';

    // 检测
    output_file(StartTime);

    // 计算
    TimeIntegral();

    // 输出
    output_file(EndTime);

    auto end = clock::now();

    // 以秒为单位
    duration<double> usage = end - start;

    std::cout << usage.count() << " s\n";

    return 0;
}

std::string solver::make_output_path(real tt) const
{
    // 文件名
    std::ostringstream oss;
    oss << "t=" << tt << "s.dat";
    std::string filename = oss.str();

    // 拼接路径
    if (base_path.empty())
    {
        // 没有设置 base_path，则使用当前工作目录
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) == nullptr)
        {
            throw std::runtime_error("getcwd失败: " + std::string(strerror(errno)));
        }
        return get_full_path(cwd, filename);
    }
    else
    {
        return get_full_path(base_path, filename);
    }
}

void solver::output_file(real tt, bool A)
{
    try
    {
        // 生成完整文件路径
        std::string full_path = make_output_path(tt);

        std::cout << "正在写文件: " << full_path << std::endl;

        // 打开文件
        std::ofstream f(full_path);
        if (!f.is_open())
        {
            throw std::runtime_error("无法打开文件 " + full_path + ": " + std::string(strerror(errno)));
        }

        // 写文件头（Tecplot格式）
        f << "title=\"t=" << tt << " s\"\n";
        f << "variables = x, y, u, v, p, rho, mass\n";

        // 液体粒子 zone
        f << "zone i= " << NumLiquidParticle << " f=point\n";
        for (un i = 0; i < NumLiquidParticle; ++i)
        {
            f << std::setprecision(10) << position[0][i] << "   "
              << position[1][i] << "   "
              << velocity[0][i] << "   "
              << velocity[1][i] << "   "
              << pressure[i] << "   "
              << rho[i] << "   "
              << mass[i] << "\n";
        }

        // 固体壁面粒子 zone
        if (TotalParticleNum > NumLiquidParticle)
        {
            f << "zone i= " << (TotalParticleNum - NumLiquidParticle) << " f=point\n";
            for (un i = NumLiquidParticle; i < TotalParticleNum; ++i)
            {
                f << std::setprecision(10) << position[0][i] << "   "
                  << position[1][i] << "   "
                  << velocity[0][i] << "   "
                  << velocity[1][i] << "   "
                  << pressure[i] << "   "
                  << rho[i] << "   "
                  << mass[i] << "\n";
            }
        }

        f.close();
    }
    catch (const std::exception &e)
    {
        std::cerr << "文件输出错误: " << e.what() << std::endl;
        throw;
    }
}

void solver::output_file(real tt)
{
    try
    {
        // 生成完整文件路径
        std::string full_path = make_output_path(tt);
        std::cout << "正在写文件: " << full_path << "\t";

        std::ofstream f(full_path, std::ios::out);
        if (!f.is_open())
            throw std::runtime_error("无法打开文件 " + full_path);

        // Tecplot ASCII 文件头
        f << "title=\"t=" << tt << " s\"\n";
        f << "variables = x, y, u, v, p, rho, mass\n";

        // 每行最大字符长度
        const size_t line_len = 128;
        // 每块粒子数，可调节
        const size_t block_size = 1000;

        // lambda: 格式化单行
        auto fill_line = [](char *ptr, double x, double y, double u, double v,
                            double p, double r, double m)
        {
            int n = snprintf(ptr, 128, "%.10f %.10f %.10f %.10f %.10f %.10f %.10f\n",
                             x, y, u, v, p, r, m);
            if (n < 0 || n >= 128)
                throw std::runtime_error("单行长度溢出");
            return n;
        };

        // lambda: 写粒子块
        auto write_particles_block = [&](un start_idx, un count)
        {
            for (un b = 0; b < count; b += block_size)
            {
                un this_block = (block_size < (count - b)) ? block_size : (count - b);
                std::vector<char> buffer(this_block * line_len);

                std::vector<int> line_bytes(this_block);

// 并行生成每行字符串
#pragma omp parallel for
                for (un i = 0; i < this_block; ++i)
                {
                    char *ptr = buffer.data() + i * line_len;
                    un idx = start_idx + b + i;
                    line_bytes[i] = fill_line(ptr,
                                              position[0][idx],
                                              position[1][idx],
                                              velocity[0][idx],
                                              velocity[1][idx],
                                              pressure[idx],
                                              rho[idx],
                                              mass[idx]);
                }

                // 主线程顺序写入文件，每行写入实际字符长度
                for (un i = 0; i < this_block; ++i)
                {
                    f.write(buffer.data() + i * line_len, line_bytes[i]);
                }
            }
        };

        // ----------------- 液体粒子 -----------------
        f << "zone i= " << NumLiquidParticle << " f=point\n";
        write_particles_block(0, NumLiquidParticle);

        // ----------------- 固体壁面粒子 -----------------
        if (TotalParticleNum > NumLiquidParticle)
        {
            un wall_count = TotalParticleNum - NumLiquidParticle;
            f << "zone i= " << wall_count << " f=point\n";
            write_particles_block(NumLiquidParticle, wall_count);
        }

        f.close();
        std::cout << "文件写入完成: " << full_path << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "文件输出错误: " << e.what() << std::endl;
        throw;
    }
}

void solver::TimeIntegral()
{

    const real dt2 = dt * 0.5;
    real Time = StartTime;

    // -------------------- 初始化 --------------------
    Search();   // 邻域搜索
    Equation(); // 物理公式

    auto update_half_step = [this, dt2]()
    {
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < NumLiquidParticle; ++i)
        {
            rhonBuff[i] = rho[i];
            rho[i] += drho[i] * dt2;

#pragma unroll 2
            for (index_t j = 0; j < dimension; ++j)
            {
                velocityBuff[j][i] = velocity[j][i];
                velocity[j][i] += acceleration[j][i] * dt2;
            }
        }
    };

    auto update_full_step = [this]()
    {
#pragma omp parallel for schedule(static)
        for (index_t i = 0; i < NumLiquidParticle; ++i)
        {
            rho[i] = rhonBuff[i] + drho[i] * dt;
#pragma unroll 2
            for (index_t j = 0; j < dimension; ++j)
            {
                velocity[j][i] = velocityBuff[j][i] + acceleration[j][i] * dt;
                position[j][i] += velocity[j][i] * dt;
            }
        }
    };

    // -------------------- 时间步循环 --------------------
    for (index_t TimeStep = 1; TimeStep <= TotalTimeStep; ++TimeStep)
    {

        Time += dt;

        update_half_step(); // 半步更新

        Search();
        Equation();

        update_full_step(); // 全步更新

        // -------------------- 可选操作 --------------------
        // if (DensityRei > 0 && TimeStep % DensityRei == 0)
        // density_correction();

        if (TimeStep % NTsave == 0)
        {
            output_file(Time);
            printf("  Time: %f  Step: %zu\n", Time, TimeStep);
        }
    }
}

void solver::Search()
{
        std::fill(drho.begin(), drho.end(), 0.0);

#pragma omp parallel for schedule(static)
        for (auto &vec : acceleration)
            std::fill(vec.begin(), vec.end(), 0.0);


#pragma omp parallel
    {
        int tid = omp_get_thread_num();
        ThreadBuffer &buf = thread_buffers[tid];
        // 清空当前线程的 buffer
        buf.clear();

// ------------------ 构建 cell ------------------
#pragma omp for schedule(static)
        for (un Index = 0; Index < TotalParticleNum; Index++)
        {
            un cx = static_cast<un>(std::floor((position[0][Index] - Xmin) / kh));
            un cy = static_cast<un>(std::floor((position[1][Index] - Ymin) / kh));

            un count = buf.cell[MaxParticleOneCell][cx][cy];

            if (count <= MaxParticleOneCell)
            {
                buf.cell[count][cx][cy] = Index; // 储存索引

                buf.cell[MaxParticleOneCell][cx][cy]++; // 最后一位用于计数
            }
            else
            {
                std::cerr << "Particle concentration in one cell.\n";
            }
        }

// ------------------ 搜索相邻粒子 ------------------
#pragma omp for schedule(static)
        for (un Index = 0; Index < TotalParticleNum; Index++)
        {
            un cx = static_cast<un>(std::floor((position[0][Index] - Xmin) / kh));
            un cy = static_cast<un>(std::floor((position[1][Index] - Ymin) / kh));

// 遍历 3x3 区域
#pragma unroll 3
            for (int a = -1; a < 2; a++)
            {
#pragma unroll 3
                for (int b = -1; b < 2; b++)
                {
                    // 每个线程的数据都查询
                    for (auto &Allbuf : thread_buffers)
                    {
                        un NumParticleInCell = Allbuf.cell[MaxParticleOneCell][cx + a][cy + b];

                        for (size_t i = 0; i < NumParticleInCell; i++)
                        {
                            un NeighborIndex = Allbuf.cell[i][cx + a][cy + b];
                            if (Index <= NeighborIndex)
                                continue;
                            else
                            {
                                real dr;
                                real dx[dimension], dwdxCache[dimension];

                                dx[0] = position[0][Index] - position[0][NeighborIndex];
                                dx[1] = position[1][Index] - position[1][NeighborIndex];

                                dr = sqrt(dx[0] * dx[0] + dx[1] * dx[1]);
                                if (dr < (kh - 0.0002))
                                {
                                    if (buf.NumParticlePair < MaxParticlePair)
                                    {
                                        buf.pair_local[buf.NumParticlePair][0] = Index;
                                        buf.pair_local[buf.NumParticlePair][1] = NeighborIndex;

                                        kernel(dr, dx, hs, &buf.w[buf.NumParticlePair], dwdxCache);

                                        buf.dwdx_local[buf.NumParticlePair][0] = dwdxCache[0];
                                        buf.dwdx_local[buf.NumParticlePair][1] = dwdxCache[1];

                                        buf.NumParticlePair++;
                                    }

                                    else
                                        std::cerr << "Particle concentration for too many neighbors.\n";
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

inline void solver::Equation_of_State(index_t &i)
{
    pressure[i] = SoundVelocity * SoundVelocity * (rho[i] - RefRho);
}

inline void solver::gravity(index_t &i)
{
    if (i < NumLiquidParticle)
    {
        acceleration[1][i] = acceleration[1][i] + Gravity;
    }
}

inline void solver::kernel(real r, real *dx, real h, real *w, real *dwdx)
{
    real q, a;
    q = r / h;
    a = 5.0 / (14.0 * PI * h * h);
    if (1.0 <= q && q < 2.0)
    {
        *w = a * (2.0 - q) * (2.0 - q) * (2.0 - q);
        dwdx[0] = -3.0 * (*w) / (2.0 - q) / h * dx[0] / r;
        dwdx[1] = -3.0 * (*w) / (2.0 - q) / h * dx[1] / r;
    }

    else if (q < 1.0)
    {
        *w = a * (4.0 - 6.0 * q * q + 3.0 * q * q * q);
        dwdx[0] = a * (-12.0 * q + 9.0 * q * q) / h * dx[0] / r;
        dwdx[1] = a * (-12.0 * q + 9.0 * q * q) / h * dx[1] / r;
    }

    else
    {
        *w = 0;
        dwdx[0] = 0;
        dwdx[1] = 0;
    }
}

void solver::density_correction()
{

    std::fill(rhot.begin(), rhot.end(), 0.0);
    std::fill(wv.begin(), wv.end(), 0.0);

#pragma omp parallel for
    for (size_t t = 0; t < thread_buffers.size(); t++)
    {
        ThreadBuffer &buf = thread_buffers[t];

        // 遍历该线程的粒子对
        for (size_t k = 0; k < buf.NumParticlePair; k++)
        {
            un i = buf.pair_local[k][0];
            un j = buf.pair_local[k][1];

            buf.rhot[i] = buf.rhot[i] + mass[j] * w[k];
            buf.rhot[j] = buf.rhot[j] + mass[i] * w[k];
            buf.wv[i] = buf.wv[i] + buf.w[k] * mass[j] / rho[j];
            buf.wv[j] = buf.wv[j] + buf.w[k] * mass[i] / rho[i];
        }
    }

#pragma omp parallel for
    for (un i = 0; i < TotalParticleNum; i++)
    {
        for (size_t t = 0; t < thread_buffers.size(); t++)
        {
            ThreadBuffer &buf = thread_buffers[t];
            rhot[i] += buf.rhot[i];
            wv[i] += buf.wv[i];
        }

        real r, ws, rw[dimension];
        r = 0;
        memset(rw, 0, sizeof(rw));

        kernel(r, rw, hs, &ws, rw);
        rhot[i] = rhot[i] + mass[i] * ws;
        wv[i] = wv[i] + ws * mass[i] / rho[i];
        rho[i] = rhot[i] / wv[i];
    }
}

void solver::Equation()
{
#pragma omp parallel for
    for (index_t i = 0; i < TotalParticleNum; i++)
    {
        Equation_of_State(i);
        gravity(i);
    }

#pragma omp parallel for
    for (size_t t = 0; t < thread_buffers.size(); t++)
    {
        ThreadBuffer &buf = thread_buffers[t];

        // 遍历该线程的粒子对
        for (size_t k = 0; k < buf.NumParticlePair; k++)
        {
            un i = buf.pair_local[k][0];
            un j = buf.pair_local[k][1];

            continuous(i, j, buf, k);
            pressure_grad(i, j, buf, k);
            viscous_laminar(i, j, buf, k);
            viscous_artifical(i, j, buf, k);
        }
    }

    reduction();
}

inline void solver::continuous(un &i, un &j, solver::ThreadBuffer &buf, size_t &k)
{
    real vcc = 0.0;
#pragma unroll 2
    for (size_t d = 0; d < dimension; d++)
        vcc += (velocity[d][i] - velocity[d][j]) * buf.dwdx_local[k][d];

    // 更新该线程局部 drho
    buf.drho_local[i] += mass[j] * vcc;
    buf.drho_local[j] += mass[i] * vcc;
}

inline void solver::pressure_grad(un &i, un &j, solver::ThreadBuffer &buf, size_t &k)
{
    real vr[dimension];
    for (size_t d = 0; d < dimension; d++)
    {
        vr[d] = (pressure[i] / (rho[i] * rho[i]) + pressure[j] / (rho[j] * rho[j])) * buf.dwdx_local[k][d];
        buf.acceleration_local[d][i] -= mass[j] * vr[d];
        buf.acceleration_local[d][j] += mass[i] * vr[d];
    }
}

inline void solver::viscous_artifical(un &i, un &j, solver::ThreadBuffer &buf, size_t &k)
{
    real dx[dimension], vr = 0;
    for (size_t d = 0; d < dimension; d++)
        dx[d] = position[d][i] - position[d][j];

    vr = (velocity[0][i] - velocity[0][j]) * dx[0] + (velocity[1][i] - velocity[1][j]) * dx[1];
    if (vr < 0)
    {
        real rr = dx[0] * dx[0] + dx[1] * dx[1];
        real mrho = 0.5 * (rho[i] + rho[j]);
        real muv = hs * vr / (rr + eta);
        real vart = -Alpha * SoundVelocity * muv / mrho;

        for (size_t d = 0; d < dimension; d++)
        {
            dx[d] = vart * buf.dwdx_local[k][d];
            buf.acceleration_local[d][i] -= mass[j] * dx[d];
            buf.acceleration_local[d][j] += mass[i] * dx[d];
        }
    }
}

inline void solver::viscous_laminar(un &i, un &j, solver::ThreadBuffer &buf, size_t &k)
{
    real vr[dimension];
    real dx[dimension], rr = 0, md;
    for (size_t d = 0; d < dimension; d++)
    {
        dx[d] = position[d][i] - position[d][j];
        rr += dx[d] * dx[d];
    }

    md = 2.0 * Mu * (dx[0] * buf.dwdx_local[k][0] + dx[1] * buf.dwdx_local[k][1]) / (rho[i] * rho[j] * (rr + eta));

    for (size_t d = 0; d < dimension; d++)
    {
        vr[d] = md * (velocity[d][i] - velocity[d][j]);
        buf.acceleration_local[d][i] += mass[j] * vr[d];
        buf.acceleration_local[d][j] -= mass[i] * vr[d];
    }
}

void solver::reduction()
{

#pragma omp parallel for
    for (un i = 0; i < TotalParticleNum; i++)
    {
        for (size_t t = 0; t < thread_buffers.size(); t++)
        {
            ThreadBuffer &buf = thread_buffers[t];
            drho[i] += buf.drho_local[i];
#pragma unroll 2
            for (size_t d = 0; d < dimension; d++)
            {
                acceleration[d][i] += buf.acceleration_local[d][i];
            }
        }
    }
}

