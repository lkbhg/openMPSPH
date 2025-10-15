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




// #define  _CRT_SECURE_NO_WARNINGS
// #include <stdio.h>
// #include <math.h>
// #include<string.h>
// #include<time.h>

// const char DFolder[50] = "data/";
// #define  NSD  2             // Number of Spatial Dension
// #define  NPmax  50000       // maximum Number of Particles
// #define  NPPmax  20 * NPmax // maximum Number of Particle Pairs
// #define  NTDR  20           // Number of Timesteps for Density Reinitialization
// #define  NTprint  500       // Number of Timesteps to print info
// #define  NTsave  500        // Number of Timesteps to save data
// #define  PI  3.14159265358979
// #define  Grav  -9.8 // Gravitational accelaration
// #define  Rhor  1000 // reference density
// #define  Mu  1.0E-3 // viscosity
// #define  CAV  0.01  // alpha for artificial viscosity

// // variable

// int npr, npw, npp;
// double ds, hs, kh, cc, dt, tstart, tend, eta, maxtstep;
// double xt[NSD][NPmax+1], vt[NSD][NPmax+1], at[NSD][NPmax+1];
// double rho[NPmax+1], mass[NPmax+1], p[NPmax+1], drho[NPmax+1];
// double w[NPPmax+1], dwdx[NSD][NPPmax+1];
// int pair[2][NPPmax+1];

// void leapfrog();
// void SPH_GE();
// void output_file(double tt);
// void output_para();
// void density_correction();
// void link_list();
// void pairing(int i, int j);
// void pressure();
// void density_con();
// void body_force();
// void pres_grad();
// void visc_laminar();
// void visc_artifical();
// void kernel(double r, double *dx, double h, double *w, double *dwdx);

// int main()
// {
//     clock_t start,end;
//     double usage;
//     // 初始化
//     start=clock();
//     int np, nx, ny;
//     double wx, wy, bx, by, dx, dy;

//     tstart = 0;
//     tend = 2.0;
//     cc = 50.0;
//     wx = 1.0;
//     wy = 0.5;
//     bx = 0.3;
//     by = 0.2;
//     ds = 0.01;
//     dt = ds / (cc * 5);
//     np = 0;
//     nx = round(bx / ds);
//     ny = round(by / ds);
//     dx = bx / nx;
//     dy = by / ny;

//     for (int i = 1; i <= nx; i++)
//     {
//         for (int j = 1; j <= ny; j++)
//         {
//             np = np + 1;
//             xt[0][np] = i * dx - dx / 2;
//             xt[1][np] = j * dy - dy / 2;
//             for (int k = 0; k < NSD; k++)
//             {
//                 vt[k][np] = 0;
//             }
//             p[np] = 0;
//             rho[np] = Rhor;
//             mass[np] = rho[np] * dx * dy;
//         }
//     }
//     npr = np;
//     // particles of solid wall
//     nx = round(wx / ds);
//     ny = round(wy / ds);
//     dx = wx / nx;
//     dy = wy / ny;

//     // left

//     for (int i = -2; i <= 0; i++)
//     {
//         for (int j = 1; j <= ny; j++)
//         {
//             np = np + 1;
//             xt[0][np] = i * dx - dx / 2;
//             xt[1][np] = j * dy - dy / 2;
//             for (int k = 0; k < NSD; k++)
//             {
//                 vt[k][np] = 0;
//             }
//             p[np] = 0;
//             rho[np] = Rhor;
//             mass[np] = rho[np] * dx * dy;
//         }
//     }

//     // bottom
//     for (int i = -2; i <= nx + 3; i++)
//     {
//         for (int j = -2; j <= 0; j++)
//         {
//             np = np + 1;
//             xt[0][np] = i * dx - dx / 2;
//             xt[1][np] = j * dy - dy / 2;
//             for (int k = 0; k < NSD; k++)
//             {
//                 vt[k][np] = 0;
//             }
//             p[np] = 0;
//             rho[np] = Rhor;
//             mass[np] = rho[np] * dx * dy;
//         }
//     }

//     // right
//     for (int i = nx + 1; i <= nx + 3; i++)
//     {
//         for (int j = 1; j <= ny; j++)
//         {
//             np = np + 1;
//             xt[0][np] = i * dx - dx / 2;
//             xt[1][np] = j * dy - dy / 2;
//             for (int k = 0; k < NSD; k++)
//             {
//                 vt[k][np] = 0;
//             }
//             p[np] = 0;
//             rho[np] = Rhor;
//             mass[np] = rho[np] * dx * dy;
//         }
//     }

//     // top
//     for (int i = -2; i <= nx + 3; i++)
//     {
//         for (int j = ny + 1; j <= ny + 3; j++)
//         {
//             np = np + 1;
//             xt[0][np] = i * dx - dx / 2;
//             xt[1][np] = j * dy - dy / 2;
//             for (int k = 0; k < NSD; k++)
//             {
//                 vt[k][np] = 0;
//             }
//             p[np] = 0;
//             rho[np] = Rhor;
//             mass[np] = rho[np] * dx * dy;
//         }
//     }
//     npw = np;

//     if (tend <= tstart)
//     {
//         printf("tstart>=tend.");
//         return 0;
//     }
//     maxtstep = round((tend - tstart) / dt);

//     printf("    Fluid particle:%d\n", npr);
//     printf("    wall particle:%d\n", npw - npr);
//     printf("**************************\n");
//     hs = 1.0 * ds;
//     kh = 2.0 * hs;
//     eta = 0.01 * hs * hs;
//     output_para();
//     output_file(tstart);
//     // 计算
//     leapfrog();
//     // 输出
//     output_file(tend);
//     end=clock();
//     usage=(double)(end-start)/CLOCKS_PER_SEC;
//     printf("%f\n",usage);
//     getchar();
//     return 0;
// }

// void output_file(double tt)
// {
//     char fname[50];
//     FILE *f;
//     sprintf(fname, "t=%fs.dat", tt);
//     f = fopen(fname, "w");
//     fprintf(f, "title=\"t=\"%f s\"\n", tt);
//     fprintf(f, "variables = x, y, u, v, p, rho, mass\n");
//     fprintf(f, "zone i= %d f=point\n", npr);
//     for (int i = 1; i <= npr; i++)
//     {
//         fprintf(f, "%f   %f   %f   %f   %f   %f   %f\n", xt[0][i], xt[1][i], vt[0][i], vt[1][i], p[i], rho[i], mass[i]);
//     }

//     if (npw > npr)
//     {
//         fprintf(f, "zone i= %d f=point\n", npw - npr);
//         for (int i = npr+1; i <= npw; i++)
//         {
//             fprintf(f, "%f   %f   %f   %f   %f   %f   %f\n", xt[0][i], xt[1][i], vt[0][i], vt[1][i], p[i], rho[i], mass[i]);
//         }
//     }
//     fclose(f);
// }
// void output_para()
// {
//     char fname[50] = "2D_parameters.txt";
//     FILE *f;
//     f = fopen(fname, "w");
//     fprintf(f,"npr=%d   npw=%d\n",npr,npw);
//     fprintf(f,"hs=%e    dt=%e   cc=%e\n",hs,dt,cc);
//     fclose(f);
// }
// void leapfrog()
// {
//     int nt;
//     double tt, dt2;
//     double rhon[NPmax];
//     double vtn[NSD][NPmax];
//     memset(rhon,0,sizeof(rhon));
//     memset(vtn,0,sizeof(vtn));

//     tt = tstart;
//     dt2 = dt / 2.0;
//     SPH_GE();
//     for (int nt = 1; nt <= maxtstep; nt++)
//     {
//         tt += dt;
//         for (int i = 1; i <= npr; i++)
//         {
//             rhon[i] = rho[i];
//             rho[i] = rho[i] + drho[i] * dt2;
//             for (int j = 0; j < NSD; j++)
//             {
//                 vtn[j][i] = vt[j][i];
//                 vt[j][i] = vt[j][i] + at[j][i] * dt2;
//             }
//         }

//         SPH_GE();

//         for (int i = 1; i <= npr; i++)
//         {
//             rho[i] = rhon[i] + drho[i] * dt;
//             for (int j = 0; j < NSD; j++)
//             {
//                 vt[j][i] = vtn[j][i] + at[j][i] * dt;
//                 xt[j][i] = xt[j][i] + vt[j][i] * dt;
//             }
//         }

//         if (NTDR > 0 && nt % NTDR == 0)
//             density_correction();
//         if (nt>0 && nt % NTsave == 0)
//             output_file(tt);
//         if (nt>0 && nt % NTprint == 0)
//             printf("  Time:%f num=%d\n", tt, nt);
//     }
// }

// void SPH_GE()
// {
//     memset(drho, 0, sizeof(drho));
//     memset(at, 0, sizeof(at));

//     link_list();
//     pressure();
//     density_con();
//     body_force();
//     pres_grad();
//     visc_laminar();
//     visc_artifical();
// }
// void density_correction()
// {
//     int i, j;
//     double r, ws, rw[NSD];
//     double rhot[NPmax], wv[NPmax];

//     memset(rhot, 0, sizeof(rhot));
//     memset(wv, 0, sizeof(wv));

//     for (int k = 1; k <= npp; k++)
//     {
//         i = pair[0][k];
//         j = pair[1][k];
//         rhot[i] = rhot[i] + mass[j] * w[k];
//         rhot[j] = rhot[j] + mass[i] * w[k];
//         wv[i] = wv[i] + w[k] * mass[j] / rho[j];
//         wv[j] = wv[j] + w[k] * mass[i] / rho[i];
//     }

//     r = 0;
//     memset(rw, 0, sizeof(rw));

//     kernel(r, rw, hs, &ws, rw);

//     for (int i = 1; i <= npw; i++)
//     {
//         rhot[i] = rhot[i] + mass[i] * ws;
//         wv[i] = wv[i] + ws * mass[i] / rho[i];
//         rho[i] = rhot[i] / wv[i];
//     }
// }
// void link_list()
// {
//     int NPCmax = 20; // maximum Number of Particles in a Cell
//     int np, npn, n, i, j, ci=0, cj=0, pai=0, paj=0, ni=0, nj=0, cxmin, cxmax, cymin, cymax, err;
//     double xxmin, xxmax, yymin, yymax;

//     xxmin = xt[0][1];
//     xxmax = xt[0][1];
//     yymin = xt[1][1];
//     yymax = xt[1][1];

//     for (int i = 1; i <= npw; i++)
//     {
//         if (xt[0][i] < xxmin)
//         {
//             xxmin = xt[0][i];
//         }
//         if (xt[0][i] > xxmax)
//         {
//             xxmax = xt[0][i];
//         }
//         if (xt[1][i] < yymin)
//         {
//             yymin = xt[1][i];
//         }
//         if (xt[1][i] > yymax)
//         {
//             yymax = xt[1][i];
//         }
//     }
//     /*
//         xxmin = min(xt(1, 1 : npw));
//         xxmax = max(xt(1, 1 : npw));
//         yymin = min(xt(2, 1 : npw));
//         yymax = max(xt(2, 1 : npw));
//     */

//     cxmin = ceil(xxmin / kh);
//     cxmax = ceil(xxmax / kh);
//     cymin = ceil(yymin / kh);
//     cymax = ceil(yymax / kh);

//     int n1 = NPCmax + 1;
//     int n2 = cxmax - cxmin + 3;
//     int n3 = cymax - cymin + 3;
//     int n2_s = cxmin - 1;
//     int n3_s = cymin - 1;

//     //int ***cell = (int ***)malloc(n1 * sizeof(int **) +        /* level1 pointer */
//     //                              n1 * n2 * sizeof(int *) +    /* level2 pointer */
//     //                              n1 * n2 * n3 * sizeof(int)); /* data pointer */
//     //for (int i = 0; i < n1; ++i)
//     //{
//     //    cell[i] = (int **)(cell + n1) + i * n2;
//     //    for (int j = 0; j < n2; ++j)
//     //        cell[i][j] = (int *)(cell + n1 + n1 * n2) + i * n2 * n3 + j * n3;
//     //}

//     // int cell[NPCmax][cxmin-1:cxmax+1, cymin-1:cymax+1];
//     int cell[n1][n2][n3];
//     memset(cell,0,sizeof(cell));
//     //int cell[NPCmax][cxmax - cxmin + 2][cymin - cymax + 2];
//     //memset(cell, 0, sizeof(n1 * sizeof(int **) + n1 * n2 * sizeof(int *)+n1 * n2 * n3 * sizeof(int)));
//     npp = 0;
//     npn = npw;

//     for (np = 1; np <= npw; np++)
//     {
//         i = ceil(xt[0][np] / kh);
//         j = ceil(xt[1][np] / kh);
//         n = cell[0][i - n2_s][j - n3_s] + 1;
//         if (n > NPCmax)
//         {
//             printf(" Too many particles in a cell.\n");
//         }

//         else
//         {
//             cell[0][i - n2_s][j - n3_s] = n;
//             cell[n][i - n2_s][j - n3_s] = np;
//         }
//     }
//     for (cj = cymin; cj <= cymax; cj++)
//     {
//         for (ci = cxmin; ci <= cxmax; ci++)
//         {
//             n = cell[0][ci - n2_s][cj - n3_s];
//             for (ni = 1; ni <= n; ni++)
//             {
//                 pai = cell[ni][ci - n2_s][cj - n3_s];
//                 for (nj = ni + 1; nj <= n; nj++)
//                 {
//                     paj = cell[nj][ci - n2_s][cj - n3_s];
//                     if (pai <= npn || paj <= npn)
//                         pairing(pai, paj);
//                 }
//                 i = ci + 1;
//                 for (np = 1; np <= cell[0][i - n2_s][cj - n3_s]; np++)
//                 {
//                     paj = cell[np][i - n2_s][cj - n3_s];
//                     if (pai <= npn || paj <= npn)
//                         pairing(pai, paj);
//                 }
//                 j = cj + 1;
//                 for (i = (ci - 1); i <= (ci + 1); i++)
//                 {
//                     for (np = 1; np <= cell[0][i - n2_s][j - n3_s]; np++)
//                     {
//                         paj = cell[np][i - n2_s][j - n3_s];
//                         if (pai <= npn || paj <= npn)
//                             pairing(pai, paj);
//                     }
//                 }
//             }
//         }
//     }

//     //free(cell);
// }
// void pairing(int i, int j)
// {
//     double dr;
//     double dx[NSD], tdwdx[NSD];

//     dx[0] = xt[0][i] - xt[0][j];
//     dx[1] = xt[1][i] - xt[1][j];

//     dr = sqrt(dx[0] * dx[0] + dx[1] * dx[1]);
//     //printf("%e\n", dr);
//     if (dr < (kh-0.0002))
//     {
//         if (npp < NPPmax)
//         {
//             npp = npp + 1;
//             //printf("%d\n", npp);
//             pair[0][npp] = i;
//             pair[1][npp] = j;
//             kernel(dr, dx, hs, &w[npp], tdwdx);
//             dwdx[0][npp] = tdwdx[0];
//             dwdx[1][npp] = tdwdx[1];
//         }

//         else
//             printf(" Too many particle pairs.");
        
//     }
// }

// void pressure()
// {
//     for (int i = 1; i <= npw; i++)
//     {
//         p[i] = cc * cc * (rho[i] - Rhor);
//     }
// }
// void density_con()
// {
//     int i, j;
//     double vcc = 0;
//     for (int k = 1; k <= npp; k++)
//     {
//         i = pair[0][k];
//         j = pair[1][k];
//         vcc = ((vt[0][i] - vt[0][j]) * dwdx[0][k] + (vt[1][i] - vt[1][j]) * dwdx[1][k]);
//         drho[i] += mass[j] * vcc;
//         drho[j] += mass[i] * vcc;
//     }
// }
// void body_force()
// {
//     for (int i = 1; i <= npr; i++)
//     {
//         at[1][i] = at[1][i] + Grav;
//     }
// }
// void pres_grad()
// {
//     int i, j;
//     double vr[NSD];

//     for (int k = 1; k <= npp; k++)
//     {
//         i = pair[0][k];
//         j = pair[1][k];
//         vr[0] = (p[i] / (rho[i] * rho[i]) + p[j] / (rho[j] * rho[j])) * dwdx[0][k];
//         at[0][i] = at[0][i] - mass[j] * vr[0];
//         at[0][j] = at[0][j] + mass[i] * vr[0];

//         vr[1] = (p[i] / (rho[i] * rho[i]) + p[j] / (rho[j] * rho[j])) * dwdx[1][k];
//         at[1][i] = at[1][i] - mass[j] * vr[1];
//         at[1][j] = at[1][j] + mass[i] * vr[1];
//     }
// }
// void visc_laminar()
// {
//     int i, j;
//     double rr, r, md, vr[NSD];
//     for (int k = 1; k <= npp; k++)
//     {
//         i = pair[0][k];
//         j = pair[1][k];

//         vr[0] = xt[0][i] - xt[0][j];
//         vr[1] = xt[1][i] - xt[1][j];

//         rr = vr[0] * vr[0] + vr[1] * vr[1];
//         md = 2.0 * Mu * (vr[0] * dwdx[0][k] + vr[1] * dwdx[1][k]) / (rho[i] * rho[j] * (rr + eta));

//         vr[0] = md * (vt[0][i] - vt[0][j]);
//         at[0][i] = at[0][i] + mass[j] * vr[0];
//         at[0][j] = at[0][j] - mass[i] * vr[0];

//         vr[1] = md * (vt[1][i] - vt[1][j]);
//         at[1][i] = at[1][i] + mass[j] * vr[1];
//         at[1][j] = at[1][j] - mass[i] * vr[1];
//     }
// }
// void visc_artifical()
// {
//     int i, j;
//     double vr, rr, muv, mrho, vart, dx[NSD];

//     for (int k = 1; k <= npp; k++)
//     {
//         i = pair[0][k];
//         j = pair[1][k];

//         dx[0] = xt[0][i] - xt[0][j];
//         dx[1] = xt[1][i] - xt[1][j];
//         vr = (vt[0][i] - vt[0][j]) * dx[0] + (vt[1][i] - vt[1][j]) * dx[1];

//         if (vr < 0)
//         {
//             rr = dx[0] * dx[0] + dx[1] * dx[1];
//             mrho = (rho[i] + rho[j]) / 2.0;
//             muv = hs * vr / (rr + eta);
//             vart = -CAV * cc * muv / mrho;

//             dx[0] = vart * dwdx[0][k];
//             at[0][i] = at[0][i] - mass[j] * dx[0];
//             at[0][j] = at[0][j] + mass[i] * dx[0];

//             dx[1] = vart * dwdx[1][k];
//             at[1][i] = at[1][i] - mass[j] * dx[1];
//             at[1][j] = at[1][j] + mass[i] * dx[1];
//         }
//     }
// }
// void kernel(double r, double *dx, double h, double *w, double *dwdx)
// {
//     double q, a;
//     q = r / h;
//     a = 5.0 / (14.0 * PI * h * h);
//     if (1.0 <= q && q < 2.0)
//     {
//         *w = a * (2.0 - q) * (2.0 - q) * (2.0 - q);
//         dwdx[0] = -3.0 * (*w) / (2.0 - q) / h * dx[0] / r;
//         dwdx[1] = -3.0 * (*w) / (2.0 - q) / h * dx[1] / r;
//     }

//     else if (q < 1.0)
//     {
//         *w = a * (4.0 - 6.0 * q * q + 3.0 * q * q * q);
//         dwdx[0] = a * (-12.0 * q + 9.0 * q * q) / h * dx[0] / r;
//         dwdx[1] = a * (-12.0 * q + 9.0 * q * q) / h * dx[1] / r;
//     }

//     else
//     {
//         *w = 0;
//         dwdx[0] = 0;
//         dwdx[1] = 0;
//     }
// }