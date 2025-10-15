# openMPSPH

💻 Project Name

openMPSPH

A C++ Simulation Framework with OpenMP Acceleration
Developed by Kan Liu

🧩 Overview

This project is a personal software developed by Hong Yin for scientific computing and numerical simulation.
It is implemented in C++, parallelized with OpenMP, and built using CMake for easy compilation and cross-platform deployment.

The software is designed for high-performance simulations involving complex physical processes, such as fluid–structure interaction, heat transfer, or particle-based modeling.
Its modular structure allows researchers to efficiently modify and extend the framework for customized simulation scenarios.

This is a completely demo for SPH simulation with openMP

The demo includes complete parallel in particle generation, simulation, and output. 
However, vector is a low container. I may use Eigen to improve it in the future.

⚙️ Key Features

🧮 Efficient C++ implementation for computational science

🔀 Shared-memory parallelization with OpenMP

🧱 Easy build and configuration using CMake

📈 High scalability for multi-core CPUs

🧰 Modular structure for quick extension and integration


🛠️ Build Instructions
1. Prerequisites

Make sure the following tools are installed:

C++ compiler with OpenMP support (e.g., GCC ≥ 9.0, Clang ≥ 10.0, or MSVC ≥ 2019)

CMake ≥ 3.15

(Optional) Git for cloning the repository

2. Clone the Repository
git clone https://github.com/lkbhg/openMPSPH.git
cd openMPSPH

3. Build the Project
mkdir build
cd build
cmake ..
make -j


The compiled executable will appear in the build/ or bin/ directory.

🚀 Run the Program

After building, execute the main program


Simulation results (e.g., .dat, .csv, .vtk) will be automatically stored in the /data directory.



📜 License

This project is released under the GPL License.

👤 Author

Kan Liu, Master.
Computational Fluid Dynamics | SPH | High-Performance Simulation
📧 Email: lkbhg@outlook.com

🌐 GitHub: https://github.com/lkbhg

📚 Citation

If you use this software in your research, please cite it as:

Kan Liu, Project Name: A C++ OpenMP Simulation Framework, GitHub Repository, 2025.
URL: https://github.com/lkbhg/openMPSPH

🌟 Acknowledgments

Special thanks to the open-source community for providing the foundation of modern C++ and parallel computing tools.
Also thanks Huawei to support this project from hardware and software.
This work was independently developed and maintained as part of personal research efforts.
