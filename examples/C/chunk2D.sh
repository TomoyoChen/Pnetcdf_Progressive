#!/bin/bash
#SBATCH --job-name=chunk_2D_test
#SBATCH --account=r01156       # 替换为你的账户名
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ruihchen@iu.edu
#SBATCH --partition=debug               # 替换为合适的分区名
#SBATCH --nodes=1                     # 使用1个节点
#SBATCH --ntasks-per-node=4           # 每个节点4个任务（MPI进程）
#SBATCH --cpus-per-task=1             # 每个任务1个CPU核心
#SBATCH --time=00:30:00               # 运行时间限制30分钟
#SBATCH --output=chunk_2D_%j.out      # 标准输出文件
#SBATCH --error=chunk_2D_%j.err       # 标准错误文件

# 打印作业信息
echo "Job started at: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"

export HOME=/N/slate/ruihchen
source ~/.bashrc

# 设置库路径（根据你的安装路径调整）
export PNETCDF_ROOT=$HOME/pnetcdf_chunk/install         # 替换为实际PNetCDF安装路径
export ZLIB_ROOT=$HOME/zlib                # 替换为实际ZLIB安装路径  
export SZ_ROOT=$HOME/SZ2             # 替换为实际SZ安装路径
export ZSTD_ROOT=$HOME/zstd_install              # 替换为实际Zstandard安装路径

# 设置环境变量
export LD_LIBRARY_PATH=$PNETCDF_ROOT/lib:$ZLIB_ROOT/lib:$SZ_ROOT/lib:$ZSTD_ROOT/lib:$LD_LIBRARY_PATH
export PKG_CONFIG_PATH=$PNETCDF_ROOT/lib/pkgconfig:$PKG_CONFIG_PATH

# 编译程序
echo "Compiling chunk_2D.c..."
mpicc -O2 chunk_2D.c -o chunk_2D \
    -I$PNETCDF_ROOT/include \
    -I$ZLIB_ROOT/include \
    -I$SZ_ROOT/include \
    -I$ZSTD_ROOT/include \
    -L$PNETCDF_ROOT/lib \
    -L$ZLIB_ROOT/lib \
    -L$SZ_ROOT/lib \
    -L$ZSTD_ROOT/lib \
    -lpnetcdf -lz -lm -ldl -lSZ -lzstd

# 检查编译是否成功
if [ $? -eq 0 ]; then
    echo "Compilation successful!"
else
    echo "Compilation failed!"
    exit 1
fi

cd results

# 运行程序
echo "Running chunk_2D with $SLURM_NTASKS MPI processes..."
echo "Start time: $(date)"

# 使用srun运行MPI程序
srun -n $SLURM_NTASKS ../chunk_2D testfile_multivariable4.nc

# 检查运行是否成功
if [ $? -eq 0 ]; then
    echo "Program execution successful!"
    
    # 检查输出文件
    if [ -f "testfile_multivariable4.nc" ]; then
        echo "Output file created successfully!"
        echo "File size: $(ls -lh testfile_multivariable4.nc | awk '{print $5}')"
        
        # 如果有ncmpidump工具，显示文件内容摘要
        if command -v ncmpidump &> /dev/null; then
            echo "NetCDF file header:"
            ncmpidump -h testfile_multivariable4.nc
        fi
    else
        echo "Warning: Output file not found!"
    fi
else
    echo "Program execution failed!"
    exit 1
fi

echo "End time: $(date)"
echo "Job completed!"