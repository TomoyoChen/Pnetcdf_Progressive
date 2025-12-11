/*
 * 增强版SZ过滤器：支持变量级别的压缩参数配置
 * 
 * 需要修改以下文件以支持变量级别的SZ参数：
 * 1. ncchk_filter_sz.c - 主要修改文件
 * 2. ncchk_filter_driver.h - 可能需要扩展接口
 * 3. ncchkioi_var_init.c - 读取变量级别的SZ参数
 */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <mpi.h>
#include <pnc_debug.h>
#include <ncchkio_driver.h>
#include <ncchk_filter_driver.h>
#include <common.h>
#include <sz.h>

/* 变量级别的SZ参数结构 */
typedef struct {
    int error_bound_mode;   // ABS or REL
    double abs_error_bound; // 绝对误差界限
    double rel_error_bound; // 相对误差界限
    int sz_mode;           // SZ压缩模式
    int sample_distance;   // 采样距离
} variable_sz_params_t;

/* 从变量属性获取SZ参数 */
static int get_variable_sz_params(void* ncp, int varid, variable_sz_params_t* params, 
                                  PNC_driver* driver) {
    int err = NC_NOERR;
    MPI_Offset len;
    
    // 设置默认值
    params->error_bound_mode = ABS;
    params->abs_error_bound = 1E-3;
    params->rel_error_bound = 1E-5;
    params->sz_mode = SZ_BEST_COMPRESSION;
    params->sample_distance = 50;
    
    // 读取变量特定的SZ参数
    if (driver->inq_att(ncp, varid, "_sz_error_mode", NULL, &len) == NC_NOERR && len == 1) {
        driver->get_att(ncp, varid, "_sz_error_mode", &(params->error_bound_mode), MPI_INT);
    }
    
    if (driver->inq_att(ncp, varid, "_sz_abs_error", NULL, &len) == NC_NOERR && len == 1) {
        driver->get_att(ncp, varid, "_sz_abs_error", &(params->abs_error_bound), MPI_DOUBLE);
    }
    
    if (driver->inq_att(ncp, varid, "_sz_rel_error", NULL, &len) == NC_NOERR && len == 1) {
        driver->get_att(ncp, varid, "_sz_rel_error", &(params->rel_error_bound), MPI_DOUBLE);
    }
    
    if (driver->inq_att(ncp, varid, "_sz_mode", NULL, &len) == NC_NOERR && len == 1) {
        driver->get_att(ncp, varid, "_sz_mode", &(params->sz_mode), MPI_INT);
    }
    
    if (driver->inq_att(ncp, varid, "_sz_sample_dist", NULL, &len) == NC_NOERR && len == 1) {
        driver->get_att(ncp, varid, "_sz_sample_dist", &(params->sample_distance), MPI_INT);
    }
    
    return err;
}

/* 
 * 修改后的compress函数，支持传递变量信息
 * 需要扩展NCCHK_filter接口来传递变量ID和驱动器指针
 */
int ncchk_sz_compress_enhanced(void *in, int in_len, void *out, int *out_len, 
                              int ndim, int *dims, MPI_Datatype dtype,
                              void* ncp, int varid, PNC_driver* driver) {
    int err = NC_NOERR;
    variable_sz_params_t params;
    sz_params sz;
    
    // 获取变量特定的SZ参数
    get_variable_sz_params(ncp, varid, &params, driver);
    
    // 设置SZ参数
    memset(&sz, 0, sizeof(sz_params));
    sz.sol_ID = SZ;
    sz.sampleDistance = params.sample_distance;
    sz.quantization_intervals = 0;
    sz.max_quant_intervals = 65536;
    sz.predThreshold = 0.98;
    sz.szMode = params.sz_mode;
    sz.losslessCompressor = ZSTD_COMPRESSOR;
    sz.gzipMode = 1;
    sz.errorBoundMode = params.error_bound_mode;
    sz.absErrBound = params.abs_error_bound;
    sz.relBoundRatio = params.rel_error_bound;
    
    // 临时重新初始化SZ（注意：这可能影响性能）
    SZ_Init_Params(&sz);
    
    // 执行压缩（使用原有的压缩逻辑）
    int i;
    int szdtype;
    size_t r[4];
    size_t outsize;
    void *buf = NULL;

    szdtype = mpi_to_sz_type(dtype);  // 需要暴露这个函数
    if (szdtype < 0){
        DEBUG_ASSIGN_ERROR(err, NC_EINVAL)
        goto out;
    }

    for(i = 0; i < 4; i++){
        if (i < ndim){
            r[i] = dims[i];
        }
        else{
            r[i] = 0;
        }
    }
    for(i = 4; i < ndim; i++){
        r[3] *= dims[i];
    }

    buf = SZ_compress(szdtype, in, &outsize, 0, r[3], r[2], r[1], r[0]);
    
    if (out_len != NULL){
        if (*out_len < outsize){
            DEBUG_ASSIGN_ERROR(err, NC_ENOMEM)
            goto out;
        }
        *out_len = outsize;
    }

    memcpy(out, buf, outsize);

out:
    if (buf != NULL){
        free(buf);
    }
    
    return err;
}

/*
 * 为了演示目的的简化实现
 * 实际实现需要修改NCCHK_filter接口来传递更多参数
 */ 