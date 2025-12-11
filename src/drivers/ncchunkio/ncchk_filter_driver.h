#ifndef NCCHK_FILTER_DRIVER_H
#define NCCHK_FILTER_DRIVER_H

#include <mpi.h>

/* Forward declaration to avoid circular dependencies */
struct NC_chk_var;

/* Variable context structure passed to compression functions */
typedef struct {
    double sz_abs_err_bound;
    double sz_rel_bound_ratio;
    int zlib_level;
    int varid;
    
    /* IPComp progressive compression parameters */
    int ipcomp_layers;
    int ipcomp_interp;
    int ipcomp_direction;
    int ipcomp_level_progressive;
    size_t ipcomp_block_size;
    double *ipcomp_ebs;
    int ipcomp_num_ebs;
    double ipcomp_data_range;
    double ipcomp_data_min;
    double ipcomp_data_max;
    int    ipcomp_has_minmax;
    int    ipcomp_has_fill;
    double ipcomp_fill_value;
    size_t ipcomp_header_size;
} NCCHK_var_context;

struct NCCHK_filter {
    int (*init)(MPI_Info);
    int (*finalize)();
    int (*inq_cpsize)(void*, int, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
    int (*compress)(void*, int, void*, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
    int (*compress_alloc)(void*, int, void**, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
    int (*inq_dcsize)(void*, int, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
    int (*decompress)(void*, int, void*, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
    int (*decompress_alloc)(void*, int, void**, int*, int, int*, MPI_Datatype, NCCHK_var_context*);
};

typedef struct NCCHK_filter NCCHK_filter;

extern NCCHK_filter* ncchk_dummy_inq_driver(void);

#ifndef IPCOMP_DEFAULT_BLOCK_SIZE
#define IPCOMP_DEFAULT_BLOCK_SIZE 50000
#endif

#if ENABLE_ZLIB
extern NCCHK_filter* ncchk_zlib_inq_driver(void);
#endif

#if ENABLE_SZ
extern NCCHK_filter* ncchk_sz_inq_driver(void);
#endif

#if ENABLE_IPCOMP
extern NCCHK_filter* ncchk_ipcomp_inq_driver(void);
extern int ncchk_ipcomp_decompress_progressive_error(void *in, int in_len, void *out, int *out_len, 
                                                    int ndim, int *dims, MPI_Datatype dtype, 
                                                    double target_rel_eb, NCCHK_var_context* ctx);
extern int ncchk_ipcomp_decompress_progressive_bitrate(void *in, int in_len, void *out, int *out_len, 
                                                      int ndim, int *dims, MPI_Datatype dtype, 
                                                      double max_bitrate, NCCHK_var_context* ctx);
#endif

#endif