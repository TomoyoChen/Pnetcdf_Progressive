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
    size_t ipcomp_interp_dim_limit; /* Interpolation dimension limit (even), for IPComp/SZ3 */
    double *ipcomp_ebs;
    int ipcomp_num_ebs;
    double ipcomp_data_range;
    double ipcomp_data_min;
    double ipcomp_data_max;
    int    ipcomp_has_minmax;
    int    ipcomp_has_fill;
    double ipcomp_fill_value;
    size_t ipcomp_header_size;
    /* Shared mask/cache for IPComp */
    unsigned char *ipcomp_mask_bits;      /* valid mask bitset */
    unsigned char *ipcomp_boundary_bits;  /* guard-band bitset */
    size_t         ipcomp_mask_bytes;
    size_t         ipcomp_mask_valid_count;
    int            ipcomp_guard_radius;
    uint32_t       ipcomp_mask_crc32;
    int            ipcomp_mask_version;
    int            ipcomp_mask_shared;    /* 1 if we reuse across time steps */
    int            ipcomp_mask_ready;     /* 1 if cache filled */
    int            ipcomp_mask_written;   /* 1 if already embedded once */
    /* Attribute access callbacks for mask sidecar */
    /* Use driver signature: put_att(ncp, varid, name, nc_type, len, buf, MPI_Datatype) */
    int (*ipcomp_put_att)(void *ncp, int varid, const char *name,
                          nc_type xtype, MPI_Offset len, const void *buf, MPI_Datatype buftype);
    int (*ipcomp_get_att)(void *ncp, int varid, const char *name,
                          void *buf, MPI_Datatype buftype);
    void *ipcomp_ncp;   /* backend file handle */
    int   ipcomp_varid; /* variable id */
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

#ifndef IPCOMP_DEFAULT_INTERP_DIM_LIMIT
/* Must be even. Used by IPComp/SZ3 to avoid extrapolation. */
#define IPCOMP_DEFAULT_INTERP_DIM_LIMIT 50000
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