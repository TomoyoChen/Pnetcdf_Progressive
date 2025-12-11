/*
 *  Copyright (C) 2025, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */

 #ifdef HAVE_CONFIG_H
 # include <config.h>
 #endif
 
 #include <stdlib.h>
 #include <string.h>
 #include <stdio.h>
 #include <assert.h>
 
 #include <mpi.h>
 #include <pnetcdf.h>
 #include <pnc_debug.h>
 #include <common.h>
 #include <dispatch.h>
 #include <limits.h>
 #include <stdint.h>
#include "../drivers/ncmpio/ncmpio_NC.h"

typedef struct {
    size_t prelude_size;
    size_t dims_offset;
    size_t header_bytes_offset;
    size_t index_bytes_offset;
    size_t payload_offset_offset;
    size_t range_offset;
    size_t total_size;
} HeaderV1Layout;

static void compute_header_v1_layout(int logical_ndim, HeaderV1Layout *layout) {
    const size_t prelude_size = 4u + 7u * sizeof(uint32_t);
    size_t offset = 0;

    layout->prelude_size = prelude_size;
    offset += prelude_size;

    layout->dims_offset = offset;
    offset += (size_t)logical_ndim * sizeof(uint64_t);

    layout->header_bytes_offset = offset;
    offset += sizeof(uint64_t);

    layout->index_bytes_offset = offset;
    offset += sizeof(uint64_t);

    layout->payload_offset_offset = offset;
    offset += sizeof(uint64_t);

    layout->range_offset = offset;
    offset += sizeof(uint64_t); /* store double as 8 bytes */

    layout->total_size = offset;
}

static void pack_u32_le(uint8_t *dst, uint32_t value) {
    dst[0] = (uint8_t)(value & 0xFFu);
    dst[1] = (uint8_t)((value >> 8) & 0xFFu);
    dst[2] = (uint8_t)((value >> 16) & 0xFFu);
    dst[3] = (uint8_t)((value >> 24) & 0xFFu);
}

static void pack_u64_le(uint8_t *dst, uint64_t value) {
    for (int i = 0; i < 8; i++) {
        dst[i] = (uint8_t)((value >> (8 * i)) & 0xFFu);
    }
}

static uint32_t unpack_u32_le(const uint8_t *src) {
    uint32_t value = 0;
    for (int i = 3; i >= 0; i--) {
        value <<= 8;
        value |= (uint32_t)src[i];
    }
    return value;
}

static uint64_t unpack_u64_le(const uint8_t *src) {
    uint64_t value = 0;
    for (int i = 7; i >= 0; i--) {
        value <<= 8;
        value |= (uint64_t)src[i];
    }
    return value;
}

static double unpack_double_le(const uint8_t *src) {
    uint64_t bits = unpack_u64_le(src);
    double value = 0.0;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static void pack_double_le(uint8_t *dst, double value) {
    uint64_t bits = 0;
    memcpy(&bits, &value, sizeof(bits));
    pack_u64_le(dst, bits);
}

static void copy_roi_to_buffer(const void *src, void *dst,
                               int ndim,
                               const MPI_Offset *start,
                               const MPI_Offset *count,
                               const int *full_dims,
                               size_t elem_size) {
    if (ndim <= 0) {
        return;
    }

    const char *src_bytes = (const char *)src;
    char *dst_bytes = (char *)dst;

    if (ndim == 1) {
        size_t inner = (size_t)count[0];
        if (inner == 0) {
            return;
        }
        size_t src_offset = (size_t)start[0];
        memcpy(dst_bytes, src_bytes + src_offset * elem_size,
               inner * elem_size);
        return;
    }

    size_t inner = (size_t)count[ndim - 1];
    if (inner == 0) {
        return;
    }

    size_t dst_strides[NC_MAX_VAR_DIMS];
    size_t full_strides[NC_MAX_VAR_DIMS];

    dst_strides[ndim - 1] = 1;
    full_strides[ndim - 1] = 1;
    for (int d = ndim - 2; d >= 0; --d) {
        dst_strides[d] = dst_strides[d + 1] * (size_t)count[d + 1];
        full_strides[d] = full_strides[d + 1] * (size_t)full_dims[d + 1];
    }

    size_t total_outer = 1;
    for (int d = 0; d < ndim - 1; ++d) {
        total_outer *= (size_t)count[d];
    }
    if (total_outer == 0) {
        return;
    }

    size_t coords[NC_MAX_VAR_DIMS];
    if (ndim > 1) {
        memset(coords, 0, sizeof(size_t) * (size_t)(ndim - 1));
    }

    for (size_t outer_idx = 0; outer_idx < total_outer; ++outer_idx) {
        size_t src_index = 0;
        size_t dst_index = 0;
        for (int d = 0; d < ndim - 1; ++d) {
            size_t coord = coords[d];
            src_index += ((size_t)start[d] + coord) * full_strides[d];
            dst_index += coord * dst_strides[d];
        }

        size_t src_offset_elems = src_index + (size_t)start[ndim - 1];
        size_t dst_offset_elems = dst_index * inner;

        memcpy(dst_bytes + dst_offset_elems * elem_size,
               src_bytes + src_offset_elems * elem_size,
               inner * elem_size);

        for (int d = ndim - 2; d >= 0; --d) {
            coords[d]++;
            if (coords[d] < (size_t)count[d]) {
                break;
            }
            coords[d] = 0;
        }
    }
}

static int header_write_bytes(int ncid, int varid, size_t offset, const uint8_t *bytes, size_t len) {
    MPI_Offset start = (MPI_Offset)offset;
    MPI_Offset count = (MPI_Offset)len;
    return ncmpi_put_vara_all(ncid, varid, &start, &count, (void*)bytes, count, MPI_UNSIGNED_CHAR);
}

static int header_read_bytes(int ncid, int varid, size_t offset, uint8_t *bytes, size_t len) {
    MPI_Offset start = (MPI_Offset)offset;
    MPI_Offset count = (MPI_Offset)len;
    return ncmpi_get_vara_all(ncid, varid, &start, &count, bytes, count, MPI_UNSIGNED_CHAR);
}

static int header_write_u64(int ncid, int varid, size_t offset, uint64_t value) {
    uint8_t buf[8];
    pack_u64_le(buf, value);
    return header_write_bytes(ncid, varid, offset, buf, sizeof(buf));
}

static int header_read_u64(int ncid, int varid, size_t offset, uint64_t *value) {
    uint8_t buf[8];
    int err = header_read_bytes(ncid, varid, offset, buf, sizeof(buf));
    if (err != NC_NOERR) return err;
    *value = unpack_u64_le(buf);
    return NC_NOERR;
}

static int header_read_double(int ncid, int varid, size_t offset, double *value) {
    uint8_t buf[8];
    int err = header_read_bytes(ncid, varid, offset, buf, sizeof(buf));
    if (err != NC_NOERR) return err;
    *value = unpack_double_le(buf);
    return NC_NOERR;
}

static int header_write_double(int ncid, int varid, size_t offset, double value) {
    uint8_t buf[8];
    pack_double_le(buf, value);
    return header_write_bytes(ncid, varid, offset, buf, sizeof(buf));
}
 
#if defined(MPI_UNSIGNED_LONG_LONG)
#  define MPI_U64 MPI_UNSIGNED_LONG_LONG
#elif defined(MPI_UINT64_T)
#  define MPI_U64 MPI_UINT64_T
#else
#  error "Required MPI 64-bit unsigned datatype not available"
#endif
 
 #ifdef ENABLE_IPCOMP
 #include <ncchk_filter_driver.h>
 /* External function declarations */
 extern NCCHK_filter* ncchk_ipcomp_inq_driver(void);
 #endif
 
/* Header V1 structure for progressive container (as per design doc §5) */
typedef struct {
    uint8_t  magic[4];         /* "PCF1" */
    uint32_t version;          /* 1 */
    uint32_t logical_type;     /* nc_type (NC_FLOAT, NC_DOUBLE, etc.) */
    uint32_t logical_ndim;     /* rank of logical array */
    uint32_t layers;           /* compression layers */
    uint32_t level_progressive;/* progressive levels */
    uint32_t group_bits;       /* bit grouping parameter */
    uint32_t reserved;         /* alignment/extension */
    /* followed by logical_ndim x uint64_t for dimension sizes */
} HeaderV1Prelude;

/* Build header v1 binary from container attributes (design doc §5) */
static int build_header_v1(int ncid, int varid,
                           uint8_t **header_out, size_t *header_size_out) {
    int err = NC_NOERR;
    int logical_type_int = 0;
    int logical_ndim = 0;
    long long *logical_dims_ll = NULL;
    int layers = 0;
    int level_progressive = 0;
    int group_bits = 0;

    err = ncmpi_get_att_int(ncid, varid, "comp:logical_type", &logical_type_int);
    if (err != NC_NOERR) return err;

    err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim);
    if (err != NC_NOERR) return err;

    if (logical_ndim < 0 || logical_ndim > NC_MAX_VAR_DIMS)
        DEBUG_RETURN_ERROR(NC_EINVAL)

    if (logical_ndim > 0) {
        logical_dims_ll = (long long*)malloc((size_t)logical_ndim * sizeof(long long));
        if (logical_dims_ll == NULL) DEBUG_RETURN_ERROR(NC_ENOMEM)

        err = ncmpi_get_att_longlong(ncid, varid, "comp:logical_dims",
                                     logical_dims_ll);
        if (err != NC_NOERR) {
            free(logical_dims_ll);
            return err;
        }
    }

    err = ncmpi_get_att_int(ncid, varid, "comp:layers", &layers);
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_get_att_int(ncid, varid, "comp:level_progressive", &level_progressive);
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_get_att_int(ncid, varid, "comp:group_bits", &group_bits);
    if (err != NC_NOERR) goto cleanup;

    HeaderV1Layout layout;
    compute_header_v1_layout(logical_ndim, &layout);

    uint8_t *header = (uint8_t*)malloc(layout.total_size);
    if (header == NULL) {
        err = NC_ENOMEM;
        goto cleanup;
    }
    memset(header, 0, layout.total_size);

    header[0] = 'P';
    header[1] = 'C';
    header[2] = 'F';
    header[3] = '1';

    pack_u32_le(header + 4, 1u);
    pack_u32_le(header + 8, (uint32_t)logical_type_int);
    pack_u32_le(header + 12, (uint32_t)logical_ndim);
    pack_u32_le(header + 16, (uint32_t)layers);
    pack_u32_le(header + 20, (uint32_t)level_progressive);
    pack_u32_le(header + 24, (uint32_t)group_bits);
    pack_u32_le(header + 28, 0u); /* reserved */

    uint8_t *dims64 = header + layout.dims_offset;
    for (int i = 0; i < logical_ndim; i++)
        pack_u64_le(dims64 + (size_t)i * sizeof(uint64_t), (uint64_t)logical_dims_ll[i]);

    pack_u64_le(header + layout.header_bytes_offset, (uint64_t)layout.total_size);
    pack_u64_le(header + layout.index_bytes_offset, 0u);
    pack_u64_le(header + layout.payload_offset_offset, (uint64_t)layout.total_size);
    pack_double_le(header + layout.range_offset, 0.0);

    free(logical_dims_ll);
    *header_out = header;
    *header_size_out = layout.total_size;
    return NC_NOERR;

cleanup:
    if (logical_dims_ll) free(logical_dims_ll);
    return err;
}

static int read_progressive_frames_error_impl(int ncid, int varid, void *buf,
                                      const MPI_Offset *start,
                                      const MPI_Offset *count, MPI_Datatype buftype, int ndim,
                                      const double *target_rel_ebs, int num_target_ebs,
                                      double fallback_target_rel_eb);

static int read_progressive_frames_error(int ncid, int varid, void *buf,
                                      const MPI_Offset *start,
                                      const MPI_Offset *count, MPI_Datatype buftype, int ndim,
                                      double target_rel_eb) {
    const double *list = NULL;
    int list_count = 0;
    if (target_rel_eb > 0.0) {
        list = &target_rel_eb;
        list_count = 1;
    }
    return read_progressive_frames_error_impl(ncid, varid, buf, start, count, buftype, ndim,
                                              list, list_count, target_rel_eb);
}

/* Convert an existing variable into a progressive container (design doc §4/§6) */
static int convert_progressive_container(int ncid, int varid,
                                         int layers, int level_progressive,
                                         int group_bits) {
    int err = NC_NOERR;
    PNC *pncp = NULL;
    NC *ncp = NULL;
    NC_var *varp = NULL;
    PNC_var *pvar = NULL;
    char var_name[NC_MAX_NAME];
    char dim_name[NC_MAX_NAME + 32];
    nc_type logical_type;
    int logical_ndim = 0;
    int *logical_dimids = NULL;
    MPI_Offset *logical_dims = NULL;
    long long *logical_dims_ll = NULL;
    int container_dimid = -1;
    int old_recdim = -1;
    int was_record = 0;
    int logical_type_int;
    MPI_Offset *pnc_new_shape = NULL;
    int *new_dimids = NULL;
    MPI_Offset *new_shape = NULL;
    MPI_Offset *new_dsizes = NULL;

    if (group_bits <= 0)
        DEBUG_RETURN_ERROR(NC_EINVAL)

    err = PNC_check_id(ncid, &pncp);
    if (err != NC_NOERR) return err;

    if (pncp == NULL || pncp->ncp == NULL)
        DEBUG_RETURN_ERROR(NC_ENOTNC)

    if (varid < 0 || varid >= pncp->nvars)
        DEBUG_RETURN_ERROR(NC_ENOTVAR)

    ncp = (NC*)pncp->ncp;
    if (varid >= ncp->vars.ndefined)
        DEBUG_RETURN_ERROR(NC_ENOTVAR)

    varp = ncp->vars.value[varid];
    if (varp == NULL)
        DEBUG_RETURN_ERROR(NC_ENOTVAR)

    pvar = &pncp->vars[varid];

    err = ncmpi_inq_varname(ncid, varid, var_name);
    if (err != NC_NOERR) return err;

    err = ncmpi_inq_vartype(ncid, varid, &logical_type);
    if (err != NC_NOERR) return err;

    err = ncmpi_inq_varndims(ncid, varid, &logical_ndim);
    if (err != NC_NOERR) return err;

    if (logical_ndim < 0 || logical_ndim > NC_MAX_VAR_DIMS)
        DEBUG_RETURN_ERROR(NC_EINVAL)

    if (logical_ndim > 0) {
        logical_dimids = (int*)malloc((size_t)logical_ndim * sizeof(int));
        logical_dims = (MPI_Offset*)malloc((size_t)logical_ndim * sizeof(MPI_Offset));
        logical_dims_ll = (long long*)malloc((size_t)logical_ndim * sizeof(long long));
        if (logical_dimids == NULL || logical_dims == NULL || logical_dims_ll == NULL) {
            err = NC_ENOMEM;
            goto cleanup;
        }

        err = ncmpi_inq_vardimid(ncid, varid, logical_dimids);
        if (err != NC_NOERR) goto cleanup;

        for (int i = 0; i < logical_ndim; i++) {
            err = ncmpi_inq_dimlen(ncid, logical_dimids[i], &logical_dims[i]);
            if (err != NC_NOERR) goto cleanup;
            logical_dims_ll[i] = (long long)logical_dims[i];
        }
    }

    /* Define or reuse the byte-stream unlimited dimension */
    snprintf(dim_name, sizeof(dim_name), "__pc_bytes_%s", var_name);
    if ((int)strlen(dim_name) >= NC_MAX_NAME)
        snprintf(dim_name, sizeof(dim_name), "__pc_bytes");

    err = ncmpi_def_dim(ncid, dim_name, NC_UNLIMITED, &container_dimid);
    if (err == NC_EUNLIMIT || err == NC_ENAMEINUSE) {
        /* Unlimited dimension already exists, reuse it */
        if (pncp->unlimdimid >= 0)
            container_dimid = pncp->unlimdimid;
        if (err == NC_ENAMEINUSE)
            err = ncmpi_inq_dimid(ncid, dim_name, &container_dimid);
        else
            err = NC_NOERR;
    }
    if (err != NC_NOERR) goto cleanup;

    /* Prepare new dimension arrays for the NC_var structure */
    new_dimids = (int*)NCI_Malloc(sizeof(int));
    new_shape  = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset));
    new_dsizes = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset));
    if (new_dimids == NULL || new_shape == NULL || new_dsizes == NULL) {
        err = NC_ENOMEM;
        goto cleanup;
    }

    new_dimids[0] = container_dimid;
    new_shape[0]  = NC_UNLIMITED;
    new_dsizes[0] = 1;

    old_recdim = pvar->recdim;
    was_record = (old_recdim >= 0);

    if (varp->dimids) NCI_Free(varp->dimids);
    if (varp->shape)  NCI_Free(varp->shape);
    if (varp->dsizes) NCI_Free(varp->dsizes);

    varp->dimids = new_dimids; new_dimids = NULL;
    varp->shape  = new_shape;  new_shape = NULL;
    varp->dsizes = new_dsizes; new_dsizes = NULL;
    varp->ndims  = 1;
    varp->xtype  = NC_UBYTE;
    varp->xsz    = 1;

    err = ncmpio_NC_var_shape64(varp, &ncp->dims);
    if (err != NC_NOERR) goto cleanup;

    if (pvar->shape) NCI_Free(pvar->shape);
    pnc_new_shape = (MPI_Offset*)NCI_Malloc(sizeof(MPI_Offset));
    if (pnc_new_shape == NULL) {
        err = NC_ENOMEM;
        goto cleanup;
    }
    pvar->shape = pnc_new_shape;
    pvar->shape[0] = 0;
    pvar->ndims = 1;
    pvar->xtype = NC_UBYTE;
    pvar->recdim = container_dimid;
    pnc_new_shape = NULL;

    if (!was_record) {
        pncp->nrec_vars++;
        ncp->vars.num_rec_vars++;
    }

    /* Write container metadata attributes */
    logical_type_int = (int)logical_type;

    err = ncmpi_put_att_text(ncid, varid, "comp:container", 5, "pc-v1");
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_put_att_int(ncid, varid, "comp:logical_type", NC_INT, 1, &logical_type_int);
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_put_att_int(ncid, varid, "comp:logical_ndim", NC_INT, 1, &logical_ndim);
    if (err != NC_NOERR) goto cleanup;

    if (logical_ndim > 0) {
        err = ncmpi_put_att_longlong(ncid, varid, "comp:logical_dims", NC_INT64,
                                     logical_ndim, logical_dims_ll);
        if (err != NC_NOERR) goto cleanup;
    }

    err = ncmpi_put_att_int(ncid, varid, "comp:layers", NC_INT, 1, &layers);
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_put_att_int(ncid, varid, "comp:level_progressive", NC_INT, 1, &level_progressive);
    if (err != NC_NOERR) goto cleanup;

    err = ncmpi_put_att_int(ncid, varid, "comp:group_bits", NC_INT, 1, &group_bits);
    if (err != NC_NOERR) goto cleanup;

    HeaderV1Layout header_layout;
    compute_header_v1_layout(logical_ndim, &header_layout);
    long long header_bytes_ll = (long long)header_layout.total_size;
    err = ncmpi_put_att_longlong(ncid, varid, "comp:header_bytes", NC_INT64, 1, &header_bytes_ll);
    if (err != NC_NOERR) goto cleanup;

    long long payload_offset_ll = (long long)header_layout.total_size;
    err = ncmpi_put_att_longlong(ncid, varid, "comp:payload_offset", NC_INT64, 1, &payload_offset_ll);
    if (err != NC_NOERR) goto cleanup;

cleanup:
    if (logical_dimids) free(logical_dimids);
    if (logical_dims)   free(logical_dims);
    if (logical_dims_ll) free(logical_dims_ll);
    if (new_dimids) NCI_Free(new_dimids);
    if (new_shape)  NCI_Free(new_shape);
    if (new_dsizes) NCI_Free(new_dsizes);
    if (pnc_new_shape) NCI_Free(pnc_new_shape);
    return err;
}
 
/* Helper to append bytes to container (design doc §7.1) */
static int append_u8(int ncid, int varid, const void* bytes, MPI_Offset nbytes,
                     long long *cursor) {
    MPI_Offset start_offset = (MPI_Offset)(*cursor);
    MPI_Offset count_bytes = nbytes;
    int err = ncmpi_put_vara_all(ncid, varid, &start_offset, &count_bytes,
                                  (void*)bytes, count_bytes, MPI_UNSIGNED_CHAR);
    if (err == NC_NOERR) {
        *cursor += (long long)nbytes;
    }
    return err;
}

/* Helper function to write frame data in container mode (design doc §7) */
static int write_progressive_frames(int ncid, int varid, const void *buf,
                                  const MPI_Offset *start,
                                  const MPI_Offset *count, MPI_Datatype buftype, int ndim) {
   int err = NC_NOERR;
   int layers, level_progressive;
   char *algo = NULL;  /* 改为动态分配，避免栈溢出 */
   MPI_Offset algo_len = 0;
   nc_type attr_type = NC_NAT;
   void *compressed_data = NULL;
   int compressed_size = 0;
#ifdef ENABLE_IPCOMP
   NCCHK_var_context *ctx = NULL;
#endif
   long long cursor = 0;  /* Running offset in container */
   int *dims = NULL;  /* 动态分配的维度数组，在函数作用域声明 */
   size_t total_elements = 1;  /* 总元素数 */
   int has_zero_dim = 0;      /* 是否有零维度 */
   int input_size = 0;        /* 输入数据大小（字节） */
   HeaderV1Layout header_layout;
    
    /* 验证输入参数 */
    if (ndim <= 0 || ndim > NC_MAX_VAR_DIMS) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    if (start == NULL || count == NULL || buf == NULL) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    
    compute_header_v1_layout(ndim, &header_layout);

    /* Get progressive compression parameters */
    err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
    if (err != NC_NOERR) return err;
 
     if (attr_type != NC_CHAR) {
         DEBUG_RETURN_ERROR(NC_EBADTYPE);
     }
     
     /* 总是动态分配 algo 以避免栈写入 */
     algo = (char*)malloc((size_t)algo_len + 1);
     if (!algo) { DEBUG_RETURN_ERROR(NC_ENOMEM); }
     
     err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
     if (err != NC_NOERR) { free(algo); return err; }
     algo[(size_t)algo_len] = '\0';
     
    err = ncmpi_get_att_int(ncid, varid, "comp:layers", &layers);
    if (err != NC_NOERR) return err;
    
    err = ncmpi_get_att_int(ncid, varid, "comp:level_progressive", &level_progressive);
    if (err != NC_NOERR) return err;

    int block_size_attr = 0;
    err = ncmpi_get_att_int(ncid, varid, "comp:block_size", &block_size_attr);
    if (err != NC_NOERR) {
        block_size_attr = IPCOMP_DEFAULT_BLOCK_SIZE;
    }
    size_t effective_block_size = (block_size_attr > 0)
                                      ? (size_t)block_size_attr
                                      : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;
    
    /* Get current container length to determine cursor position (design doc §7.2 step 1) */
    {
        int container_ndims = 0;
        int *container_dimids = NULL;
        MPI_Offset container_len = 0;

        err = ncmpi_inq_varndims(ncid, varid, &container_ndims);
        if (err != NC_NOERR) return err;

        if (container_ndims != 1) {
            DEBUG_RETURN_ERROR(NC_EINVAL)
        }

        container_dimids = (int*)malloc(sizeof(int) * (size_t)container_ndims);
        if (!container_dimids) {
            DEBUG_RETURN_ERROR(NC_ENOMEM)
        }

        err = ncmpi_inq_vardimid(ncid, varid, container_dimids);
        if (err != NC_NOERR) {
            free(container_dimids);
            return err;
        }

        err = ncmpi_inq_dimlen(ncid, container_dimids[0], &container_len);
        free(container_dimids);
        if (err != NC_NOERR) return err;

        cursor = (long long)container_len;
    }
     
#ifdef ENABLE_IPCOMP
if (strcmp(algo, "ipcomp") == 0) {
    /* 添加详细计时 */
    double timer_setup = 0, timer_range = 0, timer_compress = 0, timer_payload = 0, timer_index = 0;
    double t_start, t_end;
    int rank = 0;
    
    /* Use IPComp filter for compression */
    t_start = MPI_Wtime();
    NCCHK_filter *filter = ncchk_ipcomp_inq_driver();
    if (filter == NULL) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }

    /* Set up compression context (fill safe defaults) */
    ctx = (NCCHK_var_context*)calloc(1, sizeof(*ctx));
    if (!ctx) { DEBUG_RETURN_ERROR(NC_ENOMEM); }

    /* fill fields safely */
    ctx->ipcomp_layers             = (layers > 0) ? layers : 1;
    ctx->ipcomp_level_progressive  = level_progressive;
    ctx->ipcomp_interp             = 1;   /* cubic by default (matches CLI) */
    ctx->ipcomp_direction          = 0;
    ctx->ipcomp_block_size         = effective_block_size;
    
    t_end = MPI_Wtime();
    timer_setup = t_end - t_start;
    
    /* ---- 计算并保存 GLOBAL data range（用于解压缩） ---- */
    t_start = MPI_Wtime();
    double data_range = 0.0;
    double global_min = 0.0, global_max = 0.0;
    
    if (!has_zero_dim && input_size > 0) {
        /* 计算局部数据范围 */
        double local_min = 0.0, local_max = 0.0;
        
        if (buftype == MPI_FLOAT) {
            const float* fdata = (const float*)buf;
            local_min = fdata[0];
            local_max = fdata[0];
            for (size_t i = 1; i < total_elements; i++) {
                if (fdata[i] < local_min) local_min = fdata[i];
                if (fdata[i] > local_max) local_max = fdata[i];
            }
        } else if (buftype == MPI_DOUBLE) {
            const double* ddata = (const double*)buf;
            local_min = ddata[0];
            local_max = ddata[0];
            for (size_t i = 1; i < total_elements; i++) {
                if (ddata[i] < local_min) local_min = ddata[i];
                if (ddata[i] > local_max) local_max = ddata[i];
            }
        }
        
        /* 全局通信获取全局 min/max */
        PNC *pncp;
        err = PNC_check_id(ncid, &pncp);
        if (err == NC_NOERR && pncp != NULL && pncp->comm != MPI_COMM_NULL) {
            MPI_Allreduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, pncp->comm);
            MPI_Allreduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, pncp->comm);
            MPI_Comm_rank(pncp->comm, &rank);
        } else {
            global_min = local_min;
            global_max = local_max;
        }
        
        data_range = global_max - global_min;
        ctx->ipcomp_data_range = data_range;
        
        /* 保存全局 range 到 header 中 */
        err = header_write_double(ncid, varid, header_layout.range_offset, data_range);
        if (err != NC_NOERR) goto cleanup;
        
        if (rank == 0) {
            fprintf(stderr, "[DEBUG] Global data range: [%g, %g], range=%g\n", 
                    global_min, global_max, data_range);
        }
    }
    
    t_end = MPI_Wtime();
    timer_range = t_end - t_start;
 
    /* 动态分配维度数组 */
    dims = (int*)calloc((size_t)ndim, sizeof(int));
    if (dims == NULL) {
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }
    
    /* 计算总元素数和检查零维度 */
    for (int i = 0; i < ndim; i++) {
        MPI_Offset extent = count[i];
        if (extent < 0) {
            free(dims);
            DEBUG_RETURN_ERROR(NC_EINVAL)
        }
        if (extent == 0) {
            has_zero_dim = 1;
        }
        if (extent > INT_MAX) {
            free(dims);
            DEBUG_RETURN_ERROR(NC_EINVAL)
        }
        dims[i] = (int)extent;

        if (!has_zero_dim) {
            size_t extent_sz = (size_t)extent;
            if (__builtin_mul_overflow(total_elements, extent_sz, &total_elements)) {
                free(dims);
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
        }
    }

    if (has_zero_dim) {
        total_elements = 0;
    }

    if (buf == NULL) {
        free(dims);
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }

    int element_size = 0;
    MPI_Type_size(buftype, &element_size);

    size_t input_size_sz = 0;
    if (!has_zero_dim) {
        if (__builtin_mul_overflow(total_elements, (size_t)element_size, &input_size_sz)) {
            free(dims);
            DEBUG_RETURN_ERROR(NC_EINVAL)
        }
    }

    if (input_size_sz > (size_t)INT_MAX) {
        free(dims);
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    input_size = (int)input_size_sz;
 
    /* ---- Invoke compressor (with zero-size early-out) ---- */
    t_start = MPI_Wtime();
    compressed_data = NULL;
    compressed_size = 0;

    if (!has_zero_dim && input_size > 0) {
        err = filter->compress_alloc((void*)buf,
                                     input_size,
                                     &compressed_data,
                                     &compressed_size,
                                     ndim,
                                     dims,
                                     buftype,
                                     ctx);
        if (err != NC_NOERR) {
            goto cleanup;
        }
        /* 防御性检查：指针与长度需匹配 */
        if ((compressed_size > 0 && compressed_data == NULL) ||
            (compressed_size == 0 && compressed_data != NULL)) {
            err = NC_EPLUGIN;
            goto cleanup;
        }
    } else {
        /* 零尺寸：不压缩、不写 payload，仅写 index（length=0） */
        compressed_data = NULL;
        compressed_size = 0;
    }
    
    t_end = MPI_Wtime();
    timer_compress = t_end - t_start;
 
   /* ---- Step 2: If first write (cursor==0), build and write HEADER (design doc §7.2 step 2) ---- */
   t_start = MPI_Wtime();
   if (cursor == 0) {
       uint8_t *header = NULL;
       size_t header_size = 0;
       err = build_header_v1(ncid, varid, &header, &header_size);
       if (err != NC_NOERR) goto cleanup;
       
       err = append_u8(ncid, varid, header, (MPI_Offset)header_size, &cursor);
       free(header);
       if (err != NC_NOERR) goto cleanup;
   }
   t_end = MPI_Wtime();
   double timer_header = t_end - t_start;
   
   /* ---- Step 4: Write PAYLOAD to container (skip if zero-length, design doc §7.2 step 4) ---- */
   t_start = MPI_Wtime();
   long long payload_offset = cursor;
   if (compressed_size > 0) {
       err = append_u8(ncid, varid, compressed_data, (MPI_Offset)compressed_size, &cursor);
       if (err != NC_NOERR) goto cleanup;
   }
   t_end = MPI_Wtime();
   timer_payload = t_end - t_start;
 
   /* ---- Step 5: Write INDEX to container (offset, length pair, design doc §7.2 step 5) ---- */
   t_start = MPI_Wtime();
   {
       /* Build index entry: [offset, length] as uint64_t pair */
       uint64_t index_entry[2];
       index_entry[0] = (uint64_t)payload_offset;
       index_entry[1] = (uint64_t)compressed_size;
       
       size_t index_bytes = 2 * sizeof(uint64_t);
       err = append_u8(ncid, varid, (uint8_t*)index_entry, (MPI_Offset)index_bytes, &cursor);
       if (err != NC_NOERR) goto cleanup;
       
       /* Step 6: Update index_bytes stored in header */
       uint64_t prev_index_bytes = 0;
       err = header_read_u64(ncid, varid, header_layout.index_bytes_offset, &prev_index_bytes);
       if (err != NC_NOERR) goto cleanup;

       uint64_t new_index_bytes = prev_index_bytes + (uint64_t)index_bytes;
       err = header_write_u64(ncid, varid, header_layout.index_bytes_offset, new_index_bytes);
       if (err != NC_NOERR) goto cleanup;
   }
   t_end = MPI_Wtime();
   timer_index = t_end - t_start;
    
   /* 打印详细计时（仅 rank 0） */
   if (rank == 0) {
       double total = timer_setup + timer_range + timer_compress + timer_header + timer_payload + timer_index;
       fprintf(stderr, "\n[Progressive Write Timing Breakdown (Container Mode)]\n");
       fprintf(stderr, "  Setup:           %8.4f sec  (%5.1f%%)\n", timer_setup, 100.0*timer_setup/total);
       fprintf(stderr, "  Compute range:   %8.4f sec  (%5.1f%%)\n", timer_range, 100.0*timer_range/total);
       fprintf(stderr, "  IPComp compress: %8.4f sec  (%5.1f%%)  ← BOTTLENECK\n", 
               timer_compress, 100.0*timer_compress/total);
       fprintf(stderr, "  Write header:    %8.4f sec  (%5.1f%%)\n", timer_header, 100.0*timer_header/total);
       fprintf(stderr, "  Write payload:   %8.4f sec  (%5.1f%%)\n", timer_payload, 100.0*timer_payload/total);
       fprintf(stderr, "  Write index:     %8.4f sec  (%5.1f%%)\n", timer_index, 100.0*timer_index/total);
       fprintf(stderr, "  ─────────────────────────────────────────\n");
       fprintf(stderr, "  Total:           %8.4f sec\n", total);
        
        /* 压缩性能分析 */
        if (timer_compress / total > 0.8) {
            fprintf(stderr, "\n[Analysis] IPComp compression is the bottleneck (%.1f%% of time)\n", 
                    100.0*timer_compress/total);
            fprintf(stderr, "  Reasons for slow compression:\n");
            fprintf(stderr, "  • Multiple layers (%d layers)\n", ctx->ipcomp_layers);
            if (ctx->ipcomp_level_progressive > 0) {
                fprintf(stderr, "  • Multiple levels (%d progressive levels)\n", ctx->ipcomp_level_progressive);
                fprintf(stderr, "  • Total passes: %d × %d = %d\n",
                        ctx->ipcomp_layers, ctx->ipcomp_level_progressive,
                        ctx->ipcomp_layers * ctx->ipcomp_level_progressive);
            } else {
                fprintf(stderr, "  • Progressive levels: auto (library-managed)\n");
            }
            fprintf(stderr, "  • Each pass: interpolation + quantization + encoding + zstd\n");
            fprintf(stderr, "  • Data size: %.2f MB\n", input_size / (1024.0*1024.0));
            
            double throughput = (input_size / (1024.0*1024.0)) / timer_compress;
            fprintf(stderr, "  • Compression throughput: %.2f MB/s\n", throughput);
            
            if (throughput < 2.0) {
                fprintf(stderr, "\n  Possible optimizations:\n");
                fprintf(stderr, "  - Reduce layers (3 → 2 or 1)\n");
                if (ctx->ipcomp_level_progressive > 0) {
                    fprintf(stderr, "  - Reduce progressive levels (%d → %d)\n",
                            ctx->ipcomp_level_progressive,
                            ctx->ipcomp_level_progressive > 1 ? ctx->ipcomp_level_progressive - 1 : 0);
                } else {
                    fprintf(stderr, "  - Use smaller auto progressive depth\n");
                }
                fprintf(stderr, "  - Use simpler interpolation\n");
                fprintf(stderr, "  - Parallel compression (future work)\n");
            }
        }
        fprintf(stderr, "\n");
    }
}

#endif /* ENABLE_IPCOMP */
     
 cleanup:
     if (algo) {
         free(algo);
     }
     if (dims) {
         free(dims);
     }
#ifdef ENABLE_IPCOMP
    if (ctx) {
        free(ctx);
    }
#endif
     if (compressed_data) {
         free(compressed_data);
     }
     
     return err;
 }
 
/* Helper function to read progressive frames based on error bound (container mode) */
static int read_progressive_frames_error_impl(int ncid, int varid, void *buf,
                                      const MPI_Offset *start,
                                      const MPI_Offset *count, MPI_Datatype buftype, int ndim,
                                      const double *target_rel_ebs, int num_target_ebs,
                                      double fallback_target_rel_eb) {
    int err = NC_NOERR;
    char *algo = NULL;  /* 改为动态分配 */
    char *interp_attr = NULL;
    #ifdef ENABLE_IPCOMP
    double *ctx_local_ebs = NULL;
    #endif
    MPI_Offset algo_len = 0;
    nc_type attr_type = NC_NAT;
    int *full_dims = NULL;
    long long *logical_dims_ll = NULL;
    void *full_buffer = NULL;
    void *compressed_data = NULL;
    uint64_t header_bytes = 0;
    uint64_t index_bytes = 0;
    HeaderV1Layout header_layout;
    int use_custom_list = (target_rel_ebs != NULL && num_target_ebs > 0);
    if (use_custom_list) {
        for (int i = 0; i < num_target_ebs; i++) {
            if (!(target_rel_ebs[i] > 0.0)) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
        }
    }
    
    /* 验证输入参数 */
    if (ndim <= 0 || ndim > NC_MAX_VAR_DIMS) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    if (start == NULL || count == NULL || buf == NULL) {
        DEBUG_RETURN_ERROR(NC_EINVAL)
    }
    
    compute_header_v1_layout(ndim, &header_layout);

    err = header_read_u64(ncid, varid, header_layout.header_bytes_offset, &header_bytes);
    if (err != NC_NOERR) goto cleanup_read_error;

    err = header_read_u64(ncid, varid, header_layout.index_bytes_offset, &index_bytes);
    if (err != NC_NOERR) goto cleanup_read_error;

    
    /* Get progressive compression parameters */
    err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
    if (err != NC_NOERR) goto cleanup_read_error;
 
     if (attr_type != NC_CHAR) {
         DEBUG_RETURN_ERROR(NC_EBADTYPE);
     }
     
     /* 总是动态分配 algo 以避免栈写入 */
     algo = (char*)malloc((size_t)algo_len + 1);
     if (!algo) { DEBUG_RETURN_ERROR(NC_ENOMEM); }
     
    err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
    if (err != NC_NOERR) { free(algo); return err; }
    algo[(size_t)algo_len] = '\0';
    
#ifdef ENABLE_IPCOMP
    if (strcmp(algo, "ipcomp") == 0) {
       /* Read INDEX from container (design doc §8 step 3)
        * For simplicity, assuming single index entry at end of container
        * Layout: [HEADER | PAYLOAD | INDEX(offset, length)]
        */
       uint64_t frame_offset = 0;
       uint64_t frame_length = 0;

       /* Load IPComp metadata for decompression context */
       NCCHK_var_context ctx_local;
       memset(&ctx_local, 0, sizeof(ctx_local));

       int layers_attr = 0;
       err = ncmpi_get_att_int(ncid, varid, "comp:layers", &layers_attr);
       if (err != NC_NOERR) goto cleanup_read_error;

       int level_prog_attr = 0;
       err = ncmpi_get_att_int(ncid, varid, "comp:level_progressive", &level_prog_attr);
       if (err != NC_NOERR) goto cleanup_read_error;

       int block_size_attr = 0;
       err = ncmpi_get_att_int(ncid, varid, "comp:block_size", &block_size_attr);
       if (err != NC_NOERR) block_size_attr = 0;

       double stored_range = 0.0;
       err = header_read_double(ncid, varid, header_layout.range_offset, &stored_range);
       if (err != NC_NOERR) goto cleanup_read_error;

       MPI_Offset interp_len = 0;
       nc_type interp_type = NC_NAT;
       err = ncmpi_inq_att(ncid, varid, "comp:interp", &interp_type, &interp_len);
       if (err == NC_NOERR && interp_type == NC_CHAR) {
           interp_attr = (char*)malloc((size_t)interp_len + 1);
           if (interp_attr == NULL) {
               err = NC_ENOMEM;
               goto cleanup_read_error;
           }
           err = ncmpi_get_att_text(ncid, varid, "comp:interp", interp_attr);
           if (err != NC_NOERR) goto cleanup_read_error;
           interp_attr[(size_t)interp_len] = '\0';
       }

      int interp_op = 1;
      if (interp_attr != NULL) {
          if (strcmp(interp_attr, "linear") == 0) {
              interp_op = 0;
          } else if (strcmp(interp_attr, "cubic") == 0) {
              interp_op = 1;
          }
      }

      size_t effective_block_size_read = (block_size_attr > 0)
                                             ? (size_t)block_size_attr
                                             : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;

      ctx_local.ipcomp_layers = layers_attr;
        ctx_local.ipcomp_level_progressive = level_prog_attr;
        ctx_local.ipcomp_block_size = effective_block_size_read;
        ctx_local.ipcomp_interp = interp_op;
        ctx_local.ipcomp_direction = 0;
      ctx_local.ipcomp_data_range = stored_range;

       if (layers_attr > 0) {
           nc_type ebs_type = NC_NAT;
           MPI_Offset ebs_len = 0;
           int err_inq = ncmpi_inq_att(ncid, varid, "comp:ebs", &ebs_type, &ebs_len);
           if (err_inq == NC_NOERR && ebs_type == NC_DOUBLE && ebs_len > 0) {
               ctx_local.ipcomp_num_ebs = (int)ebs_len;
               ctx_local.ipcomp_ebs = (double*)malloc((size_t)ebs_len * sizeof(double));
               if (ctx_local.ipcomp_ebs == NULL) {
                   err = NC_ENOMEM;
                   goto cleanup_read_error;
               }
               err = ncmpi_get_att_double(ncid, varid, "comp:ebs", ctx_local.ipcomp_ebs);
               if (err != NC_NOERR) {
                   goto cleanup_read_error;
               }
           } else if (err_inq == NC_ENOTATT) {
               ctx_local.ipcomp_num_ebs = 0;
               ctx_local.ipcomp_ebs = NULL;
               err = NC_NOERR;
           } else if (err_inq == NC_NOERR) {
               ctx_local.ipcomp_num_ebs = 0;
               ctx_local.ipcomp_ebs = NULL;
               fprintf(stderr, "[WARN] comp:ebs attribute has unexpected type or length\n");
           } else if (err_inq != NC_NOERR) {
               err = err_inq;
               goto cleanup_read_error;
           }
           ctx_local_ebs = ctx_local.ipcomp_ebs;
       }

        if (use_custom_list) {
            if (ctx_local_ebs != NULL) {
                free(ctx_local_ebs);
                ctx_local_ebs = NULL;
            }
            ctx_local_ebs = (double*)malloc((size_t)num_target_ebs * sizeof(double));
            if (ctx_local_ebs == NULL) {
                err = NC_ENOMEM;
                goto cleanup_read_error;
            }
            for (int i = 0; i < num_target_ebs; i++) {
                ctx_local_ebs[i] = target_rel_ebs[i];
            }
            ctx_local.ipcomp_ebs = ctx_local_ebs;
            ctx_local.ipcomp_num_ebs = num_target_ebs;
        }
       
       /* Get total container length */
       MPI_Offset container_total = 0;
       {
           int container_dimid;
           int _ndims = 0;
       err = ncmpi_inq_varndims(ncid, varid, &_ndims);
           if (err != NC_NOERR || _ndims != 1) {
               err = NC_EINVAL;
               goto cleanup_read_error;
           }
       err = ncmpi_inq_vardimid(ncid, varid, &container_dimid);
           if (err != NC_NOERR) goto cleanup_read_error;
           err = ncmpi_inq_dimlen(ncid, container_dimid, &container_total);
           if (err != NC_NOERR) goto cleanup_read_error;
       }
       
       if (index_bytes > (uint64_t)container_total) {
           err = NC_EINVAL;
           goto cleanup_read_error;
       }
       /* Calculate index position: index_offset = TOTAL - index_bytes */
       MPI_Offset index_offset = container_total - (MPI_Offset)index_bytes;
       
       /* Read index entry (offset, length) from container */
       uint64_t index_entry[2] = {0, 0};
       MPI_Offset index_start = index_offset;
       MPI_Offset index_count = 2 * sizeof(uint64_t);
       err = ncmpi_get_vara_all(ncid, varid, &index_start, &index_count,
                                (void*)index_entry, index_count, MPI_UNSIGNED_CHAR);
       if (err != NC_NOERR) goto cleanup_read_error;
       
       frame_offset = index_entry[0];
       frame_length = index_entry[1];

        /* Validate frame_offset and frame_length to prevent invalid values */
        if (frame_offset > (uint64_t)container_total) {
            err = NC_EINVAL;
            goto cleanup_read_error;
        }
        if (frame_length > (uint64_t)SIZE_MAX || frame_length > (uint64_t)INT_MAX) {
            err = NC_EINVAL;
            goto cleanup_read_error;
        }
        /* Ensure frame doesn't extend beyond container */
        if (frame_offset + frame_length > (uint64_t)container_total) {
            err = NC_EINVAL;
            goto cleanup_read_error;
        }

       size_t frame_bytes = (size_t)frame_length;
       int compressed_int_len = (int)frame_length;

       compressed_data = calloc(1, frame_bytes > 0 ? frame_bytes : 1);
       if (compressed_data == NULL) {
           err = NC_ENOMEM;
           goto cleanup_read_error;
       }

       /* Read PAYLOAD from container (design doc §8 step 5) */
       MPI_Offset payload_start_pos = (MPI_Offset)frame_offset;
       MPI_Offset payload_count_bytes = (MPI_Offset)frame_length;
       err = ncmpi_get_vara_all(ncid, varid, &payload_start_pos, &payload_count_bytes,
                                compressed_data, payload_count_bytes, MPI_UNSIGNED_CHAR);
       if (err != NC_NOERR) {
           free(compressed_data);
           compressed_data = NULL;
           goto cleanup_read_error;
       }

        int element_size = 0;
        MPI_Type_size(buftype, &element_size);
        if (element_size <= 0) {
            err = NC_EBADTYPE;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        size_t roi_elements = 1;
        int zero_dim = 0;
        for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }
            if (extent == 0) {
                zero_dim = 1;
            }
            if (!zero_dim) {
                if (__builtin_mul_overflow(roi_elements, (size_t)extent, &roi_elements)) {
                    err = NC_EINVAL;
                    free(compressed_data);
                    compressed_data = NULL;
                    goto cleanup_read_error;
                }
            }
        }

        if (zero_dim || roi_elements == 0) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        int logical_ndim_attr = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim_attr);
        if (err != NC_NOERR) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }
        if (logical_ndim_attr != ndim) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        logical_dims_ll = (long long*)malloc((size_t)ndim * sizeof(long long));
        if (logical_dims_ll == NULL) {
            err = NC_ENOMEM;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }
        err = ncmpi_get_att_longlong(ncid, varid, "comp:logical_dims", logical_dims_ll);
        if (err != NC_NOERR) {
            free(logical_dims_ll);
            logical_dims_ll = NULL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        full_dims = (int*)malloc((size_t)ndim * sizeof(int));
        if (full_dims == NULL) {
            err = NC_ENOMEM;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        size_t full_elements = 1;
        int full_request = 1;
        for (int i = 0; i < ndim; i++) {
            long long dim_ll = logical_dims_ll[i];
            if (dim_ll <= 0 || dim_ll > INT_MAX) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }
            full_dims[i] = (int)dim_ll;
            if (__builtin_mul_overflow(full_elements, (size_t)full_dims[i], &full_elements)) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }

            MPI_Offset start_i = start[i];
            MPI_Offset count_i = count[i];
            if (start_i < 0 || start_i > (MPI_Offset)full_dims[i]) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }
            if (count_i < 0 || count_i > (MPI_Offset)full_dims[i]) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }
            MPI_Offset remaining = (MPI_Offset)full_dims[i] - start_i;
            if (count_i > remaining) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }
            if (start_i != 0 || count_i != (MPI_Offset)full_dims[i]) {
                full_request = 0;
            }
        }

        size_t full_bytes = 0;
        if (__builtin_mul_overflow(full_elements, (size_t)element_size, &full_bytes)) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }
        if (full_bytes > (size_t)INT_MAX) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        int tmp_out_len = (int)full_bytes;
        if (compressed_int_len == 0 || tmp_out_len == 0) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_error;
        }

        double effective_target_rel_eb = use_custom_list ? 0.0 : fallback_target_rel_eb;

        if (full_request) {
            double t0 = MPI_Wtime();
            err = ncchk_ipcomp_decompress_progressive_error(compressed_data, compressed_int_len,
                                                            buf, &tmp_out_len,
                                                            ndim, full_dims, buftype,
                                                            effective_target_rel_eb, &ctx_local);
            double t1 = MPI_Wtime();
            if (err == NC_NOERR) {
                fprintf(stderr, "decompress_progressive time = %.6gs\n", t1 - t0);
            }
            free(compressed_data);
            compressed_data = NULL;
            if (err != NC_NOERR) goto cleanup_read_error;
        } else {
            full_buffer = malloc(full_bytes);
            if (full_buffer == NULL) {
                err = NC_ENOMEM;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_error;
            }

            int full_out_len = (int)full_bytes;
            double t0 = MPI_Wtime();
            err = ncchk_ipcomp_decompress_progressive_error(compressed_data, compressed_int_len,
                                                            full_buffer, &full_out_len,
                                                            ndim, full_dims, buftype,
                                                            effective_target_rel_eb, &ctx_local);
            double t1 = MPI_Wtime();
            if (err == NC_NOERR) {
                fprintf(stderr, "decompress_progressive time = %.6gs\n", t1 - t0);
            }
            free(compressed_data);
            compressed_data = NULL;
            if (err != NC_NOERR) goto cleanup_read_error;

            copy_roi_to_buffer(full_buffer, buf, ndim, start, count, full_dims, (size_t)element_size);
            free(full_buffer);
            full_buffer = NULL;
        }
     }
 #endif /* ENABLE_IPCOMP */
     
 cleanup_read_error:
     if (algo) free(algo);
    if (interp_attr) free(interp_attr);
    if (full_buffer) free(full_buffer);
    if (full_dims) free(full_dims);
    if (logical_dims_ll) free(logical_dims_ll);
     if (compressed_data) free(compressed_data);
    #ifdef ENABLE_IPCOMP
    if (ctx_local_ebs) free(ctx_local_ebs);
    #endif
     return err;
 }
 
/* Helper function to read progressive frames based on bitrate constraint (container mode) */
static int read_progressive_frames_bitrate(int ncid, int varid, void *buf,
                                        const MPI_Offset *start,
                                        const MPI_Offset *count, MPI_Datatype buftype, int ndim,
                                        double max_ratio_or_bytes) {
    int err = NC_NOERR;
    char *algo = NULL;  /* 动态分配，避免大栈对象 */
    char *interp_attr = NULL;
    #ifdef ENABLE_IPCOMP
    double *ctx_local_ebs = NULL;
    #endif
    MPI_Offset algo_len = 0;
    nc_type attr_type = NC_NAT;
    int *full_dims = NULL;
    long long *logical_dims_ll = NULL;
    void *full_buffer = NULL;
    void *compressed_data = NULL;
    uint64_t header_bytes = 0;
    uint64_t index_bytes = 0;
    HeaderV1Layout header_layout;

    if (ndim <= 0 || ndim > NC_MAX_VAR_DIMS)
        DEBUG_RETURN_ERROR(NC_EINVAL)
    if (start == NULL || count == NULL || buf == NULL)
        DEBUG_RETURN_ERROR(NC_EINVAL)

    compute_header_v1_layout(ndim, &header_layout);

    err = header_read_u64(ncid, varid, header_layout.header_bytes_offset, &header_bytes);
    if (err != NC_NOERR) goto cleanup_read_bitrate;

    err = header_read_u64(ncid, varid, header_layout.index_bytes_offset, &index_bytes);
    if (err != NC_NOERR) goto cleanup_read_bitrate;


    err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
    if (err != NC_NOERR) goto cleanup_read_bitrate;
    if (attr_type != NC_CHAR) {
        err = NC_EBADTYPE;
        goto cleanup_read_bitrate;
    }

    algo = (char*)malloc((size_t)algo_len + 1);
    if (algo == NULL) {
        err = NC_ENOMEM;
        goto cleanup_read_bitrate;
    }
    err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
    if (err != NC_NOERR) goto cleanup_read_bitrate;
    algo[(size_t)algo_len] = '\0';

#ifdef ENABLE_IPCOMP
    if (strcmp(algo, "ipcomp") == 0) {
        NCCHK_var_context ctx_local;
        memset(&ctx_local, 0, sizeof(ctx_local));

        int layers_attr = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:layers", &layers_attr);
        if (err != NC_NOERR) goto cleanup_read_bitrate;

        int level_prog_attr = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:level_progressive", &level_prog_attr);
        if (err != NC_NOERR) goto cleanup_read_bitrate;

        int block_size_attr = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:block_size", &block_size_attr);
        if (err != NC_NOERR) block_size_attr = 0;

        double stored_range = 0.0;
        err = header_read_double(ncid, varid, header_layout.range_offset, &stored_range);
        if (err != NC_NOERR) goto cleanup_read_bitrate;

        MPI_Offset interp_len = 0;
        nc_type interp_type = NC_NAT;
        err = ncmpi_inq_att(ncid, varid, "comp:interp", &interp_type, &interp_len);
        if (err == NC_NOERR && interp_type == NC_CHAR) {
            interp_attr = (char*)malloc((size_t)interp_len + 1);
            if (interp_attr == NULL) {
                err = NC_ENOMEM;
                goto cleanup_read_bitrate;
            }
            err = ncmpi_get_att_text(ncid, varid, "comp:interp", interp_attr);
            if (err != NC_NOERR) goto cleanup_read_bitrate;
            interp_attr[(size_t)interp_len] = '\0';
        }

        int interp_op = 1;
        if (interp_attr != NULL) {
            if (strcmp(interp_attr, "linear") == 0) {
                interp_op = 0;
            } else if (strcmp(interp_attr, "cubic") == 0) {
                interp_op = 1;
            }
        }

        ctx_local.ipcomp_layers = layers_attr;
        ctx_local.ipcomp_level_progressive = level_prog_attr;
        size_t effective_block_size_read = (block_size_attr > 0)
                                               ? (size_t)block_size_attr
                                               : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;
        ctx_local.ipcomp_block_size = effective_block_size_read;
        ctx_local.ipcomp_interp = interp_op;
        ctx_local.ipcomp_direction = 0;
        ctx_local.ipcomp_data_range = stored_range;

        if (layers_attr > 0) {
            nc_type ebs_type = NC_NAT;
            MPI_Offset ebs_len = 0;
            int err_inq = ncmpi_inq_att(ncid, varid, "comp:ebs", &ebs_type, &ebs_len);
            if (err_inq == NC_NOERR && ebs_type == NC_DOUBLE && ebs_len > 0) {
                ctx_local.ipcomp_num_ebs = (int)ebs_len;
                ctx_local.ipcomp_ebs = (double*)malloc((size_t)ebs_len * sizeof(double));
                if (ctx_local.ipcomp_ebs == NULL) {
                    err = NC_ENOMEM;
                    goto cleanup_read_bitrate;
                }
                err = ncmpi_get_att_double(ncid, varid, "comp:ebs", ctx_local.ipcomp_ebs);
                if (err != NC_NOERR) {
                    goto cleanup_read_bitrate;
                }
            } else if (err_inq == NC_ENOTATT) {
                ctx_local.ipcomp_num_ebs = 0;
                ctx_local.ipcomp_ebs = NULL;
                err = NC_NOERR;
            } else if (err_inq == NC_NOERR) {
                ctx_local.ipcomp_num_ebs = 0;
                ctx_local.ipcomp_ebs = NULL;
                fprintf(stderr, "[WARN] comp:ebs attribute has unexpected type or length\n");
            } else if (err_inq != NC_NOERR) {
                err = err_inq;
                goto cleanup_read_bitrate;
            }
            ctx_local_ebs = ctx_local.ipcomp_ebs;
        }

        MPI_Offset total_elements_offset = 1;
        int has_zero_dim = 0;
        for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                err = NC_EINVAL;
                goto cleanup_read_bitrate;
            }
            if (extent == 0)
                has_zero_dim = 1;
            if (!has_zero_dim) {
                if (__builtin_mul_overflow(total_elements_offset, extent, &total_elements_offset)) {
                    err = NC_EINVAL;
                    goto cleanup_read_bitrate;
                }
            }
        }

        int element_size = 0;
        MPI_Type_size(buftype, &element_size);

        size_t original_size = 0;
        if (!has_zero_dim) {
            if (__builtin_mul_overflow((size_t)total_elements_offset, (size_t)element_size, &original_size)) {
                err = NC_EINVAL;
                goto cleanup_read_bitrate;
            }
        }

        size_t byte_budget = 0;
        if (max_ratio_or_bytes < 1.0 && max_ratio_or_bytes > 0.0)
            byte_budget = (size_t)((double)original_size * max_ratio_or_bytes);
        else if (max_ratio_or_bytes >= 1.0)
            byte_budget = (size_t)max_ratio_or_bytes;

        uint64_t frame_offset = 0;
        uint64_t frame_length = 0;

        MPI_Offset container_total = 0;
        {
            int dimid = 0;
            int _ndims = 0;
            err = ncmpi_inq_varndims(ncid, varid, &_ndims);
            if (err != NC_NOERR || _ndims != 1) {
                err = NC_EINVAL;
                goto cleanup_read_bitrate;
            }
            err = ncmpi_inq_vardimid(ncid, varid, &dimid);
            if (err != NC_NOERR) goto cleanup_read_bitrate;
            err = ncmpi_inq_dimlen(ncid, dimid, &container_total);
            if (err != NC_NOERR) goto cleanup_read_bitrate;
        }

        if (index_bytes > (uint64_t)container_total) {
            err = NC_EINVAL;
            goto cleanup_read_bitrate;
        }
        MPI_Offset index_position = container_total - (MPI_Offset)index_bytes;

        uint64_t index_entry[2] = {0, 0};
        MPI_Offset idx_start = index_position;
        MPI_Offset idx_count = 2 * (MPI_Offset)sizeof(uint64_t);
        err = ncmpi_get_vara_all(ncid, varid, &idx_start, &idx_count,
                                 (void*)index_entry, idx_count, MPI_UNSIGNED_CHAR);
        if (err != NC_NOERR) goto cleanup_read_bitrate;

        frame_offset = index_entry[0];
        frame_length = index_entry[1];

        /* Validate frame_offset before applying budget constraints */
        if (frame_offset > (uint64_t)container_total) {
            err = NC_EINVAL;
            goto cleanup_read_bitrate;
        }
        
        /* 
         * 关键修复：不要截断压缩数据读取！
         * 即使 byte_budget 很小，也必须读取完整的压缩数据，
         * 因为 Zstd 流必须是完整的。byte_budget 应该传递给
         * IPComp 解压缩函数，让它内部决定如何处理。
         */
        /* REMOVED: if (byte_budget > 0 && frame_length > (uint64_t)byte_budget)
         *              frame_length = (uint64_t)byte_budget;
         */

        if (frame_length > (uint64_t)SIZE_MAX || frame_length > (uint64_t)INT_MAX) {
            err = NC_EINVAL;
            goto cleanup_read_bitrate;
        }
        /* Ensure frame doesn't extend beyond container */
        if (frame_offset + frame_length > (uint64_t)container_total) {
            err = NC_EINVAL;
            goto cleanup_read_bitrate;
        }

        size_t frame_bytes = (size_t)frame_length;
        int compressed_int_len = (int)frame_length;
        compressed_data = calloc(1, frame_bytes > 0 ? frame_bytes : 1);
        if (compressed_data == NULL) {
            err = NC_ENOMEM;
            goto cleanup_read_bitrate;
        }

        MPI_Offset payload_start_pos = (MPI_Offset)frame_offset;
        MPI_Offset payload_count_bytes = (MPI_Offset)frame_length;
        
        /* 调试输出 */
        fprintf(stderr, "[DEBUG] Reading bitrate payload: offset=%llu, length=%llu bytes\n",
                (unsigned long long)frame_offset, (unsigned long long)frame_length);
        
        err = ncmpi_get_vara_all(ncid, varid, &payload_start_pos, &payload_count_bytes,
                                 compressed_data, payload_count_bytes, MPI_UNSIGNED_CHAR);
        if (err != NC_NOERR) {
            fprintf(stderr, "[ERROR] Failed to read payload: err=%d\n", err);
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }
        
        /* 验证读取的数据是否有效（检查前几个字节） */
        if (frame_length >= 16) {
            fprintf(stderr, "[DEBUG] First 16 bytes of compressed data: ");
            for (int i = 0; i < 16; i++) {
                fprintf(stderr, "%02x ", ((unsigned char*)compressed_data)[i]);
            }
            fprintf(stderr, "\n");
            
            /* 检查是否是 IPComp header (IPCP magic = 0x49504350) */
            unsigned char* data_bytes = (unsigned char*)compressed_data;
            if (data_bytes[0] == 0x50 && data_bytes[1] == 0x43 && 
                data_bytes[2] == 0x50 && data_bytes[3] == 0x49) {
                fprintf(stderr, "[DEBUG] Detected IPComp header (IPCP)\n");
            } else {
                /* 可能是纯 Zstd 格式，检查 Zstd magic (0x28B52FFD) */
                uint32_t magic = 0;
                memcpy(&magic, data_bytes, sizeof(uint32_t));
                if (magic == 0xFD2FB528u) {
                    fprintf(stderr, "[DEBUG] Detected Zstd magic header\n");
                } else {
                    fprintf(stderr, "[WARNING] Unrecognized compressed data format (magic=0x%08x)\n", magic);
                }
            }
        }

        if (has_zero_dim || total_elements_offset == 0) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        if (element_size <= 0) {
            err = NC_EBADTYPE;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        int logical_ndim_attr = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim_attr);
        if (err != NC_NOERR) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }
        if (logical_ndim_attr != ndim) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        logical_dims_ll = (long long*)malloc((size_t)ndim * sizeof(long long));
        if (logical_dims_ll == NULL) {
            err = NC_ENOMEM;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }
        err = ncmpi_get_att_longlong(ncid, varid, "comp:logical_dims", logical_dims_ll);
        if (err != NC_NOERR) {
            free(logical_dims_ll);
            logical_dims_ll = NULL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        full_dims = (int*)malloc((size_t)ndim * sizeof(int));
        if (full_dims == NULL) {
            err = NC_ENOMEM;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        size_t full_elements = 1;
        int full_request = 1;
        for (int i = 0; i < ndim; i++) {
            long long dim_ll = logical_dims_ll[i];
            if (dim_ll <= 0 || dim_ll > INT_MAX) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }
            full_dims[i] = (int)dim_ll;
            if (__builtin_mul_overflow(full_elements, (size_t)full_dims[i], &full_elements)) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }

            MPI_Offset start_i = start[i];
            MPI_Offset count_i = count[i];
            if (start_i < 0 || start_i > (MPI_Offset)full_dims[i]) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }
            if (count_i < 0 || count_i > (MPI_Offset)full_dims[i]) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }
            MPI_Offset remaining = (MPI_Offset)full_dims[i] - start_i;
            if (count_i > remaining) {
                err = NC_EINVAL;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }
            if (start_i != 0 || count_i != (MPI_Offset)full_dims[i]) {
                full_request = 0;
            }
        }

        size_t full_bytes = 0;
        if (__builtin_mul_overflow(full_elements, (size_t)element_size, &full_bytes)) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }
        if (full_bytes > (size_t)INT_MAX) {
            err = NC_EINVAL;
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        int tmp_out_len = (int)full_bytes;
        if (compressed_int_len == 0 || tmp_out_len == 0) {
            free(compressed_data);
            compressed_data = NULL;
            goto cleanup_read_bitrate;
        }

        if (full_request) {
            err = ncchk_ipcomp_decompress_progressive_bitrate(compressed_data, compressed_int_len,
                                                              buf, &tmp_out_len,
                                                              ndim, full_dims, buftype, max_ratio_or_bytes, &ctx_local);
            free(compressed_data);
            compressed_data = NULL;
            if (err != NC_NOERR) goto cleanup_read_bitrate;
        } else {
            full_buffer = malloc(full_bytes);
            if (full_buffer == NULL) {
                err = NC_ENOMEM;
                free(compressed_data);
                compressed_data = NULL;
                goto cleanup_read_bitrate;
            }

            int full_out_len = (int)full_bytes;
            err = ncchk_ipcomp_decompress_progressive_bitrate(compressed_data, compressed_int_len,
                                                              full_buffer, &full_out_len,
                                                              ndim, full_dims, buftype, max_ratio_or_bytes, &ctx_local);
            free(compressed_data);
            compressed_data = NULL;
            if (err != NC_NOERR) goto cleanup_read_bitrate;

            copy_roi_to_buffer(full_buffer, buf, ndim, start, count, full_dims, (size_t)element_size);
            free(full_buffer);
            full_buffer = NULL;
        }
    }
    else {
        err = NC_EINVAL; /* Unsupported algorithm */
    }
#else
    err = NC_ENOTBUILT;
#endif /* ENABLE_IPCOMP */

cleanup_read_bitrate:
    if (algo) free(algo);
    if (interp_attr) free(interp_attr);
    if (full_buffer) free(full_buffer);
    if (full_dims) free(full_dims);
    if (logical_dims_ll) free(logical_dims_ll);
    if (compressed_data) free(compressed_data);
    #ifdef ENABLE_IPCOMP
    if (ctx_local_ebs) free(ctx_local_ebs);
    #endif
    return err;
}
 
 /* Define progressive compression for a variable */
 int ncmpi_def_var_progressive(int ncid, int varid,
                              const char* algo,
                              const char* interp,
                              int layers,
                              int group_bits,
                              const double* ebs,
                              int num_ebs,
                              size_t block_size,
                              int level_progressive) {
     int err = NC_NOERR;
     PNC *pncp;
     char attr_name[NC_MAX_NAME];
 
     /* Validate inputs */
     if (algo == NULL || interp == NULL) {
         DEBUG_RETURN_ERROR(NC_EINVAL)
     }
     
    if (layers <= 0 || level_progressive < 0 || group_bits <= 0) {
         DEBUG_RETURN_ERROR(NC_EINVAL)
     }
     
     if (ebs != NULL && num_ebs != layers) {
         DEBUG_RETURN_ERROR(NC_EINVAL)
     }
 
     /* Check if ncid is valid */
     err = PNC_check_id(ncid, &pncp);
     if (err != NC_NOERR) return err;
 
     /* Check if we are in define mode */
     if (!(pncp->flag & NC_MODE_DEF)) {
         DEBUG_RETURN_ERROR(NC_ENOTINDEFINE)
     }
 
     /* Check if varid is valid */
     if (varid < 0 || varid >= pncp->nvars) {
         DEBUG_RETURN_ERROR(NC_ENOTVAR)
     }
 
     /* Set progressive compression attributes */
     
     /* comp:algo */
     err = ncmpi_put_att_text(ncid, varid, "comp:algo", strlen(algo), algo);
     if (err != NC_NOERR) return err;
     
     /* comp:interp */
     err = ncmpi_put_att_text(ncid, varid, "comp:interp", strlen(interp), interp);
     if (err != NC_NOERR) return err;
     
     /* comp:layers */
     err = ncmpi_put_att_int(ncid, varid, "comp:layers", NC_INT, 1, &layers);
     if (err != NC_NOERR) return err;
     
     /* comp:bitgroup */
     sprintf(attr_name, "%dx%d", 32, group_bits);
     err = ncmpi_put_att_text(ncid, varid, "comp:bitgroup", strlen(attr_name), attr_name);
     if (err != NC_NOERR) return err;
     
     /* comp:level_progressive */
     err = ncmpi_put_att_int(ncid, varid, "comp:level_progressive", NC_INT, 1, &level_progressive);
     if (err != NC_NOERR) return err;
     
    /* comp:block_size */
    size_t effective_block_size_def = (block_size > 0)
                                          ? block_size
                                          : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;
    int block_size_int = (int)effective_block_size_def;
    err = ncmpi_put_att_int(ncid, varid, "comp:block_size", NC_INT, 1, &block_size_int);
     if (err != NC_NOERR) return err;
     
     /* comp:layout */
     err = ncmpi_put_att_text(ncid, varid, "comp:layout", 11, "chunk-major");
     if (err != NC_NOERR) return err;
     
     /* comp:codec_version */
     err = ncmpi_put_att_text(ncid, varid, "comp:codec_version", 5, "1.0.0");
     if (err != NC_NOERR) return err;
     
    /* comp:ebs - error bounds per layer */
    if (ebs != NULL) {
        err = ncmpi_put_att_double(ncid, varid, "comp:ebs", NC_DOUBLE, num_ebs, ebs);
        if (err != NC_NOERR) return err;
    }

  /* Convert variable into progressive container as per design doc (container mode) */
  err = convert_progressive_container(ncid, varid, layers, level_progressive, group_bits);
   if (err != NC_NOERR) return err;

   /* 标记主变量为 progressive compressed
    * enddef 时会检查这个属性并将 len 设为 4（最小值）
    */
   int is_progressive = 1;
   err = ncmpi_put_att_int(ncid, varid, "_ProgressiveCompressed", NC_INT, 1, &is_progressive);
   if (err != NC_NOERR) return err;

   return NC_NOERR;
}
 
/* Write progressive compressed data */
int ncmpi_put_vara_all_progressive(int ncid, int varid,
                                  const MPI_Offset *start, const MPI_Offset *count,
                                  const void *buf, MPI_Datatype buftype) {
     int err = NC_NOERR;
     PNC *pncp;
     char algo[NC_MAX_NAME + 1] = {0};
     MPI_Offset algo_len = 0;
     nc_type attr_type = NC_NAT;
     
     /* Check if ncid is valid */
     err = PNC_check_id(ncid, &pncp);
     if (err != NC_NOERR) return err;
 
     /* Check if varid is valid */
     if (varid < 0 || varid >= pncp->nvars) {
         DEBUG_RETURN_ERROR(NC_ENOTVAR)
     }
 
     /* Check if this variable has progressive compression attributes */
     err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
     if (err != NC_NOERR) {
         /* No progressive compression, use regular put */
         int ndim;
         err = ncmpi_inq_varndims(ncid, varid, &ndim);
         if (err != NC_NOERR) return err;
         
        MPI_Offset bufcount = 1;
        for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
            if (__builtin_mul_overflow(bufcount, extent, &bufcount)) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
        }
        return ncmpi_put_vara_all(ncid, varid, start, count, buf, bufcount, buftype);
    }
    
    /* Use logical dimensions stored on container */
    int logical_ndim = 0;
    err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim);
    if (err != NC_NOERR) return err;
    if (logical_ndim <= 0 || logical_ndim > NC_MAX_VAR_DIMS)
        DEBUG_RETURN_ERROR(NC_EINVAL)

    return write_progressive_frames(ncid, varid, buf, start, count, buftype, logical_ndim);
 }
 
/* Read with error bound constraint */
int ncmpi_get_vara_progressive_error(int ncid, int varid,
                                    const MPI_Offset *start, const MPI_Offset *count,
                                    double target_rel_eb,
                                    void *buf, MPI_Datatype buftype) {
     int err = NC_NOERR;
     PNC *pncp;
     char algo[NC_MAX_NAME + 1] = {0};
     MPI_Offset algo_len = 0;
     nc_type attr_type = NC_NAT;
     
     /* Check if ncid is valid */
     err = PNC_check_id(ncid, &pncp);
     if (err != NC_NOERR) return err;
 
     /* Check if varid is valid */
     if (varid < 0 || varid >= pncp->nvars) {
         DEBUG_RETURN_ERROR(NC_ENOTVAR)
     }
 
     /* Check if this variable has progressive compression */
     err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
     if (err != NC_NOERR) {
         /* No progressive compression, use regular get */
         int ndim;
         err = ncmpi_inq_varndims(ncid, varid, &ndim);
         if (err != NC_NOERR) return err;
         
         MPI_Offset bufcount = 1;
         for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
            if (__builtin_mul_overflow(bufcount, extent, &bufcount)) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
         }
        return ncmpi_get_vara_all(ncid, varid, start, count, buf, bufcount, buftype);
    }
    
    err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
    if (err != NC_NOERR) return err;
    if (attr_type != NC_CHAR) { DEBUG_RETURN_ERROR(NC_EBADTYPE); }

    if (algo_len > NC_MAX_NAME) {
        char *tmp = (char*)malloc((size_t)algo_len + 1);
        if (!tmp) DEBUG_RETURN_ERROR(NC_ENOMEM);
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", tmp);
        if (err != NC_NOERR) { free(tmp); return err; }
        tmp[algo_len] = '\0';
        strncpy(algo, tmp, NC_MAX_NAME);
        algo[NC_MAX_NAME] = '\0';
        free(tmp);
    } else {
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
        if (err != NC_NOERR) return err;
        algo[(size_t)algo_len] = '\0';
    }
    
    if (strcmp(algo, "ipcomp") == 0) {
        int logical_ndim = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim);
        if (err != NC_NOERR) return err;
        if (logical_ndim <= 0 || logical_ndim > NC_MAX_VAR_DIMS)
            DEBUG_RETURN_ERROR(NC_EINVAL)

        return read_progressive_frames_error(ncid, varid, buf, start, count, buftype, logical_ndim, target_rel_eb);
     } else {
         DEBUG_RETURN_ERROR(NC_EINVAL) /* Unsupported algorithm */
     }
 }

int ncmpi_get_vara_progressive_error_multi(int ncid, int varid,
                                    const MPI_Offset *start, const MPI_Offset *count,
                                    const double *target_rel_ebs, int num_target_ebs,
                                    void *buf, MPI_Datatype buftype) {
     int err = NC_NOERR;
     PNC *pncp;
     char algo[NC_MAX_NAME + 1] = {0};
     MPI_Offset algo_len = 0;
     nc_type attr_type = NC_NAT;

     if (target_rel_ebs == NULL || num_target_ebs <= 0) {
         DEBUG_RETURN_ERROR(NC_EINVAL)
     }

     /* Check if ncid is valid */
     err = PNC_check_id(ncid, &pncp);
     if (err != NC_NOERR) return err;

     /* Check if varid is valid */
     if (varid < 0 || varid >= pncp->nvars) {
         DEBUG_RETURN_ERROR(NC_ENOTVAR)
     }

     /* Check if this variable has progressive compression */
     err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
     if (err != NC_NOERR) {
         /* No progressive compression, use regular get */
         int ndim;
         err = ncmpi_inq_varndims(ncid, varid, &ndim);
         if (err != NC_NOERR) return err;

         MPI_Offset bufcount = 1;
         for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
            if (__builtin_mul_overflow(bufcount, extent, &bufcount)) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
         }
        return ncmpi_get_vara_all(ncid, varid, start, count, buf, bufcount, buftype);
    }

    if (attr_type != NC_CHAR) { DEBUG_RETURN_ERROR(NC_EBADTYPE); }

    if (algo_len > NC_MAX_NAME) {
        char *tmp = (char*)malloc((size_t)algo_len + 1);
        if (!tmp) DEBUG_RETURN_ERROR(NC_ENOMEM);
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", tmp);
        if (err != NC_NOERR) { free(tmp); return err; }
        tmp[algo_len] = '\0';
        strncpy(algo, tmp, NC_MAX_NAME);
        algo[NC_MAX_NAME] = '\0';
        free(tmp);
    } else {
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
        if (err != NC_NOERR) return err;
        algo[(size_t)algo_len] = '\0';
    }

    if (strcmp(algo, "ipcomp") == 0) {
        int logical_ndim = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim);
        if (err != NC_NOERR) return err;
        if (logical_ndim <= 0 || logical_ndim > NC_MAX_VAR_DIMS)
            DEBUG_RETURN_ERROR(NC_EINVAL)

        return read_progressive_frames_error_impl(ncid, varid, buf, start, count, buftype,
                                                  logical_ndim, target_rel_ebs, num_target_ebs, 0.0);
    } else {
        DEBUG_RETURN_ERROR(NC_EINVAL) /* Unsupported algorithm */
    }
}
 
/* Read with bitrate constraint */
int ncmpi_get_vara_progressive_bitrate(int ncid, int varid,
                                      const MPI_Offset *start, const MPI_Offset *count,
                                      double max_ratio_or_bytes,
                                      void *buf, MPI_Datatype buftype) {
     int err = NC_NOERR;
     PNC *pncp;
     char algo[NC_MAX_NAME + 1] = {0};
     MPI_Offset algo_len = 0;
     nc_type attr_type = NC_NAT;
     
     /* Check if ncid is valid */
     err = PNC_check_id(ncid, &pncp);
     if (err != NC_NOERR) return err;
 
     /* Check if varid is valid */
     if (varid < 0 || varid >= pncp->nvars) {
         DEBUG_RETURN_ERROR(NC_ENOTVAR)
     }
 
     /* Check if this variable has progressive compression */
     err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
     if (err != NC_NOERR) {
         /* No progressive compression, use regular get */
         int ndim;
         err = ncmpi_inq_varndims(ncid, varid, &ndim);
         if (err != NC_NOERR) return err;
         
         MPI_Offset bufcount = 1;
         for (int i = 0; i < ndim; i++) {
            MPI_Offset extent = count[i];
            if (extent < 0) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
            if (__builtin_mul_overflow(bufcount, extent, &bufcount)) {
                DEBUG_RETURN_ERROR(NC_EINVAL)
            }
         }
        return ncmpi_get_vara_all(ncid, varid, start, count, buf, bufcount, buftype);
    }
    
    err = ncmpi_inq_att(ncid, varid, "comp:algo", &attr_type, &algo_len);
    if (err != NC_NOERR) return err;
    if (attr_type != NC_CHAR) { DEBUG_RETURN_ERROR(NC_EBADTYPE); }

    if (algo_len > NC_MAX_NAME) {
        char *tmp = (char*)malloc((size_t)algo_len + 1);
        if (!tmp) DEBUG_RETURN_ERROR(NC_ENOMEM);
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", tmp);
        if (err != NC_NOERR) { free(tmp); return err; }
        tmp[algo_len] = '\0';
        strncpy(algo, tmp, NC_MAX_NAME);
        algo[NC_MAX_NAME] = '\0';
        free(tmp);
    } else {
        err = ncmpi_get_att_text(ncid, varid, "comp:algo", algo);
        if (err != NC_NOERR) return err;
        algo[(size_t)algo_len] = '\0';
    }
    
    if (strcmp(algo, "ipcomp") == 0) {
        int logical_ndim = 0;
        err = ncmpi_get_att_int(ncid, varid, "comp:logical_ndim", &logical_ndim);
        if (err != NC_NOERR) return err;
        if (logical_ndim <= 0 || logical_ndim > NC_MAX_VAR_DIMS)
            DEBUG_RETURN_ERROR(NC_EINVAL)

        return read_progressive_frames_bitrate(ncid, varid, buf, start, count, buftype, logical_ndim, max_ratio_or_bytes);
     } else {
         DEBUG_RETURN_ERROR(NC_EINVAL) /* Unsupported algorithm */
     }
 }