/*
 *  Copyright (C) 2025, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <limits.h>
#include <stdint.h>
#include <math.h>
#include <mpi.h>
#include <pnetcdf.h>

#include <pnc_debug.h>
#include <ncchkio_driver.h>
#include <ncchk_filter_driver.h>
#include <common.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declarations for helper functions used before definition */
static int mpi_to_ipcomp_type(MPI_Datatype dtype);
static int ipcomp_resolve_layers(MPI_Datatype dtype, NCCHK_var_context* ctx);

/* C wrapper functions for IPComp C++ interface */
extern void* ipcomp_create_compressor(int ndim, const int* dims, int interp_op, int direction_op,
                                     int layers, size_t block_size, int level_progressive);
extern void ipcomp_destroy_compressor(void* compressor);
extern int ipcomp_setup(void* compressor, const void* data, int data_type);
extern int ipcomp_setup_layers(void* compressor, const void* data, int data_type);
extern unsigned char* ipcomp_compress(void* compressor, const void* data, int data_type,
                                      size_t* compressed_size, int ndim, const int* dims);
extern void* ipcomp_decompress_error(void* compressor, const unsigned char* compressed_data,
                                     int data_type, const double* target_rel_ebs,
                                     int num_target_ebs);
extern void* ipcomp_decompress_bitrate(void* compressor, const unsigned char* compressed_data,
                                       int data_type, const double* target_bitrates,
                                       int num_target_bitrates);
extern void ipcomp_free_buffer(void* buffer);
extern int ipcomp_set_range(void* compressor, double data_range);
extern int ipcomp_set_minmax(void* compressor, double data_min, double data_max);

/* Sparse chunk header definition */
#define IPCOMP_SPARSE_MAGIC   0x49504d53u /* 'IPMS' */
#define IPCOMP_SPARSE_VERSION 1
#define IPCOMP_SPARSE_MAX_NDIM 3

#define IPCOMP_SPARSE_FLAG_HAS_MASK     0x1u
#define IPCOMP_SPARSE_FLAG_HAS_DENSE    0x2u
#define IPCOMP_SPARSE_FLAG_HAS_MINMAX   0x4u
#define IPCOMP_SPARSE_FLAG_FULL_CHUNK   0x8u

typedef struct {
    uint32_t magic;
    uint16_t version;
    uint16_t flags;
    uint32_t ndims;
    uint64_t dims[IPCOMP_SPARSE_MAX_NDIM];
    uint64_t valid_count;
    double fill_value;
    double data_min;
    double data_max;
    uint64_t mask_bytes;
    uint64_t dense_elems;
    uint32_t dense_ndims;
    uint32_t reserved;
    uint64_t dense_dims[IPCOMP_SPARSE_MAX_NDIM];
} ipcomp_sparse_header;

static size_t ipcomp_sparse_header_size(void)
{
    return sizeof(ipcomp_sparse_header);
}

static void ipcomp_sparse_header_clear(ipcomp_sparse_header *hdr)
{
    if (hdr == NULL) return;
    memset(hdr, 0, sizeof(*hdr));
    hdr->magic = IPCOMP_SPARSE_MAGIC;
    hdr->version = IPCOMP_SPARSE_VERSION;
    hdr->ndims = IPCOMP_SPARSE_MAX_NDIM;
}

static void ipcomp_sparse_header_pack(unsigned char *dst,
                                      const ipcomp_sparse_header *hdr)
{
    if (dst == NULL || hdr == NULL) return;
    memcpy(dst, hdr, sizeof(*hdr));
}

static int ipcomp_sparse_header_unpack(const unsigned char *src,
                                       size_t avail,
                                       ipcomp_sparse_header *hdr)
{
    if (src == NULL || hdr == NULL) return NC_EINVAL;
    if (avail < sizeof(*hdr)) return NC_EINVAL;
    memcpy(hdr, src, sizeof(*hdr));
    if (hdr->magic != IPCOMP_SPARSE_MAGIC) return NC_EINVAL;
    if (hdr->version != IPCOMP_SPARSE_VERSION) return NC_EINVAL;
    if (hdr->ndims == 0 || hdr->ndims > IPCOMP_SPARSE_MAX_NDIM) return NC_EINVAL;
    if (hdr->dense_ndims > IPCOMP_SPARSE_MAX_NDIM) return NC_EINVAL;
    return NC_NOERR;
}

static double ipcomp_get_fill_value(MPI_Datatype dtype, const NCCHK_var_context *ctx)
{
    if (ctx != NULL && ctx->ipcomp_has_fill) {
        return ctx->ipcomp_fill_value;
    }
    return (dtype == MPI_FLOAT) ? (double)NC_FILL_FLOAT : NC_FILL_DOUBLE;
}

static size_t ipcomp_total_elems(int ndim, const int *dims)
{
    size_t total = 1;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return 0;
        total *= (size_t)dims[i];
    }
    return total;
}

static int ipcomp_is_valid_value(double val, double fill_value, int has_fill)
{
    if (isnan(val) || isinf(val)) return 0;
    if (has_fill && val == fill_value) return 0;
    return 1;
}

static void ipcomp_mask_set(unsigned char *mask, size_t idx)
{
    mask[idx >> 3] |= (unsigned char)(1u << (idx & 7));
}

static int ipcomp_mask_test(const unsigned char *mask, size_t idx)
{
    return (mask[idx >> 3] >> (idx & 7)) & 0x1;
}

static int ipcomp_prepare_full_chunk(const void *in, int ndim, const int *dims,
                                     MPI_Datatype dtype, NCCHK_var_context *ctx,
                                     ipcomp_sparse_header *hdr,
                                     unsigned char **mask_out, size_t *mask_bytes_out,
                                     void **chunk_out, size_t *chunk_bytes_out)
{
    if (in == NULL || hdr == NULL || mask_out == NULL ||
        mask_bytes_out == NULL || chunk_out == NULL || chunk_bytes_out == NULL) {
        return NC_EINVAL;
    }

    ipcomp_sparse_header_clear(hdr);
    hdr->ndims = (uint32_t)ndim;
    for (int i = 0; i < ndim && i < IPCOMP_SPARSE_MAX_NDIM; i++) {
        hdr->dims[i] = (uint64_t)dims[i];
    }
    size_t total_elems = ipcomp_total_elems(ndim, dims);
    if (total_elems == 0) {
        *mask_out = NULL;
        *mask_bytes_out = 0;
        *chunk_out = NULL;
        *chunk_bytes_out = 0;
        hdr->valid_count = 0;
        hdr->mask_bytes = 0;
        hdr->dense_elems = 0;
        return NC_NOERR;
    }

    size_t elem_size = (dtype == MPI_FLOAT) ? sizeof(float) : sizeof(double);
    double fill_value = ipcomp_get_fill_value(dtype, ctx);
    hdr->fill_value = fill_value;
    if (ctx != NULL && ctx->ipcomp_has_minmax) {
        hdr->data_min = ctx->ipcomp_data_min;
        hdr->data_max = ctx->ipcomp_data_max;
        hdr->flags |= IPCOMP_SPARSE_FLAG_HAS_MINMAX;
    }

    size_t mask_bytes = (total_elems + 7u) >> 3;
    unsigned char *mask = NULL;
    if (mask_bytes > 0) {
        mask = (unsigned char *)calloc(mask_bytes, 1);
        if (mask == NULL) {
            return NC_ENOMEM;
        }
    }

    size_t chunk_bytes = total_elems * elem_size;
    void *chunk = malloc(chunk_bytes);
    if (chunk == NULL) {
        free(mask);
        return NC_ENOMEM;
    }

    memcpy(chunk, in, chunk_bytes);

    uint64_t valid_count = 0;
    int has_fill = (ctx != NULL) ? ctx->ipcomp_has_fill : 1;
    if (dtype == MPI_FLOAT) {
        const float *src = (const float *)in;
        float *dst = (float *)chunk;
        for (size_t idx = 0; idx < total_elems; idx++) {
            double val = (double)src[idx];
            if (ipcomp_is_valid_value(val, fill_value, has_fill)) {
                if (mask_bytes > 0 && mask != NULL) ipcomp_mask_set(mask, idx);
                valid_count++;
            } else {
                dst[idx] = 0.0f;
            }
        }
    } else {
        const double *src = (const double *)in;
        double *dst = (double *)chunk;
        for (size_t idx = 0; idx < total_elems; idx++) {
            double val = src[idx];
            if (ipcomp_is_valid_value(val, fill_value, has_fill)) {
                if (mask_bytes > 0 && mask != NULL) ipcomp_mask_set(mask, idx);
                valid_count++;
            } else {
                dst[idx] = 0.0;
            }
        }
    }

    if (valid_count == total_elems) {
        if (mask != NULL) {
            free(mask);
            mask = NULL;
        }
        mask_bytes = 0;
        hdr->flags &= ~IPCOMP_SPARSE_FLAG_HAS_MASK;
    } else if (mask_bytes > 0) {
        hdr->flags |= IPCOMP_SPARSE_FLAG_HAS_MASK;
    }

    hdr->valid_count = valid_count;
    hdr->mask_bytes = (uint64_t)mask_bytes;
    hdr->dense_elems = (uint64_t)total_elems;
    hdr->dense_ndims = (uint32_t)ndim;
    for (int i = 0; i < IPCOMP_SPARSE_MAX_NDIM; i++) {
        if (i < ndim) {
            hdr->dense_dims[i] = (uint64_t)dims[i];
        } else {
            hdr->dense_dims[i] = 1;
        }
    }
    hdr->flags |= IPCOMP_SPARSE_FLAG_HAS_DENSE;
    hdr->flags |= IPCOMP_SPARSE_FLAG_FULL_CHUNK;

    *mask_out = mask;
    *mask_bytes_out = mask_bytes;
    *chunk_out = chunk;
    *chunk_bytes_out = chunk_bytes;
    return NC_NOERR;
}

static int ipcomp_scatter_sparse_chunk(const ipcomp_sparse_header *hdr,
                                       const unsigned char *mask,
                                       const void *dense,
                                       MPI_Datatype dtype,
                                       NCCHK_var_context *ctx,
                                       void *out_buf)
{
    if (hdr == NULL || out_buf == NULL) return NC_EINVAL;

    size_t total_elems = 1;
    int ndims = (int)hdr->ndims;
    for (int i = 0; i < ndims && i < IPCOMP_SPARSE_MAX_NDIM; i++) {
        total_elems *= (size_t)hdr->dims[i];
    }
    size_t elem_size = (dtype == MPI_FLOAT) ? sizeof(float) : sizeof(double);

    double fill_value = hdr->fill_value;
    if (ctx == NULL || !ctx->ipcomp_has_fill) {
        fill_value = (dtype == MPI_FLOAT) ? (double)NC_FILL_FLOAT : NC_FILL_DOUBLE;
    }

    if (hdr->flags & IPCOMP_SPARSE_FLAG_FULL_CHUNK) {
        if (dense != NULL) {
            memcpy(out_buf, dense, total_elems * elem_size);
        } else {
            if (dtype == MPI_FLOAT) {
                float *dst = (float *)out_buf;
                float fill = (float)fill_value;
                for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
            } else {
                double *dst = (double *)out_buf;
                double fill = fill_value;
                for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
            }
            return NC_NOERR;
        }
        if ((hdr->flags & IPCOMP_SPARSE_FLAG_HAS_MASK) && mask != NULL) {
            if (dtype == MPI_FLOAT) {
                float *dst = (float *)out_buf;
                float fill = (float)fill_value;
                for (size_t i = 0; i < total_elems; i++) {
                    if (!ipcomp_mask_test(mask, i)) {
                        dst[i] = fill;
                    }
                }
            } else {
                double *dst = (double *)out_buf;
                double fill = fill_value;
                for (size_t i = 0; i < total_elems; i++) {
                    if (!ipcomp_mask_test(mask, i)) {
                        dst[i] = fill;
                    }
                }
            }
        }
        return NC_NOERR;
    }

    if (!(hdr->flags & IPCOMP_SPARSE_FLAG_HAS_MASK)) {
        if (dense != NULL) {
            memcpy(out_buf, dense, total_elems * elem_size);
        } else {
            if (dtype == MPI_FLOAT) {
                float *dst = (float *)out_buf;
                float fill = (float)fill_value;
                for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
            } else {
                double *dst = (double *)out_buf;
                double fill = fill_value;
                for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
            }
        }
        return NC_NOERR;
    }

    if (!(hdr->flags & IPCOMP_SPARSE_FLAG_HAS_DENSE) || dense == NULL) {
        if (dtype == MPI_FLOAT) {
            float *dst = (float *)out_buf;
            float fill = (float)fill_value;
            for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
        } else {
            double *dst = (double *)out_buf;
            double fill = fill_value;
            for (size_t i = 0; i < total_elems; i++) dst[i] = fill;
        }
        return NC_NOERR;
    }

    size_t dense_idx = 0;
    if (dtype == MPI_FLOAT) {
        float *dst = (float *)out_buf;
        float fill = (float)fill_value;
        const float *src = (const float *)dense;
        for (size_t i = 0; i < total_elems; i++) {
            if (mask && ipcomp_mask_test(mask, i)) {
                dst[i] = src[dense_idx++];
            } else {
                dst[i] = fill;
            }
        }
    } else {
        double *dst = (double *)out_buf;
        double fill = fill_value;
        const double *src = (const double *)dense;
        for (size_t i = 0; i < total_elems; i++) {
            if (mask && ipcomp_mask_test(mask, i)) {
                dst[i] = src[dense_idx++];
            } else {
                dst[i] = fill;
            }
        }
    }
    return NC_NOERR;
}

static int ipcomp_compress_sparse_payload(const void *in, int ndim, const int *dims,
                                          MPI_Datatype dtype, NCCHK_var_context *ctx,
                                          unsigned char **payload, size_t *payload_size)
{
    int err;
    ipcomp_sparse_header hdr;
    unsigned char *mask = NULL;
    size_t mask_bytes = 0;
    void *prepared_chunk = NULL;
    size_t prepared_bytes = 0;

    if (payload == NULL || payload_size == NULL) return NC_EINVAL;
    *payload = NULL;
    *payload_size = 0;

    err = ipcomp_prepare_full_chunk(in, ndim, dims, dtype, ctx,
                                    &hdr, &mask, &mask_bytes,
                                    &prepared_chunk, &prepared_bytes);
    (void)prepared_bytes;
    if (err != NC_NOERR) {
        return err;
    }

    unsigned char *compressed_data = NULL;
    size_t compressed_size = 0;
    void *compressor = NULL;
    int ipcomp_dtype = mpi_to_ipcomp_type(dtype);
    if (ipcomp_dtype < 0) {
        free(mask);
        free(prepared_chunk);
        return NC_EINVAL;
    }

    if (hdr.flags & IPCOMP_SPARSE_FLAG_HAS_DENSE) {
        int dense_dims = (hdr.dense_ndims > 0) ? (int)hdr.dense_ndims : 1;
        if (dense_dims > IPCOMP_SPARSE_MAX_NDIM) dense_dims = IPCOMP_SPARSE_MAX_NDIM;
        int dense_shape[IPCOMP_SPARSE_MAX_NDIM];
        for (int i = 0; i < dense_dims; i++) {
            dense_shape[i] = (int)hdr.dense_dims[i];
            if (dense_shape[i] <= 0) {
                dense_shape[i] = 1;
            }
        }
        size_t block_size = (ctx != NULL && ctx->ipcomp_block_size > 0)
                                ? ctx->ipcomp_block_size
                                : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;
        int layers = ipcomp_resolve_layers(dtype, ctx);
        int interp_op = (ctx != NULL && ctx->ipcomp_interp >= 0) ? ctx->ipcomp_interp : 1;
        int direction_op = (ctx != NULL) ? ctx->ipcomp_direction : 0;
        int level_progressive = (ctx != NULL && ctx->ipcomp_level_progressive >= 0)
                                    ? ctx->ipcomp_level_progressive : 0;

        compressor = ipcomp_create_compressor(dense_dims, dense_shape,
                                              interp_op, direction_op,
                                              layers, block_size,
                                              level_progressive);
        if (compressor == NULL) {
            free(mask);
            free(prepared_chunk);
            return NC_ENOMEM;
        }

        if (ctx != NULL && ctx->ipcomp_data_range > 0.0) {
            ipcomp_set_range(compressor, ctx->ipcomp_data_range);
        }
        if (ctx != NULL && ctx->ipcomp_has_minmax) {
            ipcomp_set_minmax(compressor, ctx->ipcomp_data_min, ctx->ipcomp_data_max);
        }

        if (ipcomp_setup_layers(compressor, prepared_chunk, ipcomp_dtype) != 0) {
            ipcomp_destroy_compressor(compressor);
            free(mask);
            free(prepared_chunk);
            return NC_EINVAL;
        }

        compressed_data = ipcomp_compress(compressor, prepared_chunk, ipcomp_dtype,
                                          &compressed_size, dense_dims, dense_shape);
        if (compressed_data == NULL) {
            ipcomp_destroy_compressor(compressor);
            free(mask);
            free(prepared_chunk);
            return NC_EINVAL;
        }
    }

    size_t header_size = ipcomp_sparse_header_size();
    if (ctx != NULL) {
        ctx->ipcomp_header_size = header_size;
    }
    size_t total_size = header_size + mask_bytes + compressed_size;
    if (total_size == 0) total_size = 1;
    unsigned char *buf = (unsigned char *)malloc(total_size);
    if (buf == NULL) {
        if (compressor != NULL) {
            ipcomp_free_buffer(compressed_data);
            ipcomp_destroy_compressor(compressor);
        }
        free(mask);
        free(prepared_chunk);
        return NC_ENOMEM;
    }

    ipcomp_sparse_header_pack(buf, &hdr);
    size_t offset = header_size;
    if (mask_bytes && mask != NULL) {
        memcpy(buf + offset, mask, mask_bytes);
        offset += mask_bytes;
    }
    if (compressed_size && compressed_data != NULL) {
        memcpy(buf + offset, compressed_data, compressed_size);
    }

    if (compressor != NULL) {
        ipcomp_free_buffer(compressed_data);
        ipcomp_destroy_compressor(compressor);
    }
    free(mask);
    free(prepared_chunk);

    *payload = buf;
    *payload_size = header_size + mask_bytes + compressed_size;
    return NC_NOERR;
}

static int ipcomp_sparse_payload_unpack(const unsigned char *payload,
                                        size_t payload_len,
                                        ipcomp_sparse_header *hdr,
                                        const unsigned char **mask_out,
                                        const unsigned char **dense_payload_out,
                                        size_t *dense_payload_size_out)
{
    if (payload == NULL || hdr == NULL) return NC_EINVAL;

    size_t header_size = ipcomp_sparse_header_size();
    int err = ipcomp_sparse_header_unpack(payload, payload_len, hdr);
    if (err != NC_NOERR) return err;
    if (payload_len < header_size + hdr->mask_bytes) return NC_EINVAL;

    if (mask_out != NULL) {
        if (hdr->flags & IPCOMP_SPARSE_FLAG_HAS_MASK) {
            *mask_out = payload + header_size;
        } else {
            *mask_out = NULL;
        }
    }

    if (dense_payload_out != NULL || dense_payload_size_out != NULL) {
        const unsigned char *dense_payload = payload + header_size + hdr->mask_bytes;
        if (dense_payload_out != NULL) {
            *dense_payload_out = dense_payload;
        }
        if (dense_payload_size_out != NULL) {
            if (payload_len > header_size + hdr->mask_bytes) {
                *dense_payload_size_out = payload_len - header_size - hdr->mask_bytes;
            } else {
                *dense_payload_size_out = 0;
            }
        }
    }
    return NC_NOERR;
}

static int ipcomp_decode_dense_payload(const ipcomp_sparse_header *hdr,
                                       const unsigned char *dense_payload,
                                       size_t dense_payload_size,
                                       MPI_Datatype dtype,
                                       NCCHK_var_context *ctx,
                                       double *param_ptr,
                                       int param_count,
                                       int use_bitrate,
                                       void **dense_out)
{
    if (hdr == NULL || dense_out == NULL) return NC_EINVAL;
    *dense_out = NULL;

    if (!(hdr->flags & IPCOMP_SPARSE_FLAG_HAS_DENSE) || hdr->dense_elems == 0) {
        return NC_NOERR;
    }
    if (dense_payload == NULL || dense_payload_size == 0) {
        return NC_EINVAL;
    }

    int ipcomp_dtype = mpi_to_ipcomp_type(dtype);
    if (ipcomp_dtype < 0) {
        return NC_EINVAL;
    }

    int dense_dims = (hdr->dense_ndims > 0) ? (int)hdr->dense_ndims : 1;
    if (dense_dims > IPCOMP_SPARSE_MAX_NDIM) dense_dims = IPCOMP_SPARSE_MAX_NDIM;
    int dense_shape[IPCOMP_SPARSE_MAX_NDIM];
    for (int i = 0; i < dense_dims; i++) {
        dense_shape[i] = (int)hdr->dense_dims[i];
        if (dense_shape[i] <= 0) {
            dense_shape[i] = 1;
        }
    }
    size_t block_size = (ctx != NULL && ctx->ipcomp_block_size > 0)
                            ? ctx->ipcomp_block_size
                            : (size_t)IPCOMP_DEFAULT_BLOCK_SIZE;
    int layers = ipcomp_resolve_layers(dtype, ctx);
    int interp_op = (ctx != NULL && ctx->ipcomp_interp >= 0) ? ctx->ipcomp_interp : 1;
    int direction_op = (ctx != NULL) ? ctx->ipcomp_direction : 0;
    int level_progressive = (ctx != NULL && ctx->ipcomp_level_progressive >= 0)
                                ? ctx->ipcomp_level_progressive : 0;

    void *compressor = ipcomp_create_compressor(dense_dims, dense_shape,
                                               interp_op, direction_op,
                                               layers, block_size,
                                               level_progressive);
    if (compressor == NULL) {
        return NC_ENOMEM;
    }
    if (ctx != NULL && ctx->ipcomp_data_range > 0.0) {
        ipcomp_set_range(compressor, ctx->ipcomp_data_range);
    }
    if (ctx != NULL && ctx->ipcomp_has_minmax) {
        ipcomp_set_minmax(compressor, ctx->ipcomp_data_min, ctx->ipcomp_data_max);
    }

    void *dense = NULL;
    if (use_bitrate) {
        dense = ipcomp_decompress_bitrate(compressor, dense_payload,
                                          ipcomp_dtype, param_ptr, param_count);
    } else {
        dense = ipcomp_decompress_error(compressor, dense_payload,
                                        ipcomp_dtype, param_ptr, param_count);
    }
    ipcomp_destroy_compressor(compressor);
    if (dense == NULL) {
        return NC_EINVAL;
    }
    *dense_out = dense;
    return NC_NOERR;
}

static int ipcomp_decompress_sparse_payload(const unsigned char *payload,
                                            size_t payload_len,
                                            MPI_Datatype dtype,
                                            NCCHK_var_context *ctx,
                                            void *out_buf)
{
    if (payload == NULL || out_buf == NULL) return NC_EINVAL;

    ipcomp_sparse_header hdr;
    const unsigned char *mask = NULL;
    const unsigned char *dense_payload = NULL;
    size_t dense_payload_size = 0;
    int err = ipcomp_sparse_payload_unpack(payload, payload_len, &hdr,
                                           &mask, &dense_payload, &dense_payload_size);
    if (err != NC_NOERR) return err;

    void *dense = NULL;
    err = ipcomp_decode_dense_payload(&hdr, dense_payload, dense_payload_size,
                                      dtype, ctx, NULL, 0, 0, &dense);
    if (err != NC_NOERR) {
        return err;
    }

    err = ipcomp_scatter_sparse_chunk(&hdr, mask, dense, dtype, ctx, out_buf);
    if (dense != NULL) {
        ipcomp_free_buffer(dense);
    }
    return err;
}


/* C wrapper functions for IPComp C++ interface */
extern void* ipcomp_create_compressor(int ndim, const int* dims, int interp_op, int direction_op, 
                                     int layers, size_t block_size, int level_progressive);
extern void ipcomp_destroy_compressor(void* compressor);
extern int ipcomp_setup(void* compressor, const void* data, int data_type);
extern int ipcomp_setup_layers(void* compressor, const void* data, int data_type);
extern unsigned char* ipcomp_compress(void* compressor, const void* data, int data_type, 
                                     size_t* compressed_size, int ndim, const int* dims);
extern void* ipcomp_decompress_error(void* compressor, const unsigned char* compressed_data,
                                   int data_type, const double* target_rel_ebs,
                                   int num_target_ebs);
extern void* ipcomp_decompress_bitrate(void* compressor, const unsigned char* compressed_data,
                                     int data_type, const double* target_bitrates,
                                     int num_target_bitrates);
extern void ipcomp_free_buffer(void* buffer);
extern int ipcomp_set_range(void* compressor, double data_range);
extern int ipcomp_set_minmax(void* compressor, double data_min, double data_max);

#ifdef __cplusplus
}
#endif

/* Data type mapping from MPI to IPComp internal types */
static int mpi_to_ipcomp_type(MPI_Datatype dtype) {
    if (dtype == MPI_FLOAT) {
        return 0; /* FLOAT */
    }
    else if (dtype == MPI_DOUBLE) {
        return 1; /* DOUBLE */
    }
    /* Add more type mappings as needed */
    return -1;
}

static int ipcomp_default_layers(MPI_Datatype dtype) {
    if (dtype == MPI_DOUBLE) {
        return 9;
    }
    return 1;
}

static int ipcomp_resolve_layers(MPI_Datatype dtype, NCCHK_var_context* ctx) {
    int layers = ipcomp_default_layers(dtype);
    if (ctx != NULL && ctx->ipcomp_layers > 0) {
        layers = ctx->ipcomp_layers;
    }
    if (layers <= 0) {
        layers = ipcomp_default_layers(dtype);
    }
    return layers;
}

int ncchk_ipcomp_init(MPI_Info info) {
    /* IPComp initialization if needed */
    return NC_NOERR;
}

int ncchk_ipcomp_finalize() {
    /* IPComp cleanup if needed */
    return NC_NOERR;
}

/* Return an estimated compressed data size
 * For IPComp progressive compression, we cannot easily estimate the size
 */
int ncchk_ipcomp_inq_cpsize(void *in, int in_len, int *out_len, int ndim, int *dims, 
                           MPI_Datatype dtype, NCCHK_var_context* ctx) {
    return NC_ENOTSUPPORT;  /* IPComp has no size estimation */
}

/* Compress data using IPComp progressive compression */
int ncchk_ipcomp_compress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, 
                         MPI_Datatype dtype, NCCHK_var_context* ctx) {
    if (out == NULL || out_len == NULL) return NC_EINVAL;

    unsigned char *payload = NULL;
    size_t payload_size = 0;
    int err = ipcomp_compress_sparse_payload(in, ndim, dims, dtype, ctx,
                                             &payload, &payload_size);
    if (err != NC_NOERR) {
        return err;
    }
    if (*out_len < (int)payload_size) {
        free(payload);
        return NC_ENOMEM;
    }
    memcpy(out, payload, payload_size);
    *out_len = (int)payload_size;
    free(payload);
    return NC_NOERR;
}

/* Compress data and allocate output buffer */
int ncchk_ipcomp_compress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, 
                               MPI_Datatype dtype, NCCHK_var_context* ctx) {
    if (out == NULL || dims == NULL) return NC_EINVAL;
    if (ndim <= 0 || in_len < 0) return NC_EINVAL;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] < 0) return NC_EINVAL;
    }

    unsigned char *payload = NULL;
    size_t payload_size = 0;

    /* Reuse sparse payload path so NaN/_FillValue are zeroed with a mask retained. */
    int err = ipcomp_compress_sparse_payload(in, ndim, dims, dtype, ctx,
                                             &payload, &payload_size);
    if (err != NC_NOERR) {
        return err;
    }

    if (payload_size > (size_t)INT_MAX) {
        free(payload);
        return NC_EINVAL;
    }

    *out = payload;
    if (out_len != NULL) {
        *out_len = (int)payload_size;
    }
    return NC_NOERR;
}

/* Return an estimated decompressed data size */
int ncchk_ipcomp_inq_dcsize(void *in, int in_len, int *out_len, int ndim, int *dims, 
                           MPI_Datatype dtype, NCCHK_var_context* ctx) {
    return NC_ENOTSUPPORT;  /* IPComp has no size estimation */
}

/* Decompress data using IPComp - default mode (full quality) */
int ncchk_ipcomp_decompress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, 
                           MPI_Datatype dtype, NCCHK_var_context* ctx) {
    if (in == NULL || out == NULL || dims == NULL) return NC_EINVAL;
    if (in_len < 0 || ndim <= 0) return NC_EINVAL;

    int elem_size = 0;
    MPI_Type_size(dtype, &elem_size);
    if (elem_size <= 0) return NC_EBADTYPE;

    int outsize = elem_size;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return NC_EINVAL;
        outsize *= dims[i];
    }

    if (out_len != NULL && *out_len < outsize) {
        return NC_ENOMEM;
    }

    int err = ipcomp_decompress_sparse_payload((const unsigned char *)in,
                                               (size_t)in_len, dtype, ctx, out);
    if (err != NC_NOERR) {
        return err;
    }

    if (out_len != NULL) {
        *out_len = outsize;
    }
    return NC_NOERR;
}

/* Decompress data and allocate output buffer */
int ncchk_ipcomp_decompress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, 
                                 MPI_Datatype dtype, NCCHK_var_context* ctx) {
    if (in == NULL || out == NULL || dims == NULL) return NC_EINVAL;
    if (in_len < 0 || ndim <= 0) return NC_EINVAL;

    int elem_size = 0;
    MPI_Type_size(dtype, &elem_size);
    if (elem_size <= 0) return NC_EBADTYPE;

    int outsize = elem_size;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) return NC_EINVAL;
        outsize *= dims[i];
    }

    void *buf = malloc(outsize > 0 ? outsize : 1);
    if (buf == NULL) {
        return NC_ENOMEM;
    }

    int err = ipcomp_decompress_sparse_payload((const unsigned char *)in,
                                               (size_t)in_len, dtype, ctx, buf);
    if (err != NC_NOERR) {
        free(buf);
        return err;
    }

    *out = buf;
    if (out_len != NULL) {
        *out_len = outsize;
    }
    return NC_NOERR;
}

/* IPComp filter driver structure */
static NCCHK_filter ncchk_driver_ipcomp = {
    ncchk_ipcomp_init,
    ncchk_ipcomp_finalize,
    ncchk_ipcomp_inq_cpsize,
    ncchk_ipcomp_compress,
    ncchk_ipcomp_compress_alloc,
    ncchk_ipcomp_inq_dcsize,
    ncchk_ipcomp_decompress,
    ncchk_ipcomp_decompress_alloc
};

/* Progressive decompression with error bound constraint */
int ncchk_ipcomp_decompress_progressive_error(void *in, int in_len, void *out, int *out_len, 
                                             int ndim, int *dims, MPI_Datatype dtype, 
                                             double target_rel_eb, NCCHK_var_context* ctx) {
    int err = NC_NOERR;
    int elem_size = 0;
    int outsize;
    
    if (in == NULL || out == NULL || dims == NULL) {
        return NC_EINVAL;
    }
    if (in_len < 0 || ndim <= 0) {
        return NC_EINVAL;
    }
    
    double data_range = (ctx != NULL) ? ctx->ipcomp_data_range : 0.0;
    double *rel_eb_ptr = NULL;
    int rel_eb_count = 0;
    double rel_eb_single = 0.0;
    int rel_eb_malloc = 0;
    
    MPI_Type_size(dtype, &elem_size);
    if (elem_size <= 0) {
        return NC_EBADTYPE;
    }
    
    outsize = elem_size;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) {
            return NC_EINVAL;
        }
        if (outsize > 0 && dims[i] > INT_MAX / outsize) {
            return NC_EINVAL;
        }
        outsize *= dims[i];
    }
    
    if (target_rel_eb > 0.0) {
        rel_eb_single = target_rel_eb;
        rel_eb_ptr = &rel_eb_single;
        rel_eb_count = 1;
    } else if (ctx != NULL && ctx->ipcomp_ebs != NULL && ctx->ipcomp_num_ebs > 0) {
        rel_eb_count = ctx->ipcomp_num_ebs;
        rel_eb_ptr = (double*)malloc((size_t)rel_eb_count * sizeof(double));
        if (rel_eb_ptr == NULL) {
            return NC_ENOMEM;
        }
        rel_eb_malloc = 1;
        for (int i = 0; i < rel_eb_count; i++) {
            double val = ctx->ipcomp_ebs[i];
            if (val > 1.0 && data_range > 0.0) {
                val /= data_range;
            }
            rel_eb_ptr[i] = val;
        }
    }

    ipcomp_sparse_header hdr;
    const unsigned char *mask = NULL;
    const unsigned char *dense_payload = NULL;
    size_t dense_payload_size = 0;
    err = ipcomp_sparse_payload_unpack((const unsigned char *)in, (size_t)in_len,
                                       &hdr, &mask, &dense_payload, &dense_payload_size);
    if (err != NC_NOERR) {
        goto cleanup;
    }

    void *dense = NULL;
    err = ipcomp_decode_dense_payload(&hdr, dense_payload, dense_payload_size,
                                      dtype, ctx, rel_eb_ptr, rel_eb_count, 0, &dense);
    if (err != NC_NOERR) {
        goto cleanup_dense;
    }

    if (out_len != NULL) {
        if (*out_len < outsize) {
            err = NC_ENOMEM;
            goto cleanup_dense;
        }
        *out_len = outsize;
    }

    err = ipcomp_scatter_sparse_chunk(&hdr, mask, dense, dtype, ctx, out);

cleanup_dense:
    if (dense != NULL) {
        ipcomp_free_buffer(dense);
    }
cleanup:
    if (rel_eb_malloc && rel_eb_ptr != NULL) {
        free(rel_eb_ptr);
    }
    return err;
}

/* Minimum number of bytes we should budget for bitrate-based progressive decompression
 * to avoid triggering underflow inside the IPComp library when it subtracts header sizes.
 */
#define IPCOMP_MIN_BITRATE_BUDGET_BYTES 512.0

/* Progressive decompression with bitrate constraint */
int ncchk_ipcomp_decompress_progressive_bitrate(void *in, int in_len, void *out, int *out_len, 
                                               int ndim, int *dims, MPI_Datatype dtype, 
                                               double max_bitrate, NCCHK_var_context* ctx) {
    int err = NC_NOERR;
    int elem_size = 0;
    int outsize;
    double bitrate_single = 0.0;
    double *bitrate_ptr = NULL;
    int bitrate_count = 0;
    
    if (in == NULL || out == NULL || dims == NULL) {
        return NC_EINVAL;
    }
    if (in_len < 0 || ndim <= 0) {
        return NC_EINVAL;
    }
    
    fprintf(stderr, "[DEBUG] decompress_progressive_bitrate: data_range=%g, max_bitrate=%g\n", 
            ctx ? ctx->ipcomp_data_range : 0.0, max_bitrate);
    if (ctx != NULL && ctx->ipcomp_ebs != NULL) {
        fprintf(stderr, "[DEBUG] Actual ebs from compression: ");
        for (int i = 0; i < ctx->ipcomp_num_ebs; i++) {
            fprintf(stderr, "%g ", ctx->ipcomp_ebs[i]);
        }
        fprintf(stderr, "\n");
    }
    
    MPI_Type_size(dtype, &elem_size);
    if (elem_size <= 0) {
        return NC_EBADTYPE;
    }
    
    outsize = elem_size;
    for (int i = 0; i < ndim; i++) {
        if (dims[i] <= 0) {
            return NC_EINVAL;
        }
        if (outsize > 0 && dims[i] > INT_MAX / outsize) {
            return NC_EINVAL;
        }
        outsize *= dims[i];
    }
    
    if (in_len <= 0) {
        fprintf(stderr, "[ERROR] Invalid in_len: %d\n", in_len);
        return NC_EINVAL;
    }
    
    fprintf(stderr, "[DEBUG] Bitrate decompress: in_len=%d, first 16 bytes: ", in_len);
    for (int i = 0; i < 16 && i < in_len; i++) {
        fprintf(stderr, "%02x ", ((unsigned char*)in)[i]);
    }
    fprintf(stderr, "\n");
    
    if (max_bitrate > 0.0) {
        double dtype_bits = (double)elem_size * 8.0;
        double total_bytes = (double)outsize;
        double bits_per_value = 0.0;

        if (max_bitrate < 1.0) {
            double ratio = max_bitrate;
            if (ratio > 1.0) ratio = 1.0;
            if (ratio < 0.0) ratio = 0.0;
            bits_per_value = ratio * dtype_bits;
        } else {
            double ratio = 0.0;
            if (total_bytes > 0.0) {
                ratio = max_bitrate / total_bytes;
            }
            if (ratio > 1.0) ratio = 1.0;
            if (ratio < 0.0) ratio = 0.0;
            bits_per_value = ratio * dtype_bits;
        }

        if (bits_per_value <= 0.0) {
            bits_per_value = dtype_bits;
        } else if (bits_per_value > dtype_bits) {
            bits_per_value = dtype_bits;
        }

        if (total_bytes > 0.0) {
            double min_budget_bytes = IPCOMP_MIN_BITRATE_BUDGET_BYTES;
            if (min_budget_bytes > total_bytes) {
                min_budget_bytes = total_bytes;
            }
            double current_budget_bytes = (bits_per_value / dtype_bits) * total_bytes;
            if (current_budget_bytes < min_budget_bytes) {
                bits_per_value = (min_budget_bytes / total_bytes) * dtype_bits;
            }
            if (bits_per_value > dtype_bits) {
                bits_per_value = dtype_bits;
            }
        }

        bitrate_single = bits_per_value;
        bitrate_ptr = &bitrate_single;
        bitrate_count = 1;
    }

    ipcomp_sparse_header hdr;
    const unsigned char *mask = NULL;
    const unsigned char *dense_payload = NULL;
    size_t dense_payload_size = 0;
    err = ipcomp_sparse_payload_unpack((const unsigned char *)in, (size_t)in_len,
                                       &hdr, &mask, &dense_payload, &dense_payload_size);
    if (err != NC_NOERR) {
        return err;
    }

    void *dense = NULL;
    err = ipcomp_decode_dense_payload(&hdr, dense_payload, dense_payload_size,
                                      dtype, ctx, bitrate_ptr, bitrate_count, 1, &dense);
    if (err != NC_NOERR) {
        if (dense != NULL) {
            ipcomp_free_buffer(dense);
        }
        return err;
    }

    if (out_len != NULL) {
        if (*out_len < outsize) {
            if (dense != NULL) ipcomp_free_buffer(dense);
            return NC_ENOMEM;
        }
        *out_len = outsize;
    }

    err = ipcomp_scatter_sparse_chunk(&hdr, mask, dense, dtype, ctx, out);
    if (dense != NULL) {
        ipcomp_free_buffer(dense);
    }
    return err;
}

NCCHK_filter* ncchk_ipcomp_inq_driver(void) {
    return &ncchk_driver_ipcomp;
}
