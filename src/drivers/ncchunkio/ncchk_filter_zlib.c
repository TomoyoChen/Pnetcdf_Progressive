/*
 *  Copyright (C) 2017, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

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

#include <zlib.h>

int ncchk_zlib_init(MPI_Info info) {
    return NC_NOERR;
}

int ncchk_zlib_finalize() {
    return NC_NOERR;
}

/* Return an estimated compressed data size
 * Actual compressed size should not exceed the estimation
 */
int ncchk_zlib_inq_cpsize(void *in, int in_len, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    return NC_ENOTSUPPORT;  // Zlib has no size estimation
}

/* If out_len is large enough, compress the data at in and save it to out. out_len is set to actual compressed data size
 * If out_len is NULL, we assume out is large enough for compressed data
 */
int ncchk_zlib_compress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    int err=NC_NOERR;
    int compression_level;

    /* Get compression level from context */
    if (ctx != NULL && ctx->zlib_level >= 1 && ctx->zlib_level <= 9) {
        compression_level = ctx->zlib_level;
    } else {
        compression_level = Z_DEFAULT_COMPRESSION;  /* Default compression level */
    }

    // zlib struct
    z_stream defstream;
    defstream.zalloc = Z_NULL;
    defstream.zfree = Z_NULL;
    defstream.opaque = Z_NULL;
    defstream.avail_in = (uInt)(in_len); // input size
    defstream.next_in = (Bytef*)in; // input
    if (out_len != NULL){
        defstream.avail_out = (uInt)(*out_len); // output buffer size
    }
    else{
        defstream.avail_out = (uInt)1000000000; // Assume it is large enough
    }
    defstream.next_out = (Bytef *)out; // output buffer

    // the actual compression work.
    err = deflateInit(&defstream, compression_level);
    if (err != Z_OK){
        printf("deflateInit fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = deflate(&defstream, Z_FINISH);
    if (err != Z_STREAM_END){
        printf("deflate fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    err = deflateEnd(&defstream);
    if (err != Z_OK){
        printf("deflateEnd fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // If buffer not large enough
    if (defstream.avail_in > 0){
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    // Size of comrpessed data
    if (out_len != NULL){
        *out_len = defstream.total_out;
    }

    return NC_NOERR;
}

/* Compress the data at in and save it to a newly allocated buffer at out. out_len is set to actual compressed data size
 * The caller is responsible to free the buffer
 * If out_len is not NULL, it will be set to buffer size allocated
 */
int ncchk_zlib_compress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    int err=NC_NOERR;
    int bsize; // Start by 1/8 of the in_len
    char *buf;
    int compression_level;

    /* Get compression level from context */
    if (ctx != NULL && ctx->zlib_level >= 1 && ctx->zlib_level <= 9) {
        compression_level = ctx->zlib_level;
    } else {
        compression_level = Z_DEFAULT_COMPRESSION;  /* Default compression level */
    }

    bsize = in_len >> 3;
    if (bsize < 6){
        bsize = 6;
    }
    buf = (char*)malloc(bsize); 

    // zlib struct
    z_stream defstream;
    defstream.zalloc = Z_NULL;
    defstream.zfree = Z_NULL;
    defstream.opaque = Z_NULL;
    defstream.avail_in = (uInt)(in_len); // input size
    defstream.next_in = (Bytef*)in; // input
    defstream.avail_out = (uInt)(bsize); // output buffer size
    defstream.next_out = (Bytef *)buf; // output buffer

    // Initialize deflat stream
    err = deflateInit(&defstream, compression_level);
    if (err != Z_OK){
        printf("deflateInit fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // The actual compression work
    err = Z_OK;
    while (err != Z_STREAM_END){
        // Compress data
        err = deflate(&defstream, Z_NO_FLUSH | Z_FINISH);
        // Check if buffer is lage enough
        if (err != Z_STREAM_END){
            // Enlarge buffer
            buf = (char*)realloc(buf, bsize << 1); 

            // Reset buffer info in stream
            defstream.next_out = (Bytef *)(buf + bsize);
            defstream.avail_out = bsize;

            // Reocrd new buffer size
            bsize <<= 1;
        }
    }

    // Finalize deflat stream
    err = deflateEnd(&defstream);
    if (err != Z_OK){
        printf("deflateEnd fail: %d: %s\n", err, defstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // Size of comrpessed data
    if (out_len != NULL){
        *out_len = defstream.total_out;

        char *env_str;
        if ((env_str = getenv("PNETCDF_COMPRESS_VERBOSE")) != NULL) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            printf("rank %d (%s at %d) compress data size %d into size %d\n",
                    rank,__func__,__LINE__,in_len,*out_len);
        }
    }

    // Compressed data
    *out = buf;

    return NC_NOERR;
}

/* Return an estimated decompressed data size
 * Actual decompressed size should not exceed the estimation
 */
int ncchk_zlib_inq_dcsize(void *in, int in_len, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    return NC_ENOTSUPPORT;  // Zlib has no size estimation
}

/* If out_len is large enough, decompress the data at in and save it to out. out_len is set to actual decompressed size
 * If out_len is NULL, we assume out is large enough for decompressed data
 */
int ncchk_zlib_decompress(void *in, int in_len, void *out, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    int err=NC_NOERR;
    int zlib_err;

    // zlib struct
    z_stream infstream;
    infstream.zalloc = Z_NULL;
    infstream.zfree = Z_NULL;
    infstream.opaque = Z_NULL;
    infstream.avail_in = (unsigned long) in_len; // input size
    infstream.next_in = (Bytef *)in; // input
    if (out_len != NULL){
        infstream.avail_out = (uInt)(*out_len); // output buffer size
    }
    else{
        infstream.avail_out = (uInt)1000000000; // Assume it is large enough
    }
    infstream.next_out = (Bytef *)out; // buffer size
    
    // the actual decompression work.
    zlib_err = inflateInit(&infstream);
    if (zlib_err != Z_OK){
        printf("inflateInit fail: %d: %s\n", zlib_err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }
    
    // Use Z_NO_FLUSH instead of Z_FINISH for better compatibility
    zlib_err = inflate(&infstream, Z_NO_FLUSH);
    if (zlib_err != Z_STREAM_END && zlib_err != Z_OK) {
        printf("inflate fail: %d: %s\n", zlib_err, infstream.msg);
        // Try to continue with remaining data
        if (zlib_err == Z_BUF_ERROR && infstream.avail_out == 0) {
            // Output buffer full but more data available - this might be OK
            printf("Warning: Output buffer may be too small for decompression\n");
        } else {
            inflateEnd(&infstream);
            DEBUG_RETURN_ERROR(NC_EIO)
        }
    }
    
    // Complete the decompression if not finished
    if (zlib_err != Z_STREAM_END) {
        zlib_err = inflate(&infstream, Z_FINISH);
        if (zlib_err != Z_STREAM_END) {
            printf("inflate finish fail: %d: %s\n", zlib_err, infstream.msg);
            inflateEnd(&infstream);
            DEBUG_RETURN_ERROR(NC_EIO)
        }
    }
    
    zlib_err = inflateEnd(&infstream);
    if (zlib_err != Z_OK){
        printf("inflateEnd fail: %d: %s\n", zlib_err, infstream.msg);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // If buffer not large enough
    if (infstream.avail_in > 0){
        printf("Warning: %lu bytes of input data not processed\n", infstream.avail_in);
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    // Size of decomrpessed data
    if (out_len != NULL){
        *out_len = infstream.total_out;

        char *env_str;
        if ((env_str = getenv("PNETCDF_COMPRESS_VERBOSE")) != NULL) {
            int rank;
            MPI_Comm_rank(MPI_COMM_WORLD,&rank);
            printf("rank %d (%s at %d) decompress data size %d into size %d\n",
                    rank,__func__,__LINE__,in_len,*out_len);
        }
    }

    return NC_NOERR;
}

/* Decompress the data at in and save it to a newly allocated buffer at out. out_len is set to actual decompressed data size
 * The caller is responsible to free the buffer
 * If out_len is not NULL, it will be set to buffer size allocated
 */
int ncchk_zlib_decompress_alloc(void *in, int in_len, void **out, int *out_len, int ndim, int *dims, MPI_Datatype dtype, NCCHK_var_context* ctx) {
    int err=NC_NOERR;
    int bsize = in_len << 3; // Start by 8 times of the in_len (increased from 2x)
    char *buf;
    int zlib_err;

    if (bsize < 1024) bsize = 1024; // Minimum buffer size
    buf = (char*)malloc(bsize); 
    if (buf == NULL) {
        DEBUG_RETURN_ERROR(NC_ENOMEM)
    }

    // zlib struct
    z_stream infstream;
    infstream.zalloc = Z_NULL;
    infstream.zfree = Z_NULL;
    infstream.opaque = Z_NULL;
    infstream.avail_in = (uInt)(in_len); // input size
    infstream.next_in = (Bytef*)in; // input
    infstream.avail_out = (uInt)(bsize); // output buffer size
    infstream.next_out = (Bytef *)buf; // output buffer

    // Initialize inflate stream
    zlib_err = inflateInit(&infstream);
    if (zlib_err != Z_OK){
        printf("inflateInit fail: %d: %s\n", zlib_err, infstream.msg);
        free(buf);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // The actual decompression work
    zlib_err = Z_OK;
    int attempts = 0;
    const int max_attempts = 10; // Prevent infinite loop
    
    while (zlib_err != Z_STREAM_END && attempts < max_attempts){
        // Decompress data
        zlib_err = inflate(&infstream, Z_NO_FLUSH);
        
        // Check different error conditions
        if (zlib_err == Z_BUF_ERROR) {
            // Buffer too small, enlarge it
            if (infstream.avail_out == 0) {
                // Output buffer full
                size_t used = bsize - infstream.avail_out;
                buf = (char*)realloc(buf, bsize << 1);
                if (buf == NULL) {
                    inflateEnd(&infstream);
                    DEBUG_RETURN_ERROR(NC_ENOMEM)
                }

                // Reset buffer info in stream
                infstream.next_out = (Bytef *)(buf + used);
                infstream.avail_out = bsize;

                // Record new buffer size
                bsize <<= 1;
            } else {
                // Input buffer issue
                printf("inflate buffer error: avail_in=%d, avail_out=%d\n", 
                       infstream.avail_in, infstream.avail_out);
                break;
            }
        } else if (zlib_err != Z_OK && zlib_err != Z_STREAM_END) {
            printf("inflate error: %d: %s\n", zlib_err, infstream.msg);
            break;
        }
        
        attempts++;
    }
    
    // If we didn't finish successfully, try Z_FINISH
    if (zlib_err != Z_STREAM_END && zlib_err == Z_OK) {
        zlib_err = inflate(&infstream, Z_FINISH);
    }

    // Check final result
    if (zlib_err != Z_STREAM_END) {
        printf("decompression failed with error: %d: %s\n", zlib_err, infstream.msg);
        inflateEnd(&infstream);
        free(buf);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // Finalize inflate stream
    zlib_err = inflateEnd(&infstream);
    if (zlib_err != Z_OK){
        printf("inflateEnd fail: %d: %s\n", zlib_err, infstream.msg);
        free(buf);
        DEBUG_RETURN_ERROR(NC_EIO)
    }

    // Size of decompressed data
    if (out_len != NULL){
        *out_len = infstream.total_out;
    }

    // Decompressed data
    *out = buf;

    return NC_NOERR;
}

static NCCHK_filter ncchk_driver_zlib = {
    ncchk_zlib_init,
    ncchk_zlib_finalize,
    ncchk_zlib_inq_cpsize,
    ncchk_zlib_compress,
    ncchk_zlib_compress_alloc,
    ncchk_zlib_inq_dcsize,
    ncchk_zlib_decompress,
    ncchk_zlib_decompress_alloc
};

NCCHK_filter* ncchk_zlib_inq_driver(void) {
    return &ncchk_driver_zlib;
}

