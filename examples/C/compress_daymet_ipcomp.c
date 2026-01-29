#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <pnetcdf.h>
#include <float.h>
#include <math.h>

#define CHUNK_T 1
#define CHUNK_X 7814
#define CHUNK_Y 8075

#define MIN_RANGE_EPS 1e-9

#define CHECK_ERR(call)                                                              \
    do {                                                                             \
        int _status = (call);                                                        \
        if (_status != NC_NOERR) {                                                   \
            fprintf(stderr, "[%d] PnetCDF error at %s:%d: %s\n",                     \
                    g_rank, __FILE__, __LINE__, ncmpi_strerror(_status));            \
            MPI_Abort(MPI_COMM_WORLD, _status);                                      \
        }                                                                            \
    } while (0)

#define CHECK_MPI(call)                                                              \
    do {                                                                             \
        int _mpi_status = (call);                                                    \
        if (_mpi_status != MPI_SUCCESS) {                                            \
            char _errstr[MPI_MAX_ERROR_STRING];                                      \
            int _errlen = 0;                                                         \
            MPI_Error_string(_mpi_status, _errstr, &_errlen);                        \
            fprintf(stderr, "[%d] MPI error at %s:%d: %s\n",                         \
                    g_rank, __FILE__, __LINE__, _errstr);                            \
            MPI_Abort(MPI_COMM_WORLD, _mpi_status);                                  \
        }                                                                            \
    } while (0)

static int g_rank = 0;

static int
chunk_owner(MPI_Offset chunk_idx, int active_ranks, int chunks_per_rank, int remainder)
{
    (void)chunks_per_rank;
    (void)remainder;
    if (active_ranks <= 0) {
        fprintf(stderr, "ERROR: active_ranks=%d is invalid!\n", active_ranks);
        return 0;
    }
    int owner = (int)(chunk_idx % active_ranks);
    return owner;
}

int main(int argc, char **argv)
{
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    g_rank = rank;

    /* Debug: print PnetCDF library version */
    if (rank == 0) {
        const char *vers = ncmpi_inq_libvers();
        fprintf(stderr, "[DEBUG] PnetCDF libvers: %s\n", vers ? vers : "unknown");
    }

    double total_start_time = MPI_Wtime();  /* Start timing the entire process */

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr,
                    "Usage: %s INPUT_NC OUTPUT_NC [layers=1]\n"
                    "  (recommended: mpirun -n 31)\n",
                    argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_path  = argv[1];
    const char *output_path = argv[2];

    int layers = 1;
    if (argc >= 4) {
        layers = atoi(argv[3]);
        if (layers <= 0) layers = 1;
    }

    int ncid_in;
    CHECK_ERR(ncmpi_open(MPI_COMM_WORLD, input_path, NC_NOWRITE,
                         MPI_INFO_NULL, &ncid_in));

    int time_dimid_in, y_dimid_in, x_dimid_in;
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "time", &time_dimid_in));
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "y", &y_dimid_in));
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "x", &x_dimid_in));

    MPI_Offset time_len, y_len, x_len;
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, time_dimid_in, &time_len));
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, y_dimid_in, &y_len));
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, x_dimid_in, &x_len));
    MPI_Offset chunk_count_t = (time_len + CHUNK_T - 1) / CHUNK_T;

    int x_varid_in, y_varid_in, time_varid_in;
    int lat_varid_in, lon_varid_in, proj_varid_in;
    int flds_varid_in;
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "x", &x_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "y", &y_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "time", &time_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lat", &lat_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lon", &lon_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lambert_conformal_conic", &proj_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "FLDS", &flds_varid_in));

    /* Capture _FillValue from source (defaults to NC_FILL_FLOAT) */
    int has_fill_value = 0;
    float fill_value = NC_FILL_FLOAT;
    nc_type fill_type;
    MPI_Offset fill_len;
    int fill_ret = ncmpi_inq_att(ncid_in, flds_varid_in, "_FillValue", &fill_type, &fill_len);
    if (fill_ret == NC_NOERR && fill_len == 1) {
        if (fill_type == NC_FLOAT) {
            CHECK_ERR(ncmpi_get_att_float(ncid_in, flds_varid_in, "_FillValue", &fill_value));
            has_fill_value = 1;
        } else if (fill_type == NC_DOUBLE) {
            double tmp_fill = 0.0;
            CHECK_ERR(ncmpi_get_att_double(ncid_in, flds_varid_in, "_FillValue", &tmp_fill));
            fill_value = (float)tmp_fill;
            has_fill_value = 1;
        }
    }

    MPI_Info info;
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_chunking", "enable");
    /* hint MPI-IO that each chunk is large (~240 MB) so aggregators allocate enough buffer */
    MPI_Info_set(info, "cb_buffer_size", "536870912");    /* 512 MB */
    MPI_Info_set(info, "striping_unit", "536870912");     /* match cb buffer */
    MPI_Info_set(info, "nc_header_align_size", "1048576");/* 1 MB header alignment */

    int ncid_out;
    CHECK_ERR(ncmpi_create(MPI_COMM_WORLD, output_path,
                           NC_CLOBBER | NC_64BIT_DATA, info, &ncid_out));
    MPI_Info_free(&info);

    int time_dimid_out, y_dimid_out, x_dimid_out;
    CHECK_ERR(ncmpi_def_dim(ncid_out, "time", time_len, &time_dimid_out));
    CHECK_ERR(ncmpi_def_dim(ncid_out, "y",    y_len,    &y_dimid_out));
    CHECK_ERR(ncmpi_def_dim(ncid_out, "x",    x_len,    &x_dimid_out));

    int x_varid_out, y_varid_out, time_varid_out;
    CHECK_ERR(ncmpi_def_var(ncid_out, "x", NC_FLOAT, 1, &x_dimid_out, &x_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "y", NC_FLOAT, 1, &y_dimid_out, &y_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "time", NC_FLOAT, 1, &time_dimid_out, &time_varid_out));

    int lat_dimids[2] = { y_dimid_out, x_dimid_out };
    int lon_dimids[2] = { y_dimid_out, x_dimid_out };
    int lat_varid_out, lon_varid_out;
    CHECK_ERR(ncmpi_def_var(ncid_out, "lat", NC_FLOAT, 2, lat_dimids, &lat_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "lon", NC_FLOAT, 2, lon_dimids, &lon_varid_out));

    int proj_varid_out;
    CHECK_ERR(ncmpi_def_var(ncid_out, "lambert_conformal_conic", NC_SHORT, 0, NULL, &proj_varid_out));

    int flds_dimids[3] = { time_dimid_out, y_dimid_out, x_dimid_out };
    int flds_varid_out;
    CHECK_ERR(ncmpi_def_var(ncid_out, "FLDS", NC_FLOAT, 3, flds_dimids, &flds_varid_out));
    CHECK_ERR(ncmpi_def_var_fill(ncid_out, flds_varid_out, NC_FILL, &fill_value));
    CHECK_ERR(ncmpi_put_att_float(ncid_out, flds_varid_out, "_FillValue",
                                  NC_FLOAT, 1, &fill_value));
    has_fill_value = 1;

    int chunk_dims[3] = {
        (int)((MPI_Offset)CHUNK_T < time_len ? CHUNK_T : (int)time_len),
        (int)((MPI_Offset)CHUNK_Y < y_len ? CHUNK_Y : (int)y_len),
        (int)((MPI_Offset)CHUNK_X < x_len ? CHUNK_X : (int)x_len)
    };
    CHECK_ERR(ncmpi_var_set_chunk(ncid_out, flds_varid_out, chunk_dims));
    CHECK_ERR(ncmpi_var_set_filter(ncid_out, flds_varid_out, NC_FILTER_IPCOMP));

    const char *algo = "ipcomp";
    const char *interp = "cubic";
    const char *codec_version = "1.0.0";
    int level_progressive = layers;
    int block_size = 8192; /* test: force block size */
    double data_range = 0.0;

    /* Basic compression attributes */
    CHECK_ERR(ncmpi_put_att_text(ncid_out, flds_varid_out, "comp:algo",
                                 (MPI_Offset)strlen(algo), algo));
    CHECK_ERR(ncmpi_put_att_text(ncid_out, flds_varid_out, "comp:interp",
                                 (MPI_Offset)strlen(interp), interp));
    CHECK_ERR(ncmpi_put_att_text(ncid_out, flds_varid_out, "comp:codec_version",
                                 (MPI_Offset)strlen(codec_version), codec_version));
    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:layers",
                                NC_INT, 1, &layers));
    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:level_progressive",
                                NC_INT, 1, &level_progressive));
    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:block_size",
                                NC_INT, 1, &block_size));
    CHECK_ERR(ncmpi_put_att_double(ncid_out, flds_varid_out, "comp:data_range",
                                   NC_DOUBLE, 1, &data_range));

    /* Additional attributes required for progressive decompression API */
    int logical_type = NC_FLOAT;  /* Data type of the variable */
    int logical_ndim = 3;         /* Number of dimensions (time, y, x) */
    int group_bits = 32;          /* Default grouping bits for progressive */
    long long logical_dims[3] = { (long long)time_len, (long long)y_len, (long long)x_len };

    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:logical_type",
                                NC_INT, 1, &logical_type));
    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:logical_ndim",
                                NC_INT, 1, &logical_ndim));
    CHECK_ERR(ncmpi_put_att_longlong(ncid_out, flds_varid_out, "comp:logical_dims",
                                     NC_INT64, 3, logical_dims));
    CHECK_ERR(ncmpi_put_att_int(ncid_out, flds_varid_out, "comp:group_bits",
                                NC_INT, 1, &group_bits));

    CHECK_ERR(ncmpi_enddef(ncid_out));

    /* Write coordinate variables using collective mode to avoid chunk I/O issues */
    /* All ranks participate, but only rank 0 has data */
    {
        float *buf_x = NULL;
        float *buf_y = NULL;
        float *buf_t = NULL;
        float *buf_latlon = NULL;
        short proj = 0;

        MPI_Offset start_1d[1] = {0};
        MPI_Offset start_2d[2] = {0, 0};
        MPI_Offset count_x[1], count_y[1], count_t[1];
        MPI_Offset count_latlon[2];

        if (rank == 0) {
            buf_x = (float *)malloc((size_t)x_len * sizeof(float));
            buf_y = (float *)malloc((size_t)y_len * sizeof(float));
            buf_t = (float *)malloc((size_t)time_len * sizeof(float));
            buf_latlon = (float *)malloc((size_t)y_len * (size_t)x_len * sizeof(float));

            if (buf_x == NULL || buf_y == NULL || buf_t == NULL || buf_latlon == NULL) {
                fprintf(stderr, "Rank 0: failed to allocate coordinate buffers\n");
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            count_x[0] = x_len;
            count_y[0] = y_len;
            count_t[0] = time_len;
            count_latlon[0] = y_len;
            count_latlon[1] = x_len;
        } else {
            /* Non-zero ranks write nothing */
            count_x[0] = 0;
            count_y[0] = 0;
            count_t[0] = 0;
            count_latlon[0] = 0;
            count_latlon[1] = 0;
        }

        /* Read on rank 0 only - use independent mode for reading */
        CHECK_ERR(ncmpi_begin_indep_data(ncid_in));
        if (rank == 0) {
            CHECK_ERR(ncmpi_get_var_float(ncid_in, x_varid_in, buf_x));
            CHECK_ERR(ncmpi_get_var_float(ncid_in, y_varid_in, buf_y));
            CHECK_ERR(ncmpi_get_var_float(ncid_in, time_varid_in, buf_t));
            CHECK_ERR(ncmpi_get_var_short(ncid_in, proj_varid_in, &proj));
            CHECK_ERR(ncmpi_get_var_float(ncid_in, lat_varid_in, buf_latlon));
        }

        /* Write using collective mode - all ranks participate */
        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, x_varid_out, start_1d, count_x, buf_x));
        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, y_varid_out, start_1d, count_y, buf_y));
        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, time_varid_out, start_1d, count_t, buf_t));
        
        /* For scalar variable, only rank 0 writes */
        if (rank == 0) {
            CHECK_ERR(ncmpi_begin_indep_data(ncid_out));
            CHECK_ERR(ncmpi_put_var_short(ncid_out, proj_varid_out, &proj));
            CHECK_ERR(ncmpi_end_indep_data(ncid_out));
        } else {
            CHECK_ERR(ncmpi_begin_indep_data(ncid_out));
            CHECK_ERR(ncmpi_end_indep_data(ncid_out));
        }

        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, lat_varid_out, start_2d, count_latlon, buf_latlon));

        /* Read lon on rank 0 and write */
        if (rank == 0) {
            CHECK_ERR(ncmpi_get_var_float(ncid_in, lon_varid_in, buf_latlon));
        }
        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, lon_varid_out, start_2d, count_latlon, buf_latlon));

        if (rank == 0) {
            free(buf_x);
            free(buf_y);
            free(buf_t);
            free(buf_latlon);
        }
    }

    /* Both files use independent mode for parallel processing.
     * Each rank processes its own chunks concurrently.
     */
    CHECK_ERR(ncmpi_begin_indep_data(ncid_out));

    if (chunk_count_t == 0) {
        CHECK_ERR(ncmpi_end_indep_data(ncid_in));
        CHECK_ERR(ncmpi_end_indep_data(ncid_out));
        CHECK_ERR(ncmpi_sync(ncid_out));
        CHECK_ERR(ncmpi_close(ncid_out));
        CHECK_ERR(ncmpi_close(ncid_in));
        MPI_Finalize();
        return 0;
    }

    int active_ranks = (int)((chunk_count_t < (MPI_Offset)nprocs) ? chunk_count_t : (MPI_Offset)nprocs);
    if (active_ranks == 0) active_ranks = 1;

    int chunks_per_rank = (int)(chunk_count_t / active_ranks);
    int remainder = (int)(chunk_count_t % active_ranks);

    if (rank == 0) {
        printf("[DEBUG] nprocs=%d, chunk_count_t=%lld, active_ranks=%d\n",
               nprocs, (long long)chunk_count_t, active_ranks);
        printf("[DEBUG] chunks_per_rank=%d, remainder=%d\n",
               chunks_per_rank, remainder);
        fflush(stdout);
    }

    if (rank == 0 && active_ranks < nprocs) {
        printf("Info: using %d out of %d ranks for %lld time chunks.\n",
               active_ranks, nprocs, (long long)chunk_count_t);
    }

    /* Count how many chunks this rank owns */
    int my_chunk_count = 0;
    for (MPI_Offset chunk_idx = 0; chunk_idx < chunk_count_t; ++chunk_idx) {
        int owner = chunk_owner(chunk_idx, active_ranks, chunks_per_rank, remainder);
        if (rank == owner) {
            my_chunk_count++;
        }
    }
    
    /* Print from all ranks to confirm parallel execution */
    printf("[Rank %d] my_chunk_count=%d (will process chunks: ", rank, my_chunk_count);
    int printed = 0;
    for (MPI_Offset ci = 0; ci < chunk_count_t && printed < 5; ++ci) {
        if (chunk_owner(ci, active_ranks, chunks_per_rank, remainder) == rank) {
            printf("%lld ", (long long)ci);
            printed++;
        }
    }
    if (my_chunk_count > 5) printf("...");
    printf(")\n");
    fflush(stdout);
    
    /* Allocate one reusable buffer per rank */
    float *chunk_buf = NULL;
    size_t chunk_buf_elems = 0;
    if (my_chunk_count > 0) {
        MPI_Offset max_t = (time_len < (MPI_Offset)CHUNK_T) ? time_len : (MPI_Offset)CHUNK_T;
        MPI_Offset max_elems = max_t * y_len * x_len;
        if (max_elems <= 0 || (double)max_elems > (double)SIZE_MAX / sizeof(float)) {
            fprintf(stderr, "[%d] Chunk buffer size exceeds size_t limit\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        chunk_buf_elems = (size_t)max_elems;
        chunk_buf = (float *)malloc(chunk_buf_elems * sizeof(float));
        if (chunk_buf == NULL) {
            fprintf(stderr, "[%d] Failed to allocate %.2f MB chunk buffer\n",
                    rank, (double)chunk_buf_elems * sizeof(float) / (1024.0 * 1024.0));
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);  /* Sync before starting main loop */

    if (rank == 0) {
        printf("\n[DEBUG] Starting main loop: chunk_count_t=%lld\n",
               (long long)chunk_count_t);
        printf("[DEBUG] Each chunk is %.2f MB\n", 
               (double)(y_len * x_len * sizeof(float)) / (1024.0 * 1024.0));
        fflush(stdout);
    }

    int total_processed = 0;
    double start_time = MPI_Wtime();
    double local_min = DBL_MAX;
    double local_max = -DBL_MAX;
    long long local_valid = 0;

    /* 
     * Process chunks in PARALLEL using independent mode.
     * Each rank processes only its own chunks concurrently.
     * All ranks work simultaneously on different chunks.
     */
    for (MPI_Offset chunk_idx = 0; chunk_idx < chunk_count_t; ++chunk_idx) {
        int owner = chunk_owner(chunk_idx, active_ranks, chunks_per_rank, remainder);
        
        /* Only owner processes this chunk */
        if (rank != owner) {
            continue;
        }

        MPI_Offset start[3] = { chunk_idx * CHUNK_T, 0, 0 };
        MPI_Offset count[3] = { CHUNK_T, y_len, x_len };

        if (start[0] + count[0] > time_len) {
            count[0] = time_len - start[0];
        }

        MPI_Offset elems = count[0] * count[1] * count[2];
        if (elems <= 0) {
            continue;
        }

        if ((size_t)elems > SIZE_MAX / sizeof(float)) {
            fprintf(stderr, "[%d] Chunk %lld exceeds size_t limit\n",
                    rank, (long long)chunk_idx);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        if (chunk_buf == NULL || (MPI_Offset)chunk_buf_elems < elems) {
            fprintf(stderr, "[%d] Chunk buffer too small for chunk %lld\n",
                    rank, (long long)chunk_idx);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* Print progress */
        printf("[Rank %d] Processing chunk %lld/%lld...\n",
               rank, (long long)chunk_idx, (long long)(chunk_count_t - 1));
        fflush(stdout);

        /* Read chunk data (independent mode) */
        CHECK_ERR(ncmpi_get_vara_float(ncid_in, flds_varid_in, start, count, chunk_buf));

        /* Update local statistics ignoring _FillValue/NaN/Inf */
        for (MPI_Offset i = 0; i < elems; i++) {
            float val = chunk_buf[i];
            if ((has_fill_value && val == fill_value) || isnan(val) || isinf(val)) {
                continue;
            }
            if (val < local_min) local_min = val;
            if (val > local_max) local_max = val;
            local_valid++;
        }

        /* Write chunk data using independent mode (parallel) */
        CHECK_ERR(ncmpi_put_vara_float(ncid_out, flds_varid_out, start, count, chunk_buf));

        total_processed++;

        printf("[Rank %d] Completed chunk %lld (my total: %d)\n", 
               rank, (long long)chunk_idx, total_processed);
        fflush(stdout);
    }

    if (chunk_buf != NULL) {
        free(chunk_buf);
        chunk_buf = NULL;
    }

    double global_min = 0.0;
    double global_max = 0.0;
    long long global_valid = 0;

    MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_valid, &global_valid, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double attr_values[3] = {0.0, 0.0, 0.0};
    long long valid_total = 0;
    if (rank == 0 && global_valid > 0) {
        double span = global_max - global_min;
        double scale = fmax(fabs(global_max), fabs(global_min));
        if (scale < 1.0) scale = 1.0;
        if (span <= 0.0) {
            span = MIN_RANGE_EPS * scale;
        }
        attr_values[0] = global_min;
        attr_values[1] = global_max;
        attr_values[2] = span;
        valid_total = global_valid;
    } else if (rank == 0) {
        valid_total = 0;
        attr_values[0] = 0.0;
        attr_values[1] = 0.0;
        attr_values[2] = 0.0;
    }

    MPI_Bcast(attr_values, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&valid_total, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    printf("[Rank %d] Finished all %d chunks in %.2f seconds\n", 
           rank, total_processed, end_time - start_time);
    fflush(stdout);

    /* Barrier to ensure all ranks have finished their independent writes */
    MPI_Barrier(MPI_COMM_WORLD);

    /* End independent mode for both files before closing */
    CHECK_ERR(ncmpi_end_indep_data(ncid_in));
    CHECK_ERR(ncmpi_end_indep_data(ncid_out));

    /* Update compression metadata with actual data range */
    CHECK_ERR(ncmpi_redef(ncid_out));
    CHECK_ERR(ncmpi_put_att_double(ncid_out, flds_varid_out, "comp:data_range",
                                   NC_DOUBLE, 1, &attr_values[2]));
    if (valid_total > 0) {
        CHECK_ERR(ncmpi_put_att_double(ncid_out, flds_varid_out, "comp:data_min",
                                       NC_DOUBLE, 1, &attr_values[0]));
        CHECK_ERR(ncmpi_put_att_double(ncid_out, flds_varid_out, "comp:data_max",
                                       NC_DOUBLE, 1, &attr_values[1]));
    }
    CHECK_ERR(ncmpi_enddef(ncid_out));

    if (rank == 0) {
        if (valid_total > 0) {
            printf("[INFO] FLDS stats: min=%.6f max=%.6f range=%.6f (valid=%lld)\n",
                   attr_values[0], attr_values[1], attr_values[2],
                   (long long)valid_total);
        } else {
            printf("[WARN] No finite FLDS samples detected; comp:data_range set to 0\n");
        }
    }
    
    CHECK_ERR(ncmpi_sync(ncid_out));

    CHECK_ERR(ncmpi_close(ncid_out));
    CHECK_ERR(ncmpi_close(ncid_in));

    double total_end_time = MPI_Wtime();
    double total_elapsed = total_end_time - total_start_time;

    /* Gather timing from all ranks to report min/max/avg */
    double max_time, min_time, sum_time;
    MPI_Reduce(&total_elapsed, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_elapsed, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&total_elapsed, &sum_time, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n========== SUMMARY ==========\n");
        printf("Compression complete: %s -> %s\n", input_path, output_path);
        printf("Total chunks: %lld, Chunk size: %d x %d x %d\n",
               (long long)chunk_count_t, CHUNK_T, CHUNK_Y, CHUNK_X);
        printf("Processes: %d\n", nprocs);
        printf("Total time: %.2f seconds (min: %.2f, max: %.2f, avg: %.2f)\n",
               max_time, min_time, max_time, sum_time / nprocs);
        printf("=============================\n");
    }

    MPI_Finalize();
    return 0;
}