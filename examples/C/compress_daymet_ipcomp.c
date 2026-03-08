#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <mpi.h>
#include <pnetcdf.h>
#include <float.h>
#include <math.h>

#ifdef ENABLE_IPCOMP
extern void ipcomp_reset_core_timers(void);
extern double ipcomp_get_core_compress_time(void);
extern unsigned long long ipcomp_get_core_compress_bytes(void);
#endif

#define CHUNK_T 1
#define CHUNK_Y 8075
#define CHUNK_X 7814

#define MIN_RANGE_EPS 1e-9
#define DEFAULT_BATCH_CHUNKS 248

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
owner_for_contiguous_chunk(MPI_Offset chunk_idx, int active_ranks, MPI_Offset total_chunks)
{
    if (active_ranks <= 0) {
        fprintf(stderr, "ERROR: active_ranks=%d is invalid!\n", active_ranks);
        return 0;
    }

    MPI_Offset base = total_chunks / active_ranks;
    MPI_Offset rem = total_chunks % active_ranks;
    if (base == 0) {
        return (int)chunk_idx;
    }

    MPI_Offset split = (base + 1) * rem;
    if (chunk_idx < split) {
        return (int)(chunk_idx / (base + 1));
    }
    return (int)(rem + (chunk_idx - split) / base);
}

static int
chunk_owner(MPI_Offset chunk_idx,
            MPI_Offset total_chunks,
            int        active_ranks)
{
    /* Keep app scheduling identical to ncchunkio owner map. */
    return owner_for_contiguous_chunk(chunk_idx, active_ranks, total_chunks);
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
    MPI_Offset chunk_grid_t = (time_len + CHUNK_T - 1) / CHUNK_T;
    MPI_Offset chunk_grid_y = (y_len + CHUNK_Y - 1) / CHUNK_Y;
    MPI_Offset chunk_grid_x = (x_len + CHUNK_X - 1) / CHUNK_X;
    MPI_Offset chunks_per_record = chunk_grid_y * chunk_grid_x;
    MPI_Offset total_chunks = chunk_grid_t * chunks_per_record;

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
    MPI_Info_set(info, "nc_chk_cown_ratio", "0");         /* disable ownership load-balance penalty */
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

    /* End independent mode used for coordinate reads.
     * FLDS chunk reads below are performed in independent mode per batch.
     */
    CHECK_ERR(ncmpi_end_indep_data(ncid_in));

    if (total_chunks == 0) {
        CHECK_ERR(ncmpi_sync(ncid_out));
        CHECK_ERR(ncmpi_close(ncid_out));
        CHECK_ERR(ncmpi_close(ncid_in));
        MPI_Finalize();
        return 0;
    }

    int active_ranks = (int)((total_chunks < (MPI_Offset)nprocs) ?
                             total_chunks : (MPI_Offset)nprocs);
    if (active_ranks == 0) active_ranks = 1;

    if (rank == 0) {
        printf("[DEBUG] nprocs=%d, total_chunks=%lld, active_ranks=%d\n",
               nprocs, (long long)total_chunks, active_ranks);
        printf("[DEBUG] chunk grid (t,y,x) = %lld x %lld x %lld\n",
               (long long)chunk_grid_t, (long long)chunk_grid_y, (long long)chunk_grid_x);
        printf("[DEBUG] chunk ownership: contiguous chunk-id blocks\n");
        fflush(stdout);
    }

    if (rank == 0 && active_ranks < nprocs) {
        printf("Info: using %d out of %d ranks for %lld total chunks.\n",
               active_ranks, nprocs, (long long)total_chunks);
    }

    /* Count how many chunks this rank owns */
    int my_chunk_count = 0;
    for (MPI_Offset chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
        int owner = chunk_owner(chunk_idx, total_chunks, active_ranks);
        if (rank == owner) {
            my_chunk_count++;
        }
    }
    
    /* Print from all ranks to confirm parallel execution */
    printf("[Rank %d] my_chunk_count=%d (will process chunks: ", rank, my_chunk_count);
    int printed = 0;
    for (MPI_Offset ci = 0; ci < total_chunks && printed < 5; ++ci) {
        if (chunk_owner(ci, total_chunks, active_ranks) == rank) {
            printf("%lld ", (long long)ci);
            printed++;
        }
    }
    if (my_chunk_count > 5) printf("...");
    printf(")\n");
    fflush(stdout);

    /* Compute per-chunk capacity (max elements in one chunk) */
    MPI_Offset max_t = (time_len < (MPI_Offset)CHUNK_T) ? time_len : (MPI_Offset)CHUNK_T;
    MPI_Offset max_y = (y_len < (MPI_Offset)CHUNK_Y) ? y_len : (MPI_Offset)CHUNK_Y;
    MPI_Offset max_x = (x_len < (MPI_Offset)CHUNK_X) ? x_len : (MPI_Offset)CHUNK_X;
    size_t chunk_buf_elems = (size_t)(max_t * max_y * max_x);

    int batch_chunks = DEFAULT_BATCH_CHUNKS;
    {
        const char *env_batch = getenv("IPCOMP_BATCH_CHUNKS");
        if (env_batch != NULL) {
            int v = atoi(env_batch);
            if (v > 0) batch_chunks = v;
        }
    }

    /* Store metadata for all chunks owned by this rank */
    MPI_Offset *all_starts = NULL;   /* my_chunk_count * 3 */
    MPI_Offset *all_counts = NULL;   /* my_chunk_count * 3 */
    MPI_Offset *all_elems  = NULL;   /* my_chunk_count */

    if (my_chunk_count > 0) {
        all_starts = (MPI_Offset *)malloc((size_t)my_chunk_count * 3 * sizeof(MPI_Offset));
        all_counts = (MPI_Offset *)malloc((size_t)my_chunk_count * 3 * sizeof(MPI_Offset));
        all_elems  = (MPI_Offset *)malloc((size_t)my_chunk_count * sizeof(MPI_Offset));
        if (all_starts == NULL || all_counts == NULL || all_elems == NULL) {
            fprintf(stderr, "[%d] Failed to allocate chunk metadata arrays\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    /* Pre-compute chunk metadata for this rank's chunks */
    {
        int ci = 0;
        MPI_Offset plane = chunks_per_record;
        for (MPI_Offset chunk_idx = 0; chunk_idx < total_chunks; ++chunk_idx) {
            if (chunk_owner(chunk_idx, total_chunks, active_ranks) != rank)
                continue;
            MPI_Offset ct = chunk_idx / plane;
            MPI_Offset rem = chunk_idx % plane;
            MPI_Offset cy = rem / chunk_grid_x;
            MPI_Offset cx = rem % chunk_grid_x;

            all_starts[ci * 3 + 0] = ct * CHUNK_T;
            all_starts[ci * 3 + 1] = cy * CHUNK_Y;
            all_starts[ci * 3 + 2] = cx * CHUNK_X;
            all_counts[ci * 3 + 0] = CHUNK_T;
            all_counts[ci * 3 + 1] = CHUNK_Y;
            all_counts[ci * 3 + 2] = CHUNK_X;
            if (all_starts[ci * 3] + all_counts[ci * 3] > time_len)
                all_counts[ci * 3] = time_len - all_starts[ci * 3];
            if (all_starts[ci * 3 + 1] + all_counts[ci * 3 + 1] > y_len)
                all_counts[ci * 3 + 1] = y_len - all_starts[ci * 3 + 1];
            if (all_starts[ci * 3 + 2] + all_counts[ci * 3 + 2] > x_len)
                all_counts[ci * 3 + 2] = x_len - all_starts[ci * 3 + 2];
            all_elems[ci] = all_counts[ci * 3 + 0]
                          * all_counts[ci * 3 + 1]
                          * all_counts[ci * 3 + 2];
            ci++;
        }
    }

    if (rank == 0) {
        printf("\n[DEBUG] Starting main loop: total_chunks=%lld\n",
               (long long)total_chunks);
        printf("[DEBUG] Each chunk is %.2f MB\n",
               (double)(chunk_buf_elems * sizeof(float)) / (1024.0 * 1024.0));
        printf("[DEBUG] Batch size: %d chunks (env IPCOMP_BATCH_CHUNKS)\n", batch_chunks);
        printf("[DEBUG] Per-rank batch buffer cap: %d chunks x %.2f MB = %.2f MB\n",
               batch_chunks,
               (double)chunk_buf_elems * sizeof(float) / (1024.0 * 1024.0),
               (double)batch_chunks * chunk_buf_elems * sizeof(float) / (1024.0 * 1024.0));
        fflush(stdout);
    }

    double local_min = DBL_MAX;
    double local_max = -DBL_MAX;
    long long local_valid = 0;

    /* Barrier before timing start */
    MPI_Barrier(MPI_COMM_WORLD);
    double t_read = 0.0;           /* sum of per-batch read time */
    double t_write = 0.0;          /* sum of per-batch write+flush time */

    /*
     * Batch pipeline:
     *   1) read a batch into temporary buffer
     *   2) compute local min/max statistics
     *   3) write batch and flush (end_indep_data) to bound memory
     *   4) free batch buffer and continue
     */
#ifdef ENABLE_IPCOMP
    ipcomp_reset_core_timers();
#endif
    for (int batch_begin = 0; batch_begin < my_chunk_count; batch_begin += batch_chunks) {
        int this_batch = my_chunk_count - batch_begin;
        if (this_batch > batch_chunks) this_batch = batch_chunks;

        size_t batch_buf_bytes = (size_t)this_batch * chunk_buf_elems * sizeof(float);
        float *batch_buf = (float *)malloc(batch_buf_bytes);
        if (batch_buf == NULL) {
            fprintf(stderr, "[%d] Failed to allocate %.2f MB batch buffer (batch=%d)\n",
                    rank, (double)batch_buf_bytes / (1024.0 * 1024.0), this_batch);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        double t0 = MPI_Wtime();
        CHECK_ERR(ncmpi_begin_indep_data(ncid_in));
        for (int bi = 0; bi < this_batch; bi++) {
            int i = batch_begin + bi;
            MPI_Offset start[3] = { all_starts[i*3], all_starts[i*3+1], all_starts[i*3+2] };
            MPI_Offset count[3] = { all_counts[i*3], all_counts[i*3+1], all_counts[i*3+2] };
            float *buf = batch_buf + (size_t)bi * chunk_buf_elems;
            CHECK_ERR(ncmpi_get_vara_float(ncid_in, flds_varid_in, start, count, buf));
        }
        CHECK_ERR(ncmpi_end_indep_data(ncid_in));
        t_read += MPI_Wtime() - t0;

        for (int bi = 0; bi < this_batch; bi++) {
            int i = batch_begin + bi;
            float *buf = batch_buf + (size_t)bi * chunk_buf_elems;
            for (MPI_Offset j = 0; j < all_elems[i]; j++) {
                float val = buf[j];
                if ((has_fill_value && val == fill_value) || isnan(val) || isinf(val))
                    continue;
                if (val < local_min) local_min = val;
                if (val > local_max) local_max = val;
                local_valid++;
            }
        }

        t0 = MPI_Wtime();
        CHECK_ERR(ncmpi_begin_indep_data(ncid_out));
        for (int bi = 0; bi < this_batch; bi++) {
            int i = batch_begin + bi;
            MPI_Offset start[3] = { all_starts[i*3], all_starts[i*3+1], all_starts[i*3+2] };
            MPI_Offset count[3] = { all_counts[i*3], all_counts[i*3+1], all_counts[i*3+2] };
            float *buf = batch_buf + (size_t)bi * chunk_buf_elems;
            CHECK_ERR(ncmpi_put_vara_float(ncid_out, flds_varid_out, start, count, buf));
        }
        CHECK_ERR(ncmpi_end_indep_data(ncid_out)); /* flush this batch */
        t_write += MPI_Wtime() - t0;

        free(batch_buf);
    }

#ifdef ENABLE_IPCOMP
    double core_comp = ipcomp_get_core_compress_time();
    unsigned long long core_comp_bytes = ipcomp_get_core_compress_bytes();
#else
    double core_comp = 0.0;
    unsigned long long core_comp_bytes = 0ULL;
#endif
    /* write_io = pure I/O portion of write (compression subtracted) */
    double write_io = t_write - core_comp;
    if (write_io < 0.0) write_io = 0.0;
    double read_io = t_read;
    double t_total = t_read + t_write;

    if (all_starts != NULL) free(all_starts);
    if (all_counts != NULL) free(all_counts);
    if (all_elems != NULL) free(all_elems);

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

    printf("[Rank %d] Finished %d chunks in %.2f seconds\n",
           rank, my_chunk_count, t_total);
    fflush(stdout);

    double *rank_total = NULL;
    double *rank_read = NULL;
    double *rank_core = NULL;
    double *rank_write = NULL;
    long long *rank_valid_count = NULL;
    unsigned long long *rank_core_bytes = NULL;
    if (rank == 0) {
        rank_total = (double *)malloc((size_t)nprocs * sizeof(double));
        rank_read = (double *)malloc((size_t)nprocs * sizeof(double));
        rank_core = (double *)malloc((size_t)nprocs * sizeof(double));
        rank_write = (double *)malloc((size_t)nprocs * sizeof(double));
        rank_valid_count = (long long *)malloc((size_t)nprocs * sizeof(long long));
        rank_core_bytes = (unsigned long long *)malloc((size_t)nprocs * sizeof(unsigned long long));
        if (rank_total == NULL || rank_read == NULL ||
            rank_core == NULL || rank_write == NULL ||
            rank_valid_count == NULL || rank_core_bytes == NULL) {
            fprintf(stderr, "[%d] Failed to allocate timing buffers\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    MPI_Gather(&t_total, 1, MPI_DOUBLE, rank_total, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&read_io, 1, MPI_DOUBLE, rank_read, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&core_comp, 1, MPI_DOUBLE, rank_core, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&write_io, 1, MPI_DOUBLE, rank_write, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Gather(&local_valid, 1, MPI_LONG_LONG, rank_valid_count, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Gather(&core_comp_bytes, 1, MPI_UNSIGNED_LONG_LONG,
               rank_core_bytes, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);

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

    if (rank == 0) {
        double total_min = rank_total[0], total_max = rank_total[0], total_sum = 0.0;
        double read_min = rank_read[0], read_max = rank_read[0], read_sum = 0.0;
        double core_min = rank_core[0], core_max = rank_core[0], core_sum = 0.0;
        double write_min = rank_write[0], write_max = rank_write[0], write_sum = 0.0;
        long long valid_min = rank_valid_count[0], valid_max = rank_valid_count[0];
        long double valid_sum = 0.0L;
        unsigned long long bytes_min = rank_core_bytes[0], bytes_max = rank_core_bytes[0];
        long double bytes_sum = 0.0L;
        for (int r = 0; r < nprocs; r++) {
            double tv = rank_total[r];
            double rv = rank_read[r];
            double cv = rank_core[r];
            double wv = rank_write[r];
            long long vv = rank_valid_count[r];
            unsigned long long bv = rank_core_bytes[r];
            if (tv < total_min) total_min = tv;
            if (tv > total_max) total_max = tv;
            if (rv < read_min) read_min = rv;
            if (rv > read_max) read_max = rv;
            if (cv < core_min) core_min = cv;
            if (cv > core_max) core_max = cv;
            if (wv < write_min) write_min = wv;
            if (wv > write_max) write_max = wv;
            if (vv < valid_min) valid_min = vv;
            if (vv > valid_max) valid_max = vv;
            if (bv < bytes_min) bytes_min = bv;
            if (bv > bytes_max) bytes_max = bv;
            total_sum += tv;
            read_sum += rv;
            core_sum += cv;
            write_sum += wv;
            valid_sum += (long double)vv;
            bytes_sum += (long double)bv;
        }
        printf("\n========== SUMMARY ==========\n");
        printf("Compression complete: %s -> %s\n", input_path, output_path);
        printf("Total chunks: %lld, Chunk size (t,y,x): %d x %d x %d\n",
               (long long)total_chunks, CHUNK_T, CHUNK_Y, CHUNK_X);
        printf("Processes: %d\n", nprocs);
        printf("Per-rank breakdown (seconds):\n");
        for (int r = 0; r < nprocs; r++) {
            printf("  rank %d: total=%.6f read_io=%.6f core_comp=%.6f write_io=%.6f "
                   "valid_count=%lld core_comp_bytes=%llu\n",
                   r, rank_total[r], rank_read[r], rank_core[r], rank_write[r],
                   rank_valid_count[r], rank_core_bytes[r]);
        }
        printf("Timing stats across ranks (seconds):\n");
        printf("  total: min=%.6f max=%.6f avg=%.6f\n",
               total_min, total_max, total_sum / (double)nprocs);
        printf("  read_io: min=%.6f max=%.6f avg=%.6f\n",
               read_min, read_max, read_sum / (double)nprocs);
        printf("  core_comp: min=%.6f max=%.6f avg=%.6f\n",
               core_min, core_max, core_sum / (double)nprocs);
        printf("  write_io: min=%.6f max=%.6f avg=%.6f\n",
               write_min, write_max, write_sum / (double)nprocs);
        printf("Data/compression diagnostics across ranks:\n");
        printf("  valid_count: min=%lld max=%lld avg=%.2Lf\n",
               valid_min, valid_max, valid_sum / (long double)nprocs);
        printf("  core_comp_bytes: min=%llu max=%llu avg=%.2Lf\n",
               bytes_min, bytes_max, bytes_sum / (long double)nprocs);
        printf("  core_comp_bytes(MiB): min=%.2f max=%.2f avg=%.2f\n",
               (double)bytes_min / (1024.0 * 1024.0),
               (double)bytes_max / (1024.0 * 1024.0),
               (double)(bytes_sum / (long double)nprocs) / (1024.0 * 1024.0));
        printf("=============================\n");
    }

    if (rank_total != NULL) free(rank_total);
    if (rank_read != NULL) free(rank_read);
    if (rank_core != NULL) free(rank_core);
    if (rank_write != NULL) free(rank_write);
    if (rank_valid_count != NULL) free(rank_valid_count);
    if (rank_core_bytes != NULL) free(rank_core_bytes);

    MPI_Finalize();
    return 0;
}