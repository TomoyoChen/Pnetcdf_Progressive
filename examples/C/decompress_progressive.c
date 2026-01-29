/*
 * decompress_progressive.c
 * 
 * Progressive decompression with user-specified error bounds.
 * This demonstrates IPComp's core feature: decompressing to different
 * precision levels from a single compressed file.
 * 
 * Usage:
 *   mpirun -n <nprocs> ./decompress_progressive COMPRESSED.nc OUTPUT_PREFIX [eb1,eb2,eb3,...]
 * 
 * Example:
 *   mpirun -n 4 ./decompress_progressive FLDS_ipcomp_4.nc FLDS_decompressed 1e-2,1e-4,1e-6
 *   
 *   This will create:
 *     - FLDS_decompressed_eb1e-02.nc  (lowest precision, smallest data transfer)
 *     - FLDS_decompressed_eb1e-04.nc  (medium precision)
 *     - FLDS_decompressed_eb1e-06.nc  (highest precision, most data transfer)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <mpi.h>
#include <pnetcdf.h>

#ifdef ENABLE_IPCOMP
extern void ipcomp_reset_core_timers(void);
extern double ipcomp_get_core_decompress_time(void);
#endif
#define MAX_ERROR_BOUNDS 10

#define CHECK_ERR(call)                                                              \
    do {                                                                             \
        int _status = (call);                                                        \
        if (_status != NC_NOERR) {                                                   \
            fprintf(stderr, "[%d] PnetCDF error at %s:%d: %s\n",                     \
                    rank, __FILE__, __LINE__, ncmpi_strerror(_status));              \
            MPI_Abort(MPI_COMM_WORLD, _status);                                      \
        }                                                                            \
    } while (0)

static int rank, nprocs;

static int env_enabled(const char *name)
{
    const char *val = getenv(name);
    if (val == NULL || *val == '\0') return 0;
    if (strcmp(val, "0") == 0) return 0;
    return 1;
}

static int env_parse_batch(const char *name, long *value, int *is_all)
{
    const char *val = getenv(name);
    if (val == NULL || *val == '\0') return 0;
    if (strcmp(val, "all") == 0 || strcmp(val, "ALL") == 0) {
        *is_all = 1;
        *value = -1;
        return 1;
    }
    char *end = NULL;
    long v = strtol(val, &end, 10);
    if (end == val) return -1;
    *is_all = 0;
    *value = v;
    return 1;
}

static int get_chunk_owner(MPI_Offset chunk_id, int nprocs, MPI_Offset total_chunks,
                           int use_block_owner)
{
    if (!use_block_owner || nprocs <= 0) {
        return (int)(chunk_id % nprocs);
    }

    MPI_Offset base = total_chunks / nprocs;
    MPI_Offset rem = total_chunks % nprocs;

    if (base == 0) {
        return (int)chunk_id;
    }

    MPI_Offset split = (base + 1) * rem;
    if (chunk_id < split) {
        return (int)(chunk_id / (base + 1));
    }
    return (int)(rem + (chunk_id - split) / base);
}

/* Parse comma-separated error bounds string */
static int parse_error_bounds(const char *str, double *ebs, int max_count) {
    int count = 0;
    char *copy = strdup(str);
    char *token = strtok(copy, ",");
    
    while (token != NULL && count < max_count) {
        ebs[count] = atof(token);
        if (ebs[count] > 0) {
            count++;
        }
        token = strtok(NULL, ",");
    }
    
    free(copy);
    return count;
}

/* Copy all attributes from one variable to another */
static int copy_var_atts(int ncid_in, int varid_in, int ncid_out, int varid_out)
{
    int natts;
    CHECK_ERR(ncmpi_inq_varnatts(ncid_in, varid_in, &natts));
    
    for (int i = 0; i < natts; i++) {
        char att_name[NC_MAX_NAME + 1];
        nc_type att_type;
        MPI_Offset att_len;
        
        CHECK_ERR(ncmpi_inq_attname(ncid_in, varid_in, i, att_name));
        
        /* Skip internal attributes and compression-related attributes (except _FillValue) */
        int is_fill_value = (strcmp(att_name, "_FillValue") == 0);
        if (att_name[0] == '_' && !is_fill_value) continue;
        if (strncmp(att_name, "comp:", 5) == 0) continue;
        
        CHECK_ERR(ncmpi_inq_att(ncid_in, varid_in, att_name, &att_type, &att_len));
        
        void *att_val = malloc((size_t)att_len * 8);
        if (att_val == NULL) continue;
        
        CHECK_ERR(ncmpi_get_att(ncid_in, varid_in, att_name, att_val));
        CHECK_ERR(ncmpi_put_att(ncid_out, varid_out, att_name, att_type, att_len, att_val));
        
        free(att_val);
    }
    return 0;
}

/* Copy global attributes */
static int copy_global_atts(int ncid_in, int ncid_out)
{
    int natts;
    CHECK_ERR(ncmpi_inq_natts(ncid_in, &natts));
    
    for (int i = 0; i < natts; i++) {
        char att_name[NC_MAX_NAME + 1];
        nc_type att_type;
        MPI_Offset att_len;
        
        CHECK_ERR(ncmpi_inq_attname(ncid_in, NC_GLOBAL, i, att_name));
        if (att_name[0] == '_') continue;
        
        CHECK_ERR(ncmpi_inq_att(ncid_in, NC_GLOBAL, att_name, &att_type, &att_len));
        
        void *att_val = malloc((size_t)att_len * 8);
        if (att_val == NULL) continue;
        
        CHECK_ERR(ncmpi_get_att(ncid_in, NC_GLOBAL, att_name, att_val));
        CHECK_ERR(ncmpi_put_att(ncid_out, NC_GLOBAL, att_name, att_type, att_len, att_val));
        
        free(att_val);
    }
    return 0;
}

/* Create output NetCDF file and copy metadata */
static int create_output_file(const char *output_path, int ncid_in,
                             MPI_Offset time_len, MPI_Offset y_len, MPI_Offset x_len,
                             int *ncid_out_ptr, int *flds_varid_out_ptr)
{
    int ncid_out;
    int time_dimid_out, y_dimid_out, x_dimid_out;
    int x_varid_out, y_varid_out, time_varid_out;
    int lat_varid_out, lon_varid_out, proj_varid_out;
    int flds_varid_out;
    
    /* Get input variable IDs */
    int x_varid_in, y_varid_in, time_varid_in;
    int lat_varid_in, lon_varid_in, proj_varid_in, flds_varid_in;
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "x", &x_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "y", &y_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "time", &time_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lat", &lat_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lon", &lon_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "lambert_conformal_conic", &proj_varid_in));
    CHECK_ERR(ncmpi_inq_varid(ncid_in, "FLDS", &flds_varid_in));
    
    /* Create output file */
    CHECK_ERR(ncmpi_create(MPI_COMM_WORLD, output_path,
                           NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, &ncid_out));

    /* Define dimensions */
    CHECK_ERR(ncmpi_def_dim(ncid_out, "time", time_len, &time_dimid_out));
    CHECK_ERR(ncmpi_def_dim(ncid_out, "y", y_len, &y_dimid_out));
    CHECK_ERR(ncmpi_def_dim(ncid_out, "x", x_len, &x_dimid_out));

    /* Define coordinate variables */
    CHECK_ERR(ncmpi_def_var(ncid_out, "x", NC_FLOAT, 1, &x_dimid_out, &x_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "y", NC_FLOAT, 1, &y_dimid_out, &y_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "time", NC_FLOAT, 1, &time_dimid_out, &time_varid_out));

    /* Define lat/lon variables */
    int latlon_dimids[2] = { y_dimid_out, x_dimid_out };
    CHECK_ERR(ncmpi_def_var(ncid_out, "lat", NC_FLOAT, 2, latlon_dimids, &lat_varid_out));
    CHECK_ERR(ncmpi_def_var(ncid_out, "lon", NC_FLOAT, 2, latlon_dimids, &lon_varid_out));

    /* Define projection variable (scalar) */
    CHECK_ERR(ncmpi_def_var(ncid_out, "lambert_conformal_conic", NC_SHORT, 0, NULL, &proj_varid_out));

    /* Define FLDS variable (uncompressed) */
    int flds_dimids[3] = { time_dimid_out, y_dimid_out, x_dimid_out };
    CHECK_ERR(ncmpi_def_var(ncid_out, "FLDS", NC_FLOAT, 3, flds_dimids, &flds_varid_out));

    /* Copy attributes */
    copy_var_atts(ncid_in, x_varid_in, ncid_out, x_varid_out);
    copy_var_atts(ncid_in, y_varid_in, ncid_out, y_varid_out);
    copy_var_atts(ncid_in, time_varid_in, ncid_out, time_varid_out);
    copy_var_atts(ncid_in, lat_varid_in, ncid_out, lat_varid_out);
    copy_var_atts(ncid_in, lon_varid_in, ncid_out, lon_varid_out);
    copy_var_atts(ncid_in, proj_varid_in, ncid_out, proj_varid_out);
    copy_var_atts(ncid_in, flds_varid_in, ncid_out, flds_varid_out);
    copy_global_atts(ncid_in, ncid_out);

    CHECK_ERR(ncmpi_enddef(ncid_out));

    /* Write coordinate variables (rank 0 only) */
    CHECK_ERR(ncmpi_begin_indep_data(ncid_in));
    CHECK_ERR(ncmpi_begin_indep_data(ncid_out));

    if (rank == 0) {
        float *buf_x = (float *)malloc((size_t)x_len * sizeof(float));
        float *buf_y = (float *)malloc((size_t)y_len * sizeof(float));
        float *buf_t = (float *)malloc((size_t)time_len * sizeof(float));
        float *buf_latlon = (float *)malloc((size_t)y_len * (size_t)x_len * sizeof(float));
        short proj;

        CHECK_ERR(ncmpi_get_var_float(ncid_in, x_varid_in, buf_x));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, y_varid_in, buf_y));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, time_varid_in, buf_t));
        CHECK_ERR(ncmpi_get_var_short(ncid_in, proj_varid_in, &proj));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, lat_varid_in, buf_latlon));

        CHECK_ERR(ncmpi_put_var_float(ncid_out, x_varid_out, buf_x));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, y_varid_out, buf_y));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, time_varid_out, buf_t));
        CHECK_ERR(ncmpi_put_var_short(ncid_out, proj_varid_out, &proj));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, lat_varid_out, buf_latlon));

        CHECK_ERR(ncmpi_get_var_float(ncid_in, lon_varid_in, buf_latlon));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, lon_varid_out, buf_latlon));

        free(buf_x);
        free(buf_y);
        free(buf_t);
        free(buf_latlon);
    }

    CHECK_ERR(ncmpi_end_indep_data(ncid_in));
    CHECK_ERR(ncmpi_end_indep_data(ncid_out));
    
    *ncid_out_ptr = ncid_out;
    *flds_varid_out_ptr = flds_varid_out;
    return NC_NOERR;
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s COMPRESSED_NC OUTPUT_PREFIX [eb1,eb2,eb3,...]\n", argv[0]);
            fprintf(stderr, "\nExample:\n");
            fprintf(stderr, "  mpirun -n 4 %s FLDS_ipcomp_4.nc FLDS_decompressed 1e-2,1e-4,1e-6\n", argv[0]);
            fprintf(stderr, "\nThis creates files with different precision levels:\n");
            fprintf(stderr, "  - FLDS_decompressed_eb1e-02.nc (coarse, smaller transfer)\n");
            fprintf(stderr, "  - FLDS_decompressed_eb1e-04.nc (medium)\n");
            fprintf(stderr, "  - FLDS_decompressed_eb1e-06.nc (fine, larger transfer)\n");
            fprintf(stderr, "\nDefault error bounds: 1e-2, 1e-3, 1e-4\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_prefix = argv[2];
    
    /* Parse error bounds */
    double error_bounds[MAX_ERROR_BOUNDS];
    int num_ebs = 0;
    
    if (argc >= 4) {
        num_ebs = parse_error_bounds(argv[3], error_bounds, MAX_ERROR_BOUNDS);
    }
    
    /* Default error bounds if not specified */
    if (num_ebs == 0) {
        error_bounds[0] = 1e-2;
        error_bounds[1] = 1e-3;
        error_bounds[2] = 1e-4;
        num_ebs = 3;
    }

    double total_start = MPI_Wtime();

    /* Process each error bound */
    for (int eb_idx = 0; eb_idx < num_ebs; eb_idx++) {
        double target_eb = error_bounds[eb_idx];
        
        if (rank == 0) {
            printf("--------------------------------------------------\n");
            printf("Processing error bound %d/%d: %e\n", eb_idx + 1, num_ebs, target_eb);
            printf("--------------------------------------------------\n");
            fflush(stdout);
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        double phase_start = MPI_Wtime();
        double t_open_in = 0.0;
        double t_meta = 0.0;
        double t_create_out = 0.0;
        double t_setup = 0.0;
        double t_begin_indep = 0.0;
        double t_pre_barrier = 0.0;
        double t_end_indep = 0.0;
        double t_close = 0.0;
        double t_clear_decomp = 0.0;
        double t_post_barrier = 0.0;
        double pre_total = 0.0;
        double post_total = 0.0;
        double pre_other = 0.0;
        double post_other = 0.0;
        double t0 = 0.0;

        /* Open compressed input for this error bound (fresh cache state) */
        MPI_Info info;
        MPI_Info_create(&info);
        MPI_Info_set(info, "nc_chunking", "enable");
        MPI_Info_set(info, "nc_chk_indep_owner_get", "1");

        int ncid_in;
        t0 = MPI_Wtime();
        CHECK_ERR(ncmpi_open(MPI_COMM_WORLD, input_path, NC_NOWRITE, info, &ncid_in));
        t_open_in = MPI_Wtime() - t0;
        MPI_Info_free(&info);

        t0 = MPI_Wtime();
        /* Get dimensions */
        int time_dimid, y_dimid, x_dimid;
        CHECK_ERR(ncmpi_inq_dimid(ncid_in, "time", &time_dimid));
        CHECK_ERR(ncmpi_inq_dimid(ncid_in, "y", &y_dimid));
        CHECK_ERR(ncmpi_inq_dimid(ncid_in, "x", &x_dimid));

        MPI_Offset time_len, y_len, x_len;
        CHECK_ERR(ncmpi_inq_dimlen(ncid_in, time_dimid, &time_len));
        CHECK_ERR(ncmpi_inq_dimlen(ncid_in, y_dimid, &y_len));
        CHECK_ERR(ncmpi_inq_dimlen(ncid_in, x_dimid, &x_len));

        int flds_varid;
        CHECK_ERR(ncmpi_inq_varid(ncid_in, "FLDS", &flds_varid));

        /* Read _FillValue attribute if present */
        int has_fill_value = 0;
        float fill_value = NC_FILL_FLOAT;
        nc_type fill_type;
        MPI_Offset fill_len;
        int fill_ret = ncmpi_inq_att(ncid_in, flds_varid, "_FillValue", &fill_type, &fill_len);
        if (fill_ret == NC_NOERR && fill_len == 1) {
            if (fill_type == NC_FLOAT) {
                CHECK_ERR(ncmpi_get_att_float(ncid_in, flds_varid, "_FillValue", &fill_value));
                has_fill_value = 1;
            } else if (fill_type == NC_DOUBLE) {
                double tmp_fill = 0.0;
                CHECK_ERR(ncmpi_get_att_double(ncid_in, flds_varid, "_FillValue", &tmp_fill));
                fill_value = (float)tmp_fill;
                has_fill_value = 1;
            }
        }

        /* Discover physical chunk dimensions (default to full slab if absent) */
        MPI_Offset chunk_shape[3] = {1, y_len, x_len};
        long long chunk_vals[3] = {0, 0, 0};
        int chunk_err = ncmpi_get_att_longlong(ncid_in, flds_varid, "_chunkdim", chunk_vals);
        if (chunk_err == NC_NOERR) {
            MPI_Offset dim_lens[3] = {time_len, y_len, x_len};
            for (int i = 0; i < 3; i++) {
                if (chunk_vals[i] > 0 && (MPI_Offset)chunk_vals[i] <= dim_lens[i]) {
                    chunk_shape[i] = (MPI_Offset)chunk_vals[i];
                } else {
                    chunk_shape[i] = dim_lens[i];
                }
            }
        } else if (chunk_err != NC_ENOTATT) {
            fprintf(stderr, "[%d] Failed to read _chunkdim attribute: %s\n",
                    rank, ncmpi_strerror(chunk_err));
            MPI_Abort(MPI_COMM_WORLD, chunk_err);
        }

        MPI_Offset chunk_grid[3];
        chunk_grid[0] = (time_len + chunk_shape[0] - 1) / chunk_shape[0];
        chunk_grid[1] = (y_len + chunk_shape[1] - 1) / chunk_shape[1];
        chunk_grid[2] = (x_len + chunk_shape[2] - 1) / chunk_shape[2];
        MPI_Offset total_chunks = chunk_grid[0] * chunk_grid[1] * chunk_grid[2];
        int use_block_owner = env_enabled("PNETCDF_DECOMP_BLOCK_OWNERSHIP");

        /* Read compression attributes */
        int layers = 1;
        double data_range = 0.0;
        ncmpi_get_att_int(ncid_in, flds_varid, "comp:layers", &layers);
        ncmpi_get_att_double(ncid_in, flds_varid, "comp:data_range", &data_range);

        double meta_min = NAN;
        double meta_max = NAN;
        if (ncmpi_get_att_double(ncid_in, flds_varid, "comp:data_min", &meta_min) != NC_NOERR) {
            meta_min = NAN;
        }
        if (ncmpi_get_att_double(ncid_in, flds_varid, "comp:data_max", &meta_max) != NC_NOERR) {
            meta_max = NAN;
        }
        int use_meta_stats = (!isnan(meta_min) && !isnan(meta_max) && meta_max >= meta_min);

        if (rank == 0 && eb_idx == 0) {
            printf("========================================================\n");
            printf("Progressive Decompression with Error Bounds\n");
            printf("========================================================\n");
            printf("Input file: %s\n", input_path);
            printf("Dimensions: time=%lld, y=%lld, x=%lld\n",
                   (long long)time_len, (long long)y_len, (long long)x_len);
            printf("Total elements: %lld (%.2f GB uncompressed)\n",
                   (long long)(time_len * y_len * x_len),
                   (double)(time_len * y_len * x_len * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
            printf("Compression layers: %d\n", layers);
            printf("Data range: %g\n", data_range);
            printf("Processes: %d\n", nprocs);
            printf("\nTarget error bounds (%d):\n", num_ebs);
            for (int i = 0; i < num_ebs; i++) {
                printf("  [%d] %e (relative)\n", i + 1, error_bounds[i]);
            }
            printf("========================================================\n\n");
            printf("Chunk layout (t,y,x): %lld x %lld x %lld\n",
                   (long long)chunk_shape[0], (long long)chunk_shape[1], (long long)chunk_shape[2]);
            printf("Chunk grid: %lld x %lld x %lld (total %lld chunks)\n",
                   (long long)chunk_grid[0], (long long)chunk_grid[1], (long long)chunk_grid[2],
                   (long long)total_chunks);
            if (use_block_owner) {
                printf("Chunk ownership: block (contiguous)\n\n");
            } else {
                printf("Chunk ownership: chunk_id %% %d (round-robin)\n\n", nprocs);
            }
        }

        MPI_Offset total_elems = time_len * y_len * x_len;
        size_t chunk_capacity = (size_t)chunk_shape[0] *
                                (size_t)chunk_shape[1] *
                                (size_t)chunk_shape[2];
        if (chunk_capacity == 0) {
            fprintf(stderr, "[%d] Invalid chunk capacity (shape %lld x %lld x %lld)\n",
                    rank,
                    (long long)chunk_shape[0],
                    (long long)chunk_shape[1],
                    (long long)chunk_shape[2]);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        MPI_Offset progress_stride = total_chunks / 10;
        if (progress_stride == 0) progress_stride = 1;

        /* Set target error bound for progressive decompression */
        int err = ncmpi_var_set_decomp_error_bound(ncid_in, flds_varid, target_eb);
        if (err != NC_NOERR && rank == 0) {
            fprintf(stderr, "Warning: Failed to set decomp error bound: %s\n", ncmpi_strerror(err));
        }
        t_meta = MPI_Wtime() - t0;
        
        /* Generate output filename */
        char output_path[512];
        t0 = MPI_Wtime();
        snprintf(output_path, sizeof(output_path), "%s_eb%.0e.nc", output_prefix, target_eb);
        int ncid_out, flds_varid_out;
        CHECK_ERR(create_output_file(output_path, ncid_in, time_len, y_len, x_len,
                                     &ncid_out, &flds_varid_out));
        t_create_out = MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        double local_min = DBL_MAX;
        double local_max = -DBL_MAX;
        double local_sum = 0.0;
        long long local_valid = 0;
        double t_read = 0.0;
        double t_write = 0.0;
        MPI_Offset chunk_id = 0;
        int use_coll_write = env_enabled("PNETCDF_DECOMP_COLLECTIVE_WRITE");
        int use_batch_write = 0;
        int batch_all = 0;
        long batch_req = 0;
        int batch_parse = env_parse_batch("PNETCDF_DECOMP_COLLECTIVE_WRITE_BATCH",
                                          &batch_req, &batch_all);
        if (batch_parse < 0 && rank == 0) {
            fprintf(stderr,
                    "Warning: invalid PNETCDF_DECOMP_COLLECTIVE_WRITE_BATCH; "
                    "use a positive integer or 'all'\n");
        }
        if (batch_parse > 0) {
            if (batch_all || batch_req < 0 || batch_req > 0) {
                use_batch_write = 1;
                use_coll_write = 1;
            } else if (rank == 0) {
                fprintf(stderr,
                        "Warning: PNETCDF_DECOMP_COLLECTIVE_WRITE_BATCH must be >0 or 'all'\n");
            }
        }

        MPI_Offset local_chunks = total_chunks / nprocs;
        MPI_Offset remainder = total_chunks % nprocs;
        if ((MPI_Offset)rank < remainder) local_chunks++;

        size_t batch_capacity_chunks = 0;
        MPI_Offset batch_global = 0;
        size_t batch_capacity_elems = 0;
        size_t batch_elems_used = 0;
        int batch_num = 0;
        float *chunk_buffer = NULL;
        float *batch_buffer = NULL;
        MPI_Offset **batch_starts = NULL;
        MPI_Offset **batch_counts = NULL;
        MPI_Offset *batch_starts_store = NULL;
        MPI_Offset *batch_counts_store = NULL;
        float empty_buf = 0.0f;

        if (use_batch_write) {
            if (batch_all || batch_req < 0) {
                batch_capacity_chunks = (size_t)local_chunks;
                batch_global = total_chunks;
            } else if (batch_req > 0) {
                batch_capacity_chunks = (size_t)batch_req;
                batch_global = (MPI_Offset)batch_req * (MPI_Offset)nprocs;
            } else {
                use_batch_write = 0;
            }
        }

        if (use_batch_write && batch_capacity_chunks > 0) {
            size_t max_size = (size_t)-1;
            if (chunk_capacity > max_size / batch_capacity_chunks) {
                fprintf(stderr, "[%d] Batch buffer size overflow\n", rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            batch_capacity_elems = chunk_capacity * batch_capacity_chunks;
            batch_buffer = (float *)malloc(batch_capacity_elems * sizeof(float));
            if (batch_buffer == NULL) {
                fprintf(stderr, "[%d] Failed to allocate %.2f MB batch buffer\n",
                        rank,
                        (double)batch_capacity_elems * sizeof(float) / (1024.0 * 1024.0));
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            batch_starts_store = (MPI_Offset *)malloc(batch_capacity_chunks * 3 * sizeof(MPI_Offset));
            batch_counts_store = (MPI_Offset *)malloc(batch_capacity_chunks * 3 * sizeof(MPI_Offset));
            batch_starts = (MPI_Offset **)malloc(batch_capacity_chunks * sizeof(MPI_Offset *));
            batch_counts = (MPI_Offset **)malloc(batch_capacity_chunks * sizeof(MPI_Offset *));
            if (batch_starts_store == NULL || batch_counts_store == NULL ||
                batch_starts == NULL || batch_counts == NULL) {
                fprintf(stderr, "[%d] Failed to allocate batch metadata\n", rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            for (size_t i = 0; i < batch_capacity_chunks; i++) {
                batch_starts[i] = batch_starts_store + i * 3;
                batch_counts[i] = batch_counts_store + i * 3;
            }
        }

        if (!use_batch_write) {
            chunk_buffer = (float *)malloc(chunk_capacity * sizeof(float));
            if (chunk_buffer == NULL) {
                fprintf(stderr, "[%d] Failed to allocate %.2f MB chunk buffer\n",
                        rank, (double)chunk_capacity * sizeof(float) / (1024.0 * 1024.0));
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }

        if (rank == 0) {
            if (use_batch_write) {
                if (batch_all || batch_req < 0) {
                    printf("Write mode: collective (put_varn_all, batch=all)\n");
                } else {
                    printf("Write mode: collective (put_varn_all, batch=%zu)\n", batch_capacity_chunks);
                }
            } else {
                printf("Write mode: %s\n",
                       use_coll_write ? "collective (put_vara_all)" : "independent (owner-only)");
            }
        }
        t_setup = MPI_Wtime() - t0;

        /*
         * Owner-only independent get:
         * Only the owning rank reads each chunk. Writes can be independent or
         * collective depending on the write mode.
         */
        t0 = MPI_Wtime();
        CHECK_ERR(ncmpi_begin_indep_data(ncid_in));
        if (!use_coll_write) {
            CHECK_ERR(ncmpi_begin_indep_data(ncid_out));
        }
        t_begin_indep = MPI_Wtime() - t0;
        t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        t_pre_barrier = MPI_Wtime() - t0;
#ifdef ENABLE_IPCOMP
        ipcomp_reset_core_timers();
#endif
        double chunk_start_time = MPI_Wtime();

        for (MPI_Offset t = 0; t < time_len; t += chunk_shape[0]) {
            MPI_Offset t_count = chunk_shape[0];
            if (t + t_count > time_len) t_count = time_len - t;

            for (MPI_Offset y = 0; y < y_len; y += chunk_shape[1]) {
                MPI_Offset y_count = chunk_shape[1];
                if (y + y_count > y_len) y_count = y_len - y;

                for (MPI_Offset x = 0; x < x_len; x += chunk_shape[2]) {
                    MPI_Offset x_count = chunk_shape[2];
                    if (x + x_count > x_len) x_count = x_len - x;

                    int owner = get_chunk_owner(chunk_id, nprocs, total_chunks, use_block_owner);
                    int owns_chunk = (rank == owner);

                    MPI_Offset read_start[3] = {0, 0, 0};
                    MPI_Offset read_count[3] = {0, 0, 0};
                    size_t chunk_elems = 0;
                    float *chunk_ptr = NULL;
                    if (owns_chunk) {
                        read_start[0] = t;
                        read_start[1] = y;
                        read_start[2] = x;
                        read_count[0] = t_count;
                        read_count[1] = y_count;
                        read_count[2] = x_count;
                        chunk_elems = (size_t)read_count[0] *
                                      (size_t)read_count[1] *
                                      (size_t)read_count[2];
                        if (use_batch_write) {
                            if (batch_num >= (int)batch_capacity_chunks) {
                                fprintf(stderr, "[%d] Batch buffer overflow: %d/%zu chunks\n",
                                        rank, batch_num, batch_capacity_chunks);
                                MPI_Abort(MPI_COMM_WORLD, -1);
                            }
                            if (batch_elems_used + chunk_elems > batch_capacity_elems) {
                                fprintf(stderr, "[%d] Batch element overflow: %zu + %zu > %zu\n",
                                        rank, batch_elems_used, chunk_elems, batch_capacity_elems);
                                MPI_Abort(MPI_COMM_WORLD, -1);
                            }
                            if (batch_buffer == NULL) {
                                fprintf(stderr, "[%d] Batch buffer not allocated\n", rank);
                                MPI_Abort(MPI_COMM_WORLD, -1);
                            }
                            chunk_ptr = batch_buffer + batch_elems_used;
                        } else {
                            chunk_ptr = chunk_buffer;
                        }
                    }

                    if (owns_chunk) {
                        fprintf(stderr,
                                "[Rank %d] reading chunk %lld (t=%lld..%lld, y=%lld..%lld)\n",
                                rank, (long long)chunk_id,
                                (long long)read_start[0], (long long)(read_start[0] + read_count[0] - 1),
                                (long long)read_start[1], (long long)(read_start[1] + read_count[1] - 1));
                        fflush(stderr);
                    }

                    if (owns_chunk) {
                        double t0 = MPI_Wtime();
                        CHECK_ERR(ncmpi_get_vara_float(ncid_in, flds_varid,
                                                       read_start, read_count, chunk_ptr));
                        t_read += MPI_Wtime() - t0;
                    }

                    if (owns_chunk) {
                        if (!use_meta_stats) {
                            for (size_t idx = 0; idx < chunk_elems; idx++) {
                                float val = chunk_ptr[idx];
                                if ((has_fill_value && val == fill_value) ||
                                    isnan(val) || isinf(val)) {
                                    continue;
                                }
                                if (val < local_min) local_min = val;
                                if (val > local_max) local_max = val;
                                local_sum += val;
                                local_valid++;
                            }
                        }
                    }

                    MPI_Offset write_start[3] = {0, 0, 0};
                    MPI_Offset write_count[3] = {0, 0, 0};
                    if (owns_chunk) {
                        write_start[0] = read_start[0];
                        write_start[1] = read_start[1];
                        write_start[2] = read_start[2];
                        write_count[0] = read_count[0];
                        write_count[1] = read_count[1];
                        write_count[2] = read_count[2];
                    }

                    if (use_batch_write) {
                        if (owns_chunk) {
                            MPI_Offset *start = batch_starts_store + (size_t)batch_num * 3;
                            MPI_Offset *count = batch_counts_store + (size_t)batch_num * 3;
                            start[0] = write_start[0];
                            start[1] = write_start[1];
                            start[2] = write_start[2];
                            count[0] = write_count[0];
                            count[1] = write_count[1];
                            count[2] = write_count[2];
                            batch_starts[batch_num] = start;
                            batch_counts[batch_num] = count;
                            batch_num++;
                            batch_elems_used += chunk_elems;
                        }
                    } else if (use_coll_write) {
                        double t0 = MPI_Wtime();
                        CHECK_ERR(ncmpi_put_vara_float_all(ncid_out, flds_varid_out,
                                                           write_start, write_count, chunk_buffer));
                        t_write += MPI_Wtime() - t0;
                    } else if (owns_chunk) {
                        double t0 = MPI_Wtime();
                        CHECK_ERR(ncmpi_put_vara_float(ncid_out, flds_varid_out,
                                                       write_start, write_count, chunk_buffer));
                        t_write += MPI_Wtime() - t0;
                    }

                    if (owns_chunk) {
                        fprintf(stderr, "[Rank %d] finished chunk %lld\n",
                                rank, (long long)chunk_id);
                        fflush(stderr);
                    }

                    chunk_id++;
                    if (use_batch_write && batch_global > 0) {
                        int flush_now = (chunk_id == total_chunks) ||
                                        (batch_global > 0 && (chunk_id % batch_global == 0));
                        if (flush_now) {
                            double t0 = MPI_Wtime();
                            float *write_buf = (batch_buffer != NULL) ? batch_buffer : &empty_buf;
                            CHECK_ERR(ncmpi_put_varn_float_all(ncid_out, flds_varid_out,
                                                               batch_num,
                                                               batch_num ? batch_starts : NULL,
                                                               batch_num ? batch_counts : NULL,
                                                               write_buf));
                            t_write += MPI_Wtime() - t0;
                            batch_num = 0;
                            batch_elems_used = 0;
                        }
                    }
                    if (rank == 0 &&
                        (chunk_id % progress_stride == 0 || chunk_id == total_chunks)) {
                        double pct = 100.0 * (double)chunk_id / (double)total_chunks;
                        printf("  > %.0f%% chunks complete (%lld/%lld)\n",
                               pct, (long long)chunk_id, (long long)total_chunks);
                        fflush(stdout);
                    }
                }
            }
        }

        double chunk_end_time = MPI_Wtime();
        double chunk_elapsed = chunk_end_time - chunk_start_time;
#ifdef ENABLE_IPCOMP
        double core_decomp = ipcomp_get_core_decompress_time();
#else
        double core_decomp = 0.0;
#endif
        double read_io = t_read - core_decomp;
        if (read_io < 0.0) read_io = 0.0;

        t0 = MPI_Wtime();
        if (!use_coll_write) {
            CHECK_ERR(ncmpi_end_indep_data(ncid_out));
        }
        CHECK_ERR(ncmpi_end_indep_data(ncid_in));
        t_end_indep = MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        CHECK_ERR(ncmpi_close(ncid_out));
        t_close = MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        ncmpi_var_clear_decomp_error_bound(ncid_in, flds_varid);
        t_clear_decomp = MPI_Wtime() - t0;

        t0 = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        t_post_barrier = MPI_Wtime() - t0;
        double phase_end = MPI_Wtime();
        pre_total = chunk_start_time - phase_start;
        post_total = phase_end - chunk_end_time;
        pre_other = pre_total - (t_open_in + t_meta + t_create_out + t_setup + t_begin_indep + t_pre_barrier);
        if (pre_other < 0.0) pre_other = 0.0;
        post_other = post_total - (t_end_indep + t_close + t_clear_decomp + t_post_barrier);
        if (post_other < 0.0) post_other = 0.0;

        double global_min = 0.0, global_max = 0.0, global_sum = 0.0;
        long long global_valid = 0;

        if (!use_meta_stats) {
            MPI_Reduce(&local_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            MPI_Reduce(&local_valid, &global_valid, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        }

        double *rank_times = NULL;
        double *rank_read_io = NULL;
        double *rank_core = NULL;
        double *rank_write = NULL;
        if (rank == 0) {
            rank_times = (double *)malloc((size_t)nprocs * sizeof(double));
            rank_read_io = (double *)malloc((size_t)nprocs * sizeof(double));
            rank_core = (double *)malloc((size_t)nprocs * sizeof(double));
            rank_write = (double *)malloc((size_t)nprocs * sizeof(double));
            if (rank_times == NULL || rank_read_io == NULL ||
                rank_core == NULL || rank_write == NULL) {
                fprintf(stderr, "[%d] Failed to allocate rank timing buffer\n", rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
        MPI_Gather(&chunk_elapsed, 1, MPI_DOUBLE,
                   rank_times, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&read_io, 1, MPI_DOUBLE,
                   rank_read_io, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&core_decomp, 1, MPI_DOUBLE,
                   rank_core, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gather(&t_write, 1, MPI_DOUBLE,
                   rank_write, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        const int metric_count = 14;
        double metrics[14] = {
            pre_total,
            t_open_in,
            t_meta,
            t_create_out,
            t_setup,
            t_begin_indep,
            t_pre_barrier,
            pre_other,
            post_total,
            t_end_indep,
            t_close,
            t_clear_decomp,
            t_post_barrier,
            post_other
        };
        double metrics_min[14];
        double metrics_max[14];
        double metrics_sum[14];
        MPI_Reduce(metrics, metrics_min, metric_count, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(metrics, metrics_max, metric_count, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
        MPI_Reduce(metrics, metrics_sum, metric_count, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            const char *metric_names[14] = {
                "pre_total",
                "pre_open",
                "pre_meta",
                "pre_create_out",
                "pre_setup",
                "pre_begin_indep",
                "pre_barrier",
                "pre_other",
                "post_total",
                "post_end_indep",
                "post_close",
                "post_clear_decomp",
                "post_barrier",
                "post_other"
            };
            printf("End-to-end time: %.2f seconds\n", phase_end - phase_start);
            printf("Per-rank breakdown (seconds):\n");
            for (int r = 0; r < nprocs; r++) {
                printf("  rank %d: total=%.6f read_io=%.6f core_decomp=%.6f write_io=%.6f\n",
                       r, rank_times[r], rank_read_io[r], rank_core[r], rank_write[r]);
            }
            if (nprocs > 0) {
                double total_min = rank_times[0], total_max = rank_times[0], total_sum = 0.0;
                double read_min = rank_read_io[0], read_max = rank_read_io[0], read_sum = 0.0;
                double core_min = rank_core[0], core_max = rank_core[0], core_sum = 0.0;
                double write_min = rank_write[0], write_max = rank_write[0], write_sum = 0.0;
                for (int r = 0; r < nprocs; r++) {
                    double total = rank_times[r];
                    double readv = rank_read_io[r];
                    double corev = rank_core[r];
                    double writev = rank_write[r];
                    if (total < total_min) total_min = total;
                    if (total > total_max) total_max = total;
                    if (readv < read_min) read_min = readv;
                    if (readv > read_max) read_max = readv;
                    if (corev < core_min) core_min = corev;
                    if (corev > core_max) core_max = corev;
                    if (writev < write_min) write_min = writev;
                    if (writev > write_max) write_max = writev;
                    total_sum += total;
                    read_sum += readv;
                    core_sum += corev;
                    write_sum += writev;
                }
                printf("Timing stats across ranks (seconds):\n");
                printf("  total: min=%.6f max=%.6f avg=%.6f\n",
                       total_min, total_max, total_sum / (double)nprocs);
                printf("  read_io: min=%.6f max=%.6f avg=%.6f\n",
                       read_min, read_max, read_sum / (double)nprocs);
                printf("  core_decomp: min=%.6f max=%.6f avg=%.6f\n",
                       core_min, core_max, core_sum / (double)nprocs);
                printf("  write_io: min=%.6f max=%.6f avg=%.6f\n",
                       write_min, write_max, write_sum / (double)nprocs);
            }
            if (nprocs > 0) {
                printf("End-to-end overhead breakdown (seconds):\n");
                for (int i = 0; i < metric_count; i++) {
                    printf("  %s: min=%.6f max=%.6f avg=%.6f\n",
                           metric_names[i],
                           metrics_min[i],
                           metrics_max[i],
                           metrics_sum[i] / (double)nprocs);
                }
            }
            if (use_meta_stats) {
                printf("Data statistics (from metadata):\n");
                printf("  Min: %.6f\n", meta_min);
                printf("  Max: %.6f\n", meta_max);
                printf("  Range: %.6f\n", meta_max - meta_min);
                printf("  Valid elements: %lld (metadata)\n", (long long)total_elems);
            } else if (global_valid > 0) {
                double mean = global_sum / (double)global_valid;
                printf("Data statistics:\n");
                printf("  Min: %.6f\n", global_min);
                printf("  Max: %.6f\n", global_max);
                printf("  Mean: %.6f\n", mean);
                printf("  Valid elements: %lld / %lld (%.2f%%)\n",
                       global_valid, (long long)total_elems,
                       100.0 * (double)global_valid / (double)total_elems);
            } else {
                printf("Data statistics: no finite values detected\n");
            }
            printf("✓ Complete: %s\n\n", output_path);
            fflush(stdout);
        }

        if (rank_times != NULL) {
            free(rank_times);
        }
        if (rank_read_io != NULL) {
            free(rank_read_io);
        }
        if (rank_core != NULL) {
            free(rank_core);
        }
        if (rank_write != NULL) {
            free(rank_write);
        }

        if (chunk_buffer != NULL) {
            free(chunk_buffer);
        }
        if (batch_buffer != NULL) {
            free(batch_buffer);
        }
        if (batch_starts != NULL) {
            free(batch_starts);
        }
        if (batch_counts != NULL) {
            free(batch_counts);
        }
        if (batch_starts_store != NULL) {
            free(batch_starts_store);
        }
        if (batch_counts_store != NULL) {
            free(batch_counts_store);
        }

        CHECK_ERR(ncmpi_close(ncid_in));
    }

    double total_end = MPI_Wtime();

    if (rank == 0) {
        printf("========================================================\n");
        printf("Progressive Decompression Complete!\n");
        printf("========================================================\n");
        printf("Total time: %.2f seconds\n", total_end - total_start);
        printf("Output files created: %d\n", num_ebs);
        for (int i = 0; i < num_ebs; i++) {
            printf("  - %s_eb%.0e.nc (error bound: %e)\n", output_prefix, error_bounds[i], error_bounds[i]);
        }
        printf("========================================================\n");
    }

    MPI_Finalize();
    return 0;
}

