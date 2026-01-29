#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <pnetcdf.h>

#define CHECK_ERR(call)                                                              \
    do {                                                                             \
        int _status = (call);                                                        \
        if (_status != NC_NOERR) {                                                   \
            fprintf(stderr, "[%d] PnetCDF error at %s:%d: %s\n",                     \
                    rank, __FILE__, __LINE__, ncmpi_strerror(_status));              \
            MPI_Abort(MPI_COMM_WORLD, _status);                                      \
        }                                                                            \
    } while (0)

/* Copy all attributes from one variable to another */
static int copy_var_atts(int ncid_in, int varid_in, int ncid_out, int varid_out, int rank)
{
    int natts;
    CHECK_ERR(ncmpi_inq_varnatts(ncid_in, varid_in, &natts));
    
    for (int i = 0; i < natts; i++) {
        char att_name[NC_MAX_NAME + 1];
        nc_type att_type;
        MPI_Offset att_len;
        
        CHECK_ERR(ncmpi_inq_attname(ncid_in, varid_in, i, att_name));
        
        /* Skip internal attributes except _FillValue */
        int is_fill_value = (strcmp(att_name, "_FillValue") == 0);
        if (att_name[0] == '_' && !is_fill_value) continue;
        
        CHECK_ERR(ncmpi_inq_att(ncid_in, varid_in, att_name, &att_type, &att_len));
        
        void *att_val = malloc((size_t)att_len * 8); /* max size for any type */
        if (att_val == NULL) continue;
        
        CHECK_ERR(ncmpi_get_att(ncid_in, varid_in, att_name, att_val));
        CHECK_ERR(ncmpi_put_att(ncid_out, varid_out, att_name, att_type, att_len, att_val));
        
        free(att_val);
    }
    return 0;
}

/* Copy global attributes */
static int copy_global_atts(int ncid_in, int ncid_out, int rank)
{
    int natts;
    CHECK_ERR(ncmpi_inq_natts(ncid_in, &natts));
    
    for (int i = 0; i < natts; i++) {
        char att_name[NC_MAX_NAME + 1];
        nc_type att_type;
        MPI_Offset att_len;
        
        CHECK_ERR(ncmpi_inq_attname(ncid_in, NC_GLOBAL, i, att_name));
        
        /* Skip internal attributes starting with '_' */
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

int main(int argc, char **argv)
{
    int rank, nprocs;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 2) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s COMPRESSED_NC [OUTPUT_NC]\n", argv[0]);
            fprintf(stderr, "  COMPRESSED_NC: Input compressed file\n");
            fprintf(stderr, "  OUTPUT_NC: Optional output file for decompressed data\n");
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = (argc >= 3) ? argv[2] : NULL;

    double total_start = MPI_Wtime();

    /* Open compressed input file */
    int ncid_in;
    CHECK_ERR(ncmpi_open(MPI_COMM_WORLD, input_path, NC_NOWRITE, MPI_INFO_NULL, &ncid_in));

    /* Get dimensions */
    int time_dimid, y_dimid, x_dimid;
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "time", &time_dimid));
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "y", &y_dimid));
    CHECK_ERR(ncmpi_inq_dimid(ncid_in, "x", &x_dimid));

    MPI_Offset time_len, y_len, x_len;
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, time_dimid, &time_len));
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, y_dimid, &y_len));
    CHECK_ERR(ncmpi_inq_dimlen(ncid_in, x_dimid, &x_len));

    /* Get variable IDs from input */
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

    /* Capture _FillValue if available */
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

    if (rank == 0) {
        printf("========================================\n");
        printf("Decompression Test\n");
        printf("========================================\n");
        printf("Input file: %s\n", input_path);
        printf("Dimensions: time=%lld, y=%lld, x=%lld\n",
               (long long)time_len, (long long)y_len, (long long)x_len);
        printf("Total elements: %lld (%.2f GB uncompressed)\n",
               (long long)(time_len * y_len * x_len),
               (double)(time_len * y_len * x_len * sizeof(float)) / (1024.0 * 1024.0 * 1024.0));
        printf("Processes: %d\n", nprocs);
        printf("========================================\n\n");
    }

    /* Create output file if requested */
    int ncid_out = -1;
    int x_varid_out, y_varid_out, time_varid_out;
    int lat_varid_out, lon_varid_out, proj_varid_out;
    int flds_varid_out = -1;
    
    if (output_path != NULL) {
        CHECK_ERR(ncmpi_create(MPI_COMM_WORLD, output_path,
                               NC_CLOBBER | NC_64BIT_DATA, MPI_INFO_NULL, &ncid_out));

        /* Define dimensions - use UNLIMITED for time like original */
        int time_dimid_out, y_dimid_out, x_dimid_out;
        CHECK_ERR(ncmpi_def_dim(ncid_out, "x", x_len, &x_dimid_out));
        CHECK_ERR(ncmpi_def_dim(ncid_out, "y", y_len, &y_dimid_out));
        CHECK_ERR(ncmpi_def_dim(ncid_out, "time", NC_UNLIMITED, &time_dimid_out));

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

        /* Define FLDS variable */
        int flds_dimids[3] = { time_dimid_out, y_dimid_out, x_dimid_out };
        CHECK_ERR(ncmpi_def_var(ncid_out, "FLDS", NC_FLOAT, 3, flds_dimids, &flds_varid_out));

        /* Copy attributes from input to output */
        copy_var_atts(ncid_in, x_varid_in, ncid_out, x_varid_out, rank);
        copy_var_atts(ncid_in, y_varid_in, ncid_out, y_varid_out, rank);
        copy_var_atts(ncid_in, time_varid_in, ncid_out, time_varid_out, rank);
        copy_var_atts(ncid_in, lat_varid_in, ncid_out, lat_varid_out, rank);
        copy_var_atts(ncid_in, lon_varid_in, ncid_out, lon_varid_out, rank);
        copy_var_atts(ncid_in, proj_varid_in, ncid_out, proj_varid_out, rank);
        copy_var_atts(ncid_in, flds_varid_in, ncid_out, flds_varid_out, rank);

        /* Copy global attributes */
        copy_global_atts(ncid_in, ncid_out, rank);

        CHECK_ERR(ncmpi_enddef(ncid_out));

        if (rank == 0) {
            printf("Output file: %s (uncompressed, with full metadata)\n\n", output_path);
        }
    }

    /* Write coordinate variables (rank 0 only) */
    CHECK_ERR(ncmpi_begin_indep_data(ncid_in));
    if (ncid_out >= 0) {
        CHECK_ERR(ncmpi_begin_indep_data(ncid_out));
    }

    if (rank == 0 && ncid_out >= 0) {
        /* Read and write x coordinate */
        float *buf_x = (float *)malloc((size_t)x_len * sizeof(float));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, x_varid_in, buf_x));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, x_varid_out, buf_x));
        free(buf_x);

        /* Read and write y coordinate */
        float *buf_y = (float *)malloc((size_t)y_len * sizeof(float));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, y_varid_in, buf_y));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, y_varid_out, buf_y));
        free(buf_y);

        /* Read and write time coordinate */
        float *buf_time = (float *)malloc((size_t)time_len * sizeof(float));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, time_varid_in, buf_time));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, time_varid_out, buf_time));
        free(buf_time);

        /* Read and write lat/lon */
        float *buf_latlon = (float *)malloc((size_t)y_len * (size_t)x_len * sizeof(float));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, lat_varid_in, buf_latlon));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, lat_varid_out, buf_latlon));
        CHECK_ERR(ncmpi_get_var_float(ncid_in, lon_varid_in, buf_latlon));
        CHECK_ERR(ncmpi_put_var_float(ncid_out, lon_varid_out, buf_latlon));
        free(buf_latlon);

        /* Read and write projection variable */
        short proj;
        CHECK_ERR(ncmpi_get_var_short(ncid_in, proj_varid_in, &proj));
        CHECK_ERR(ncmpi_put_var_short(ncid_out, proj_varid_out, &proj));

        printf("[Rank 0] Wrote coordinate variables and metadata\n");
    }

    /* Distribute time steps among processes */
    int chunks_per_proc = (int)(time_len / nprocs);
    int remainder = (int)(time_len % nprocs);
    int my_start = rank * chunks_per_proc + (rank < remainder ? rank : remainder);
    int my_count = chunks_per_proc + (rank < remainder ? 1 : 0);

    if (rank == 0) {
        printf("Each process handles approximately %d time steps\n\n", chunks_per_proc);
    }

    /* Statistics */
    double data_min = 1e30, data_max = -1e30, data_sum = 0.0;
    long long data_count = 0;
    int valid_count = 0;

    double decompress_start = MPI_Wtime();

    MPI_Offset slice_size = y_len * x_len;
    float *buffer = (float *)malloc((size_t)slice_size * sizeof(float));
    if (buffer == NULL) {
        fprintf(stderr, "[%d] Failed to allocate buffer\n", rank);
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    for (int t = my_start; t < my_start + my_count; t++) {
        MPI_Offset start[3] = {t, 0, 0};
        MPI_Offset count[3] = {1, y_len, x_len};

        printf("[Rank %d] Decompressing time step %d/%lld...\n", rank, t, (long long)time_len - 1);
        fflush(stdout);

        /* Read and decompress */
        CHECK_ERR(ncmpi_get_vara_float(ncid_in, flds_varid_in, start, count, buffer));

        /* Compute statistics */
        for (MPI_Offset i = 0; i < slice_size; i++) {
            float val = buffer[i];
            if ((has_fill_value && val == fill_value) || isnan(val) || isinf(val)) {
                data_count++;
                continue;
            }
            if (val < data_min) data_min = val;
            if (val > data_max) data_max = val;
            data_sum += val;
            valid_count++;
            data_count++;
        }

        /* Write to output file if specified */
        if (ncid_out >= 0) {
            CHECK_ERR(ncmpi_put_vara_float(ncid_out, flds_varid_out, start, count, buffer));
        }

        printf("[Rank %d] Completed time step %d\n", rank, t);
        fflush(stdout);
    }

    free(buffer);

    double decompress_end = MPI_Wtime();

    CHECK_ERR(ncmpi_end_indep_data(ncid_in));
    if (ncid_out >= 0) {
        CHECK_ERR(ncmpi_end_indep_data(ncid_out));
    }

    /* Gather statistics from all processes */
    double global_min, global_max, global_sum;
    long long global_count, global_valid;

    MPI_Reduce(&data_min, &global_min, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&data_max, &global_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&data_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&data_count, &global_count, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    
    long long valid_ll = valid_count;
    MPI_Reduce(&valid_ll, &global_valid, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);

    double local_time = decompress_end - decompress_start;
    double max_time;
    MPI_Reduce(&local_time, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    /* Close files */
    CHECK_ERR(ncmpi_close(ncid_in));
    if (ncid_out >= 0) {
        CHECK_ERR(ncmpi_sync(ncid_out));
        CHECK_ERR(ncmpi_close(ncid_out));
    }

    double total_end = MPI_Wtime();

    if (rank == 0) {
        printf("\n========================================\n");
        printf("Decompression Complete!\n");
        printf("========================================\n");
        printf("Data Statistics:\n");
        if (global_valid > 0) {
            double mean = global_sum / (double)global_valid;
            printf("  Min value: %.6f\n", global_min);
            printf("  Max value: %.6f\n", global_max);
            printf("  Mean value: %.6f\n", mean);
        } else {
            printf("  No finite values detected in dataset.\n");
        }
        printf("  Total elements: %lld\n", global_count);
        printf("  Valid elements: %lld (%.2f%%)\n", global_valid,
               100.0 * global_valid / global_count);
        printf("\nTiming:\n");
        printf("  Decompression time: %.2f seconds\n", max_time);
        printf("  Total time: %.2f seconds\n", total_end - total_start);
        printf("  Throughput: %.2f MB/s (uncompressed)\n",
               (double)(global_count * sizeof(float)) / (1024.0 * 1024.0) / max_time);
        printf("========================================\n");
    }

    MPI_Finalize();
    return 0;
}
