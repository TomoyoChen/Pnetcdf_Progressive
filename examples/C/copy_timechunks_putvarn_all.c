/*
 * copy_timechunks_putvarn_all.c
 *
 * Copy an input NetCDF file to a new file. Only FLDS is chunked by time
 * (one time step per chunk). For FLDS, each rank reads its owned time steps,
 * then all ranks call a single ncmpi_put_varn_all to write. All other variables
 * are copied via rank-0 read + collective write.
 *
 * Prints detailed timing for read/write of FLDS.
 *
 * Usage:
 *   mpirun -n <nprocs> ./copy_timechunks_putvarn_all INPUT.nc OUTPUT.nc
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <mpi.h>
#include <pnetcdf.h>

/* MAX_DIMS == NC_MAX_INT (2^31-1) in this PnetCDF build, which is
   way too large for stack arrays.  Use a sane limit instead. */
#define MAX_DIMS 32

#define CHECK_ERR(call)                                                          \
    do {                                                                         \
        int _status = (call);                                                    \
        if (_status != NC_NOERR) {                                               \
            fprintf(stderr, "[%d] PnetCDF error at %s:%d: %s\n",                 \
                    rank, __FILE__, __LINE__, ncmpi_strerror(_status));          \
            MPI_Abort(MPI_COMM_WORLD, _status);                                  \
        }                                                                        \
    } while (0)

static int rank, nprocs;

typedef struct VarInfo {
    char name[NC_MAX_NAME + 1];
    nc_type xtype;
    int ndims;
    int dimids[MAX_DIMS];
} VarInfo;

static int get_chunk_owner_block(MPI_Offset chunk_id, int nprocs, MPI_Offset total_chunks)
{
    if (nprocs <= 0) return 0;

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

static int nc_type_to_mpi(nc_type xtype, MPI_Datatype *dtype, int *esize)
{
    switch (xtype) {
        case NC_CHAR:
            *dtype = MPI_CHAR;
            *esize = 1;
            return 0;
        case NC_BYTE:
            *dtype = MPI_SIGNED_CHAR;
            *esize = 1;
            return 0;
        case NC_UBYTE:
            *dtype = MPI_UNSIGNED_CHAR;
            *esize = 1;
            return 0;
        case NC_SHORT:
            *dtype = MPI_SHORT;
            *esize = (int)sizeof(short);
            return 0;
        case NC_USHORT:
            *dtype = MPI_UNSIGNED_SHORT;
            *esize = (int)sizeof(unsigned short);
            return 0;
        case NC_INT:
            *dtype = MPI_INT;
            *esize = (int)sizeof(int);
            return 0;
        case NC_UINT:
            *dtype = MPI_UNSIGNED;
            *esize = (int)sizeof(unsigned int);
            return 0;
        case NC_INT64:
            *dtype = MPI_LONG_LONG;
            *esize = (int)sizeof(long long);
            return 0;
        case NC_UINT64:
            *dtype = MPI_UNSIGNED_LONG_LONG;
            *esize = (int)sizeof(unsigned long long);
            return 0;
        case NC_FLOAT:
            *dtype = MPI_FLOAT;
            *esize = (int)sizeof(float);
            return 0;
        case NC_DOUBLE:
            *dtype = MPI_DOUBLE;
            *esize = (int)sizeof(double);
            return 0;
        default:
            return -1;
    }
}

int main(int argc, char **argv)
{
    int mpi_err = MPI_Init(&argc, &argv);
    if (mpi_err != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init failed with error %d\n", mpi_err);
        return 1;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc < 3) {
        if (rank == 0) {
            fprintf(stderr, "Usage: %s INPUT.nc OUTPUT.nc\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }

    const char *input_path = argv[1];
    const char *output_path = argv[2];

    int ncid_in;
    CHECK_ERR(ncmpi_open(MPI_COMM_WORLD, input_path, NC_NOWRITE, MPI_INFO_NULL, &ncid_in));

    int format = NC_FORMAT_CLASSIC;
    CHECK_ERR(ncmpi_inq_format(ncid_in, &format));

    int cmode = NC_CLOBBER;
    if (format == NC_FORMAT_64BIT_OFFSET) {
        cmode |= NC_64BIT_OFFSET;
    } else if (format == NC_FORMAT_64BIT_DATA
#ifdef NC_FORMAT_CDF5
               || format == NC_FORMAT_CDF5
#endif
    ) {
        cmode |= NC_64BIT_DATA;
    }

    int ndims = 0, nvars = 0, natts = 0, unlimdimid = -1;
    CHECK_ERR(ncmpi_inq(ncid_in, &ndims, &nvars, &natts, &unlimdimid));

    MPI_Offset *dim_lens = NULL;
    int *dimids_out = NULL;
    if (ndims > 0) {
        dim_lens = (MPI_Offset *)malloc((size_t)ndims * sizeof(MPI_Offset));
        dimids_out = (int *)malloc((size_t)ndims * sizeof(int));
        if (dim_lens == NULL || dimids_out == NULL) {
            fprintf(stderr, "[%d] Failed to allocate dim arrays\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    int ncid_out;
    CHECK_ERR(ncmpi_create(MPI_COMM_WORLD, output_path, cmode,
                           MPI_INFO_NULL, &ncid_out));

    int time_dimid = -1;
    MPI_Offset time_len = 0;
    for (int dimid = 0; dimid < ndims; dimid++) {
        char name[NC_MAX_NAME + 1];
        MPI_Offset len = 0;
        CHECK_ERR(ncmpi_inq_dim(ncid_in, dimid, name, &len));
        dim_lens[dimid] = len;
        if (strcmp(name, "time") == 0) {
            time_dimid = dimid;
            time_len = len;
        }
        if (dimid == unlimdimid) {
            CHECK_ERR(ncmpi_def_dim(ncid_out, name, NC_UNLIMITED, &dimids_out[dimid]));
        } else {
            CHECK_ERR(ncmpi_def_dim(ncid_out, name, len, &dimids_out[dimid]));
        }
    }

    VarInfo *vars = NULL;
    if (nvars > 0) {
        vars = (VarInfo *)calloc((size_t)nvars, sizeof(VarInfo));
        if (vars == NULL) {
            fprintf(stderr, "[%d] Failed to allocate var info\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    int *varids_out = NULL;
    if (nvars > 0) {
        varids_out = (int *)malloc((size_t)nvars * sizeof(int));
        if (varids_out == NULL) {
            fprintf(stderr, "[%d] Failed to allocate varid map\n", rank);
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }
    for (int varid = 0; varid < nvars; varid++) {
        VarInfo *v = &vars[varid];
        CHECK_ERR(ncmpi_inq_var(ncid_in, varid, v->name, &v->xtype, &v->ndims,
                                v->dimids, NULL));
        v->name[NC_MAX_NAME] = '\0';
        int out_dimids[MAX_DIMS];
        for (int d = 0; d < v->ndims; d++) {
            out_dimids[d] = dimids_out[v->dimids[d]];
        }
        CHECK_ERR(ncmpi_def_var(ncid_out, v->name, v->xtype, v->ndims,
                                out_dimids, &varids_out[varid]));
    }

    CHECK_ERR(ncmpi_enddef(ncid_out));

    if (time_dimid < 0) {
        time_dimid = unlimdimid;
        if (time_dimid >= 0 && dim_lens != NULL) {
            time_len = dim_lens[time_dimid];
        }
    }

    if (rank == 0) {
        printf("========================================================\n");
        printf("copy_timechunks_putvarn_all\n");
        printf("========================================================\n");
        printf("Input:  %s\n", input_path);
        printf("Output: %s\n", output_path);
        printf("Processes: %d\n", nprocs);
        printf("Dimensions: %d,  Variables: %d\n", ndims, nvars);
        printf("Time dimension: id=%d, len=%lld\n", time_dimid, (long long)time_len);
        printf("========================================================\n");
        fflush(stdout);
    }

    for (int varid = 0; varid < nvars; varid++) {
        VarInfo *v = &vars[varid];
        const char *name = v->name;
        nc_type xtype = v->xtype;
        int var_ndims = v->ndims;
        int *var_dimids = v->dimids;

        MPI_Datatype dtype = MPI_BYTE;
        int esize = 0;
        if (nc_type_to_mpi(xtype, &dtype, &esize) != 0) {
            if (rank == 0) {
                fprintf(stderr, "Unsupported type for variable %s\n", name);
            }
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        int is_time_var = (time_dimid >= 0 && var_ndims >= 1 && var_dimids[0] == time_dimid);
        int is_flds = (strcmp(name, "FLDS") == 0);

        if (is_flds && is_time_var) {
            /*
             * FLDS: chunk by time -- each rank owns a block of time steps.
             * Read with ncmpi_get_varn_all, write with ncmpi_put_varn_all.
             */
            MPI_Offset total_chunks = time_len;
            MPI_Offset local_chunks = total_chunks / nprocs;
            MPI_Offset remainder = total_chunks % nprocs;
            if ((MPI_Offset)rank < remainder) local_chunks++;
            if (local_chunks > INT_MAX) {
                fprintf(stderr, "[%d] Too many chunks for int num\n", rank);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }

            MPI_Offset chunk_elems_off = 1;
            for (int d = 1; d < var_ndims; d++) {
                if (dim_lens[var_dimids[d]] != 0 &&
                    chunk_elems_off > (MPI_Offset)(LLONG_MAX / dim_lens[var_dimids[d]])) {
                    fprintf(stderr, "[%d] Chunk element count overflow for %s\n", rank, name);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
                chunk_elems_off *= dim_lens[var_dimids[d]];
            }
            size_t chunk_elems = (size_t)chunk_elems_off;

            if (rank == 0) {
                printf("FLDS: %d dims, %lld time steps, %zu elems/step, %lld chunks/rank (approx)\n",
                       var_ndims, (long long)time_len, chunk_elems, (long long)local_chunks);
                printf("FLDS buffer per rank: %.2f MB\n",
                       (double)local_chunks * (double)chunk_elems * (double)esize / (1024.0 * 1024.0));
                fflush(stdout);
            }

            MPI_Offset *starts_store = NULL;
            MPI_Offset *counts_store = NULL;
            MPI_Offset **starts = NULL;
            MPI_Offset **counts = NULL;
            void *buf = NULL;

            if (local_chunks > 0) {
                starts_store = (MPI_Offset *)malloc((size_t)local_chunks * (size_t)var_ndims *
                                                    sizeof(MPI_Offset));
                counts_store = (MPI_Offset *)malloc((size_t)local_chunks * (size_t)var_ndims *
                                                    sizeof(MPI_Offset));
                starts = (MPI_Offset **)malloc((size_t)local_chunks * sizeof(MPI_Offset *));
                counts = (MPI_Offset **)malloc((size_t)local_chunks * sizeof(MPI_Offset *));
                if (starts_store == NULL || counts_store == NULL ||
                    starts == NULL || counts == NULL) {
                    fprintf(stderr, "[%d] Failed to allocate starts/counts\n", rank);
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
                for (MPI_Offset i = 0; i < local_chunks; i++) {
                    starts[i] = starts_store + i * var_ndims;
                    counts[i] = counts_store + i * var_ndims;
                }

                if (chunk_elems > 0) {
                    if ((unsigned long long)local_chunks > SIZE_MAX / chunk_elems ||
                        (unsigned long long)(local_chunks * chunk_elems) > SIZE_MAX / (size_t)esize) {
                        fprintf(stderr, "[%d] Buffer size overflow for %s\n", rank, name);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                    buf = malloc((size_t)local_chunks * chunk_elems * (size_t)esize);
                    if (buf == NULL) {
                        fprintf(stderr, "[%d] Failed to allocate buffer for %s (%zu bytes)\n",
                                rank, name, (size_t)local_chunks * chunk_elems * (size_t)esize);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                }
            }

            MPI_Offset idx = 0;
            for (MPI_Offset t = 0; t < time_len; t++) {
                int owner = get_chunk_owner_block(t, nprocs, total_chunks);
                if (owner == rank) {
                    MPI_Offset read_start[MAX_DIMS];
                    MPI_Offset read_count[MAX_DIMS];
                    read_start[0] = t;
                    read_count[0] = 1;
                    for (int d = 1; d < var_ndims; d++) {
                        read_start[d] = 0;
                        read_count[d] = dim_lens[var_dimids[d]];
                    }
                    memcpy(starts[idx], read_start, (size_t)var_ndims * sizeof(MPI_Offset));
                    memcpy(counts[idx], read_count, (size_t)var_ndims * sizeof(MPI_Offset));
                    idx++;
                }
            }

            void *write_buf = buf;
            char dummy = 0;
            MPI_Offset bufcount = (MPI_Offset)(local_chunks * (MPI_Offset)chunk_elems);
            if (local_chunks == 0 || chunk_elems == 0) {
                write_buf = &dummy;
                bufcount = 0;
            }

            MPI_Offset dummy_start[MAX_DIMS] = {0};
            MPI_Offset dummy_count[MAX_DIMS] = {0};
            MPI_Offset *dummy_starts[1] = {dummy_start};
            MPI_Offset *dummy_counts[1] = {dummy_count};
            MPI_Offset **use_starts = (local_chunks > 0) ? starts : dummy_starts;
            MPI_Offset **use_counts = (local_chunks > 0) ? counts : dummy_counts;

            CHECK_ERR(ncmpi_get_varn_all(ncid_in, varid, (int)local_chunks,
                                         use_starts, use_counts, write_buf, bufcount, dtype));

            MPI_Barrier(MPI_COMM_WORLD);
            double t_write_start = MPI_Wtime();
            CHECK_ERR(ncmpi_put_varn_all(ncid_out, varids_out[varid], (int)local_chunks,
                                         use_starts, use_counts, write_buf, bufcount, dtype));
            MPI_Barrier(MPI_COMM_WORLD);
            double t_write = MPI_Wtime() - t_write_start;

            if (rank == 0) {
                double flds_bytes = (double)time_len * (double)chunk_elems * (double)esize;
                printf("FLDS write time: %.3f sec  (%.2f GB, %.2f GB/s)\n",
                       t_write, flds_bytes / (1024.0*1024.0*1024.0),
                       flds_bytes / t_write / (1024.0*1024.0*1024.0));
                fflush(stdout);
            }

            if (buf != NULL) free(buf);
            if (starts != NULL) free(starts);
            if (counts != NULL) free(counts);
            if (starts_store != NULL) free(starts_store);
            if (counts_store != NULL) free(counts_store);
        } else {
            /*
             * Non-FLDS variables: rank 0 reads all, collective write.
             */
            MPI_Offset total_elems_off = 1;
            if (var_ndims == 0) {
                total_elems_off = 1;
            } else {
                for (int d = 0; d < var_ndims; d++) {
                    if (dim_lens[var_dimids[d]] != 0 &&
                        total_elems_off > (MPI_Offset)(LLONG_MAX / dim_lens[var_dimids[d]])) {
                        fprintf(stderr, "[%d] Element count overflow for %s\n", rank, name);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                    total_elems_off *= dim_lens[var_dimids[d]];
                }
            }
            size_t total_elems = (size_t)total_elems_off;

            if (var_ndims == 0) {
                /* Scalar variable */
                void *buf = NULL;
                if (total_elems > 0) {
                    if ((unsigned long long)total_elems > SIZE_MAX / (size_t)esize) {
                        fprintf(stderr, "[%d] Buffer size overflow for %s\n", rank, name);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                    buf = malloc(total_elems * (size_t)esize);
                    if (buf == NULL) {
                        fprintf(stderr, "[%d] Failed to allocate buffer for %s\n", rank, name);
                        MPI_Abort(MPI_COMM_WORLD, -1);
                    }
                }
                MPI_Offset bufcount = (MPI_Offset)total_elems;
                char dummy = 0;
                void *rw_buf = (buf != NULL) ? buf : &dummy;

                CHECK_ERR(ncmpi_get_var_all(ncid_in, varid, rw_buf, bufcount, dtype));
                CHECK_ERR(ncmpi_put_var_all(ncid_out, varids_out[varid],
                                            rw_buf, bufcount, dtype));

                if (buf != NULL) free(buf);
            } else {
                /* Multi-dimensional non-FLDS variable */
                void *buf = NULL;
                if (rank == 0) {
                    if (total_elems > 0) {
                        if ((unsigned long long)total_elems > SIZE_MAX / (size_t)esize) {
                            fprintf(stderr, "[%d] Buffer size overflow for %s\n", rank, name);
                            MPI_Abort(MPI_COMM_WORLD, -1);
                        }
                        buf = malloc(total_elems * (size_t)esize);
                        if (buf == NULL) {
                            fprintf(stderr, "[%d] Failed to allocate buffer for %s\n", rank, name);
                            MPI_Abort(MPI_COMM_WORLD, -1);
                        }
                    }
                }

                MPI_Offset start0[MAX_DIMS];
                MPI_Offset count0[MAX_DIMS];
                for (int d = 0; d < var_ndims; d++) {
                    start0[d] = 0;
                    count0[d] = dim_lens[var_dimids[d]];
                }
                MPI_Offset *startp[1] = {start0};
                MPI_Offset *countp[1] = {count0};
                MPI_Offset dummy_start[MAX_DIMS] = {0};
                MPI_Offset dummy_count[MAX_DIMS] = {0};
                MPI_Offset *dummy_starts[1] = {dummy_start};
                MPI_Offset *dummy_counts[1] = {dummy_count};

                int num = (rank == 0) ? 1 : 0;
                MPI_Offset **use_starts = (rank == 0) ? startp : dummy_starts;
                MPI_Offset **use_counts = (rank == 0) ? countp : dummy_counts;
                MPI_Offset bufcount = (rank == 0) ? (MPI_Offset)total_elems : 0;
                char dummy = 0;
                void *rw_buf = (rank == 0 && buf != NULL) ? buf : &dummy;

                CHECK_ERR(ncmpi_get_varn_all(ncid_in, varid, num,
                                             use_starts, use_counts,
                                             rw_buf, bufcount, dtype));
                CHECK_ERR(ncmpi_put_varn_all(ncid_out, varids_out[varid], num,
                                             use_starts, use_counts,
                                             rw_buf, bufcount, dtype));

                if (buf != NULL) free(buf);
            }
        }
    }

    CHECK_ERR(ncmpi_close(ncid_out));
    CHECK_ERR(ncmpi_close(ncid_in));

    if (dim_lens != NULL) free(dim_lens);
    if (dimids_out != NULL) free(dimids_out);
    if (varids_out != NULL) free(varids_out);
    if (vars != NULL) free(vars);

    MPI_Finalize();
    return 0;
}
