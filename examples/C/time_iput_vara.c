#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NT 248
#define NY 8075
#define NX 7814

#define ERR { \
    if (err != NC_NOERR) { \
        printf("Error at %s:%d : %s\n", __FILE__,__LINE__, \
               ncmpi_strerror(err)); \
        goto err_out; \
    } \
}

int main(int argc, char** argv)
{
    int i, j, k, rank, nprocs, err, ncid, varid, dimid[3], ntimes;
    float *buf = NULL, *buf_ptr;
    double timing, max_t;
    MPI_Offset start[3], count[3];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    if (argc == 1) {
        if (rank == 0) printf("Usage: %s filename\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    /* create a new file */
    err = ncmpi_create(MPI_COMM_WORLD, argv[1], NC_CLOBBER|NC_64BIT_DATA,
                       MPI_INFO_NULL, &ncid); ERR

    /* define dimensions */
    err = ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &dimid[0]); ERR
    err = ncmpi_def_dim(ncid, "y",    NY,           &dimid[1]); ERR
    err = ncmpi_def_dim(ncid, "x",    NX,           &dimid[2]); ERR

    /* define variable */
    err = ncmpi_def_var(ncid, "var", NC_FLOAT, 3, dimid, &varid); ERR

    /* exit define mode and enter the data mode */
    err = ncmpi_enddef(ncid); ERR

    /* partition along time dimension */
    ntimes = (NT / nprocs);

    /* allocate and initialize buffer */
    buf = (float*) malloc(sizeof(float) * ntimes*NY*NX);
    for (i=0; i<ntimes; i++)
        for (j=0; j<NY; j++)
            for (k=0; k<NX; k++)
                buf[i*NY*NX + j*NX + k] = rank+i+j+k;

    start[0] = rank * ntimes; /* block partitioning */
    /* cyclic partitioning should use start[0] = ranks; */
    start[1] = 0;
    start[2] = 0;

    count[0] = 1;
    count[1] = NY;
    count[2] = NX;

    MPI_Barrier(MPI_COMM_WORLD);
    timing = MPI_Wtime();

    buf_ptr = buf;
    for (i=0; i<ntimes; i++) {
        /* use PnetCDF nonblocking put API */
        err = ncmpi_iput_vara_float(ncid, varid, start, count, buf_ptr, NULL);
        ERR

        buf_ptr += count[1]*count[2];

        start[0]++; /* block partitioning */
        /* cyclic partitioning should use start[0] += nprocs; */
    }

    /* collectively wait for writes to complete */
    err = ncmpi_wait_all(ncid, NC_REQ_ALL, NULL, NULL); ERR
    timing = MPI_Wtime() - timing;

    /* close the file */
    err = ncmpi_close(ncid); ERR

    MPI_Reduce(&timing, &max_t, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
        printf("Max timing: %.2f seconds\n", max_t);

err_out:
    if (buf != NULL) free(buf);

    MPI_Finalize();

    return (err != NC_NOERR);
}

