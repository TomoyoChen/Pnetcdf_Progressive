/* Simplified debug program to test data write operations */

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <pnetcdf.h>

#define NY 10
#define NX 10

int main(int argc, char** argv) {
    int rank, nprocs, err = 0;
    int ncid, varid, dimid[3];
    MPI_Offset start[3] = {0, 0, 0};
    MPI_Offset count[3] = {1, NY, NX};
    MPI_Info info;
    int *buf;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    /* Allocate and initialize data */
    buf = malloc(sizeof(int) * NY * NX);
    for (int i = 0; i < NY * NX; i++) {
        buf[i] = 1000 + rank * 100 + i;
    }
    
    if (rank == 0) {
        printf("=== Debug Write Test ===\n");
        printf("Writing data: ");
        for (int i = 0; i < 10; i++) printf("%d ", buf[i]);
        printf("...\n");
    }
    
    /* Create file with chunking */
    MPI_Info_create(&info);
    MPI_Info_set(info, "nc_chunking", "enable");
    
    err = ncmpi_create(MPI_COMM_WORLD, "debug_test.nc", NC_CLOBBER, info, &ncid);
    if (err != NC_NOERR) {
        printf("Error creating file: %s\n", ncmpi_strerror(err));
        goto cleanup;
    }
    
    /* Define dimensions */
    err = ncmpi_def_dim(ncid, "time", NC_UNLIMITED, &dimid[0]);
    if (err != NC_NOERR) printf("Error def time dim: %s\n", ncmpi_strerror(err));
    
    err = ncmpi_def_dim(ncid, "Y", NY, &dimid[1]);
    if (err != NC_NOERR) printf("Error def Y dim: %s\n", ncmpi_strerror(err));
    
    err = ncmpi_def_dim(ncid, "X", NX, &dimid[2]);
    if (err != NC_NOERR) printf("Error def X dim: %s\n", ncmpi_strerror(err));
    
    /* Define variable with NO compression first */
    err = ncmpi_def_var(ncid, "test_var", NC_INT, 3, dimid, &varid);
    if (err != NC_NOERR) {
        printf("Error defining variable: %s\n", ncmpi_strerror(err));
        goto cleanup;
    }
    
    /* Set simple chunking */
    int chunk_dim[3] = {1, 5, 5};
    err = ncmpi_var_set_chunk(ncid, varid, chunk_dim);
    if (err != NC_NOERR) {
        printf("Error setting chunk: %s\n", ncmpi_strerror(err));
        goto cleanup;
    }
    
    if (rank == 0) printf("Step 1: No compression test\n");
    
    /* Exit define mode */
    err = ncmpi_enddef(ncid);
    if (err != NC_NOERR) {
        printf("Error ending def mode: %s\n", ncmpi_strerror(err));
        goto cleanup;
    }
    
    /* Write data */
    err = ncmpi_put_vara_int_all(ncid, varid, start, count, buf);
    if (err != NC_NOERR) {
        printf("Error writing data: %s\n", ncmpi_strerror(err));
        goto cleanup;
    }
    
    if (rank == 0) printf("✓ Data write completed\n");
    
    /* Check dimension length */
    MPI_Offset dim_len;
    err = ncmpi_inq_dimlen(ncid, dimid[0], &dim_len);
    if (err == NC_NOERR && rank == 0) {
        printf("Time dimension length: %lld (should be %d)\n", dim_len, nprocs);
    }
    
    ncmpi_close(ncid);
    
    /* Test reading back */
    if (rank == 0) printf("\nStep 2: Reading back data\n");
    
    err = ncmpi_open(MPI_COMM_WORLD, "debug_test.nc", NC_NOWRITE, info, &ncid);
    if (err == NC_NOERR) {
        int *read_buf = malloc(sizeof(int) * NY * NX);
        err = ncmpi_inq_varid(ncid, "test_var", &varid);
        if (err == NC_NOERR) {
            err = ncmpi_get_vara_int_all(ncid, varid, start, count, read_buf);
            if (err == NC_NOERR && rank == 0) {
                printf("Read back data: ");
                for (int i = 0; i < 10; i++) printf("%d ", read_buf[i]);
                printf("...\n");
                
                /* Verify data */
                int errors = 0;
                for (int i = 0; i < NY * NX; i++) {
                    if (read_buf[i] != buf[i]) errors++;
                }
                printf("Verification: %d errors out of %d values\n", errors, NY * NX);
                
                if (errors == 0) {
                    printf("✓ Basic write/read test PASSED\n");
                } else {
                    printf("✗ Basic write/read test FAILED\n");
                }
            }
        }
        free(read_buf);
        ncmpi_close(ncid);
    }
    
cleanup:
    free(buf);
    MPI_Info_free(&info);
    MPI_Finalize();
    
    if (rank == 0) {
        printf("\n=== Next Steps ===\n");
        printf("If this test passes, the issue is with compression.\n");
        printf("If this test fails, there's a fundamental write problem.\n");
        printf("Run: ncdump -h debug_test.nc to check file structure\n");
        printf("Run: ncdump -v test_var debug_test.nc to check data\n");
    }
    
    return (err != NC_NOERR);
} 