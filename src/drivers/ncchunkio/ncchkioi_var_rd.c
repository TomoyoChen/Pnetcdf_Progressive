/*
 *  Copyright (C) 2019, Northwestern University and Argonne National Laboratory
 *  See COPYRIGHT notice in top-level directory.
 */
/* $Id$ */

/*
 * This file implements the following PnetCDF APIs.
 *
 * ncmpi_get_var<kind>_all()        : dispatcher->get_var()
 * ncmpi_put_var<kind>_all()        : dispatcher->put_var()
 * ncmpi_get_var<kind>_<type>_all() : dispatcher->get_var()
 * ncmpi_put_var<kind>_<type>_all() : dispatcher->put_var()
 */

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <common.h>
#include <math.h>
#include <mpi.h>
#include <ncchkio_driver.h>
#include <pnc_debug.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../ncmpio/ncmpio_NC.h"
#include "ncchkio_internal.h"
#include "ncchk_filter_driver.h"

int ncchkioi_load_var (NC_chk *ncchkp, NC_chk_var *varp, int nchunk, int *cids) {
	int err=NC_NOERR;
	int i;
	int cid;
	int get_size;

	int dsize;
	MPI_Offset bsize;

	int *lens;
	MPI_Aint *fdisps, *mdisps;
	MPI_Status status;
	MPI_Datatype ftype, mtype;	// Memory and file datatype

	char **zbufs;

	NC *ncp = (NC *)(ncchkp->ncp);

	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO)
	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_INIT)

	// -1 means all chunks
	if (nchunk < 0) {
		nchunk = varp->nmychunk;
		cids   = varp->mychunks;
	}

	// Allocate buffer for I/O
	lens   = (int *)NCI_Malloc (sizeof (int) * nchunk);
	fdisps = (MPI_Aint *)NCI_Malloc (sizeof (MPI_Aint) * nchunk * 2);
	mdisps = fdisps + nchunk;
	zbufs  = (char **)NCI_Malloc (sizeof (char *) * nchunk);

	/* Carry our coll I/O
	 * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when
	 * count is 0 We use a dummy call inplace of type with 0 count
	 */
	if (nchunk > 0) {
		// Create file type
		bsize = 0;
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			// offset and length of compressed chunks
			lens[i]	  = varp->chunk_index[cid].len;
			fdisps[i] = (MPI_Aint) (varp->chunk_index[cid].off) + ncp->begin_var;
			mdisps[i] = bsize;
			// At the same time, we record the size of buffer we need
			bsize += (MPI_Offset)lens[i];
		}

		// Allocate buffer for compressed data
		zbufs[0] = (char *)NCI_Malloc (bsize);
		for (i = 1; i < nchunk; i++) {
			zbufs[i] = zbufs[i - 1] + varp->chunk_index[cids[i - 1]].len;
		}

		ncchkioi_sort_file_offset (nchunk, fdisps, mdisps, lens);

		MPI_Type_create_hindexed (nchunk, lens, fdisps, MPI_BYTE, &ftype);
		CHK_ERR_TYPE_COMMIT (&ftype);

		MPI_Type_create_hindexed (nchunk, lens, mdisps, MPI_BYTE, &mtype);
		CHK_ERR_TYPE_COMMIT (&mtype);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_INIT, NC_CHK_TIMER_GET_IO_RD)

		// Perform MPI-IO
		// Set file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
		// Write data
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, zbufs[0], 1, mtype, &status);
		// Restore file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_RD, NC_CHK_TIMER_GET_IO_INIT)

#ifdef _USE_MPI_GET_COUNT
		MPI_Get_count (&status, MPI_BYTE, &get_size);
#else
		MPI_Type_size (ftype, &get_size);
#endif
		ncchkp->getsize += get_size;

		// Free type
		MPI_Type_free (&ftype);
		MPI_Type_free (&mtype);
	} else {
		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_INIT, NC_CHK_TIMER_GET_IO_RD)

		// Follow coll I/O with dummy call
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_RD, NC_CHK_TIMER_GET_IO_INIT)
	}

	// Decompress each chunk
	// Allocate chunk cache if not allocated
	if (varp->filter_driver != NULL) {
		varp->filter_driver->init (MPI_INFO_NULL);
		dsize = varp->chunksize;
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
				// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
			}

					// Prepare variable context for decompression
		NCCHK_var_context ctx = (NCCHK_var_context){0};
		ctx.sz_abs_err_bound = varp->sz_abs_err_bound;
		ctx.sz_rel_bound_ratio = varp->sz_rel_bound_ratio;
		ctx.zlib_level = varp->zlib_level;
		ctx.varid = varp->varid;
#ifdef ENABLE_IPCOMP
	ctx.ipcomp_put_att = ncchkp->driver->put_att;
	ctx.ipcomp_get_att = ncchkp->driver->get_att;
	ctx.ipcomp_ncp     = ncchkp->ncp;
	ctx.ipcomp_varid   = varp->varid;
#endif
#ifdef ENABLE_IPCOMP
		ctx.ipcomp_layers = varp->ipcomp_layers;
		ctx.ipcomp_interp = varp->ipcomp_interp;
		ctx.ipcomp_direction = varp->ipcomp_direction;
		ctx.ipcomp_level_progressive = varp->ipcomp_level_progressive;
		ctx.ipcomp_block_size = varp->ipcomp_block_size;
		ctx.ipcomp_interp_dim_limit = varp->ipcomp_interp_dim_limit;
		ctx.ipcomp_ebs = varp->ipcomp_ebs;
		ctx.ipcomp_num_ebs = varp->ipcomp_num_ebs;
		ctx.ipcomp_data_range = varp->ipcomp_data_range;
		ctx.ipcomp_data_min = varp->ipcomp_data_min;
		ctx.ipcomp_data_max = varp->ipcomp_data_max;
		ctx.ipcomp_has_minmax = varp->ipcomp_has_minmax;
		ctx.ipcomp_has_fill = varp->ipcomp_has_fill;
		ctx.ipcomp_fill_value = varp->ipcomp_fill_value;
		ctx.ipcomp_header_size = varp->ipcomp_header_size;
#endif

		NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_DECOM)
		
#ifdef ENABLE_IPCOMP
		/* Use progressive decompression if target error bound is set */
		if (varp->filter == NC_CHK_FILTER_IPCOMP && 
		    varp->ipcomp_use_progressive_decomp && 
		    varp->ipcomp_target_error_bound > 0.0) {
			ncchk_ipcomp_decompress_progressive_error(
				zbufs[i], lens[i], varp->chunk_cache[cid]->buf, &dsize,
				varp->ndim, varp->chunkdim, varp->etype, 
				varp->ipcomp_target_error_bound, &ctx);
		} else {
			varp->filter_driver->decompress (zbufs[i], lens[i], varp->chunk_cache[cid]->buf, &dsize,
								   varp->ndim, varp->chunkdim, varp->etype, &ctx);
		}
#else
		varp->filter_driver->decompress (zbufs[i], lens[i], varp->chunk_cache[cid]->buf, &dsize,
							   varp->ndim, varp->chunkdim, varp->etype, &ctx);
#endif
		NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_GET_IO_DECOM, NC_CHK_TIMER_GET_IO_INIT)

			if (dsize != varp->chunksize) { printf ("Decompress Error\n"); }
		}
		varp->filter_driver->finalize ();
	} else {
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
				// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

				NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_DECOM)
				memcpy (varp->chunk_cache[cid]->buf, zbufs[i], lens[i]);
				NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_GET_IO_DECOM, NC_CHK_TIMER_GET_IO_INIT)
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
			}
		}
	}

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO_INIT)

	// Free buffers
	if (nchunk > 0) { NCI_Free (zbufs[0]); }
	NCI_Free (zbufs);

	NCI_Free (lens);
	NCI_Free (fdisps);

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO)

err_out:;
	return err;
}

int ncchkioi_load_var_indep (NC_chk *ncchkp, NC_chk_var *varp, int nchunk, int *cids) {
	int err=NC_NOERR;
	int i;
	int cid;
	int dsize;
	MPI_Offset bsize;
	int *lens = NULL;
	char **zbufs = NULL;
	MPI_Status status;
	NC *ncp = (NC *)(ncchkp->ncp);

	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO)
	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_INIT)

	if (nchunk <= 0 || cids == NULL) {
		NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO_INIT)
		NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO)
		return NC_NOERR;
	}

	// Allocate buffer for I/O
	lens  = (int *)NCI_Malloc (sizeof (int) * nchunk);
	CHK_PTR (lens)
	zbufs = (char **)NCI_Malloc (sizeof (char *) * nchunk);
	CHK_PTR (zbufs)

	// Compute sizes and allocate a contiguous buffer
	bsize = 0;
	for (i = 0; i < nchunk; i++) {
		cid = cids[i];
		lens[i] = varp->chunk_index[cid].len;
		bsize += (MPI_Offset)lens[i];
	}
	if (bsize > 0) {
		zbufs[0] = (char *)NCI_Malloc (bsize);
		CHK_PTR (zbufs[0])
		for (i = 1; i < nchunk; i++) {
			zbufs[i] = zbufs[i - 1] + lens[i - 1];
		}
	} else {
		zbufs[0] = NULL;
		for (i = 1; i < nchunk; i++) {
			zbufs[i] = NULL;
		}
	}

	NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_INIT, NC_CHK_TIMER_GET_IO_RD)

	// Independent read for each chunk
	for (i = 0; i < nchunk; i++) {
		cid = cids[i];
		if (lens[i] <= 0) continue;
		err = MPI_File_read_at (ncp->independent_fh,
								(MPI_Offset)(varp->chunk_index[cid].off) + ncp->begin_var,
								zbufs[i], lens[i], MPI_BYTE, &status);
		CHK_MPIERR
	}
	ncchkp->getsize += bsize;

	NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_RD, NC_CHK_TIMER_GET_IO_INIT)

	// Decompress each chunk
	if (varp->filter_driver != NULL) {
		varp->filter_driver->init (MPI_INFO_NULL);
		dsize = varp->chunksize;
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
			}

			if (lens[i] <= 0) {
				memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
				continue;
			}

			// Prepare variable context for decompression
			NCCHK_var_context ctx = (NCCHK_var_context){0};
			ctx.sz_abs_err_bound = varp->sz_abs_err_bound;
			ctx.sz_rel_bound_ratio = varp->sz_rel_bound_ratio;
			ctx.zlib_level = varp->zlib_level;
			ctx.varid = varp->varid;
#ifdef ENABLE_IPCOMP
			ctx.ipcomp_put_att = ncchkp->driver->put_att;
			ctx.ipcomp_get_att = ncchkp->driver->get_att;
			ctx.ipcomp_ncp     = ncchkp->ncp;
			ctx.ipcomp_varid   = varp->varid;
			ctx.ipcomp_layers = varp->ipcomp_layers;
			ctx.ipcomp_interp = varp->ipcomp_interp;
			ctx.ipcomp_direction = varp->ipcomp_direction;
			ctx.ipcomp_level_progressive = varp->ipcomp_level_progressive;
			ctx.ipcomp_block_size = varp->ipcomp_block_size;
			ctx.ipcomp_interp_dim_limit = varp->ipcomp_interp_dim_limit;
			ctx.ipcomp_ebs = varp->ipcomp_ebs;
			ctx.ipcomp_num_ebs = varp->ipcomp_num_ebs;
			ctx.ipcomp_data_range = varp->ipcomp_data_range;
			ctx.ipcomp_data_min = varp->ipcomp_data_min;
			ctx.ipcomp_data_max = varp->ipcomp_data_max;
			ctx.ipcomp_has_minmax = varp->ipcomp_has_minmax;
			ctx.ipcomp_has_fill = varp->ipcomp_has_fill;
			ctx.ipcomp_fill_value = varp->ipcomp_fill_value;
			ctx.ipcomp_header_size = varp->ipcomp_header_size;
#endif

			NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_DECOM)
#ifdef ENABLE_IPCOMP
			if (varp->filter == NC_CHK_FILTER_IPCOMP &&
				varp->ipcomp_use_progressive_decomp &&
				varp->ipcomp_target_error_bound > 0.0) {
				ncchk_ipcomp_decompress_progressive_error(
					zbufs[i], lens[i], varp->chunk_cache[cid]->buf, &dsize,
					varp->ndim, varp->chunkdim, varp->etype,
					varp->ipcomp_target_error_bound, &ctx);
			} else {
				varp->filter_driver->decompress (zbufs[i], lens[i], varp->chunk_cache[cid]->buf,
												 &dsize, varp->ndim, varp->chunkdim, varp->etype, &ctx);
			}
#else
			varp->filter_driver->decompress (zbufs[i], lens[i], varp->chunk_cache[cid]->buf,
											 &dsize, varp->ndim, varp->chunkdim, varp->etype, &ctx);
#endif
			NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_GET_IO_DECOM, NC_CHK_TIMER_GET_IO_INIT)
			if (dsize != varp->chunksize) { printf ("Decompress Error\n"); }
		}
		varp->filter_driver->finalize ();
	} else {
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
			}
			if (lens[i] <= 0) {
				memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
			} else {
				memcpy (varp->chunk_cache[cid]->buf, zbufs[i], lens[i]);
			}
		}
	}

	if (nchunk > 0 && zbufs != NULL && zbufs[0] != NULL) { NCI_Free (zbufs[0]); }
	NCI_Free (zbufs);
	NCI_Free (lens);

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO)

err_out:;
	return err;
}

int ncchkioi_load_nvar (NC_chk *ncchkp, int nvar, int *varids, int *lo, int *hi) {
	int err=NC_NOERR;
	int i, j, k;
	int cid;
	int get_size;

	int nchunk;

	int dsize;
	MPI_Offset bsize;

	int *lens;
	MPI_Aint *fdisps, *mdisps;
	MPI_Status status;
	MPI_Datatype ftype, mtype;	// Memory and file datatype

	char **zbufs;

	NC *ncp = (NC *)(ncchkp->ncp);
	NC_chk_var *varp;

	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO)
	NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_INIT)

	// -1 means all chunks
	nchunk = 0;
	for (i = 0; i < nvar; i++) {
		varp = ncchkp->vars.data + varids[i];

		for (j = lo[i]; j < hi[i]; j++) {
			cid = varp->mychunks[j];
			if (varp->chunk_cache[cid] == NULL && varp->chunk_index[cid].len > 0) { nchunk++; }
		}
	}

	// Allocate buffer for I/O
	lens   = (int *)NCI_Malloc (sizeof (int) * nchunk);
	fdisps = (MPI_Aint *)NCI_Malloc (sizeof (MPI_Aint) * nchunk * 2);
	mdisps = fdisps + nchunk;
	zbufs  = (char **)NCI_Malloc (sizeof (char *) * nchunk);

	/* Carry our coll I/O
	 * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when
	 * count is 0 We use a dummy call inplace of type with 0 count
	 */
	if (nchunk > 0) {
		// Create file type
		bsize = 0;
		k	  = 0;
		for (i = 0; i < nvar; i++) {
			varp = ncchkp->vars.data + varids[i];

			for (j = lo[i]; j < hi[i]; j++) {
				cid = varp->mychunks[j];

				// We only need to read when it is not in cache
				if (varp->chunk_cache[cid] == NULL && varp->chunk_index[cid].len > 0) {
					// offset and length of compressed chunks
					lens[k]	  = varp->chunk_index[cid].len;
					fdisps[k] = (MPI_Aint) (varp->chunk_index[cid].off + ncp->begin_var);
					mdisps[k] = bsize;
					// At the same time, we record the size of buffer we need
					bsize += (MPI_Offset)lens[k++];
				}
			}
		}

		// Allocate buffer for compressed data
		// We allocate it continuously so no mem type needed
		zbufs[0] = (char *)NCI_Malloc (bsize);
		for (j = 1; j < nchunk; j++) { zbufs[j] = zbufs[j - 1] + lens[j - 1]; }

		ncchkioi_sort_file_offset (k, fdisps, mdisps, lens);

		MPI_Type_create_hindexed (nchunk, lens, fdisps, MPI_BYTE, &ftype);
		CHK_ERR_TYPE_COMMIT (&ftype);

		MPI_Type_create_hindexed (nchunk, lens, mdisps, MPI_BYTE, &mtype);
		CHK_ERR_TYPE_COMMIT (&mtype);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_INIT, NC_CHK_TIMER_GET_IO_RD)

		// Perform MPI-IO
		// Set file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
		// Write data
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, zbufs[0], 1, mtype, &status);
		// Restore file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

#ifdef _USE_MPI_GET_COUNT
		MPI_Get_count (&status, MPI_BYTE, &get_size);
#else
		MPI_Type_size (ftype, &get_size);
#endif
		ncchkp->getsize += get_size;

		// Free type
		MPI_Type_free (&ftype);
		MPI_Type_free (&mtype);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_RD, NC_CHK_TIMER_GET_IO_CACHE)

		k = 0;
		for (i = 0; i < nvar; i++) {
			varp  = ncchkp->vars.data + varids[i];
			dsize = varp->chunksize;

			// Decompress each chunk
			if (varp->filter_driver != NULL) {
				varp->filter_driver->init (MPI_INFO_NULL);

				for (j = lo[i]; j < hi[i]; j++) {
					cid = varp->mychunks[j];

					// Allocate chunk cache if not allocated
					if (varp->chunk_cache[cid] == NULL) {
						err =
							ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
						CHK_ERR
						// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

						// Perform decompression
						if (varp->chunk_index[cid].len > 0) {
							// Prepare variable context for decompression
							NCCHK_var_context ctx = (NCCHK_var_context){0};
							ctx.sz_abs_err_bound = varp->sz_abs_err_bound;
							ctx.sz_rel_bound_ratio = varp->sz_rel_bound_ratio;
							ctx.zlib_level = varp->zlib_level;
							ctx.varid = varp->varid;
#ifdef ENABLE_IPCOMP
							ctx.ipcomp_layers = varp->ipcomp_layers;
							ctx.ipcomp_interp = varp->ipcomp_interp;
							ctx.ipcomp_direction = varp->ipcomp_direction;
							ctx.ipcomp_level_progressive = varp->ipcomp_level_progressive;
							ctx.ipcomp_block_size = varp->ipcomp_block_size;
							ctx.ipcomp_interp_dim_limit = varp->ipcomp_interp_dim_limit;
							ctx.ipcomp_ebs = varp->ipcomp_ebs;
							ctx.ipcomp_num_ebs = varp->ipcomp_num_ebs;
							ctx.ipcomp_data_range = varp->ipcomp_data_range;
							ctx.ipcomp_data_min = varp->ipcomp_data_min;
							ctx.ipcomp_data_max = varp->ipcomp_data_max;
							ctx.ipcomp_has_minmax = varp->ipcomp_has_minmax;
							ctx.ipcomp_has_fill = varp->ipcomp_has_fill;
							ctx.ipcomp_fill_value = varp->ipcomp_fill_value;
#endif

							NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_DECOM)
							varp->filter_driver->decompress (zbufs[k], lens[k], varp->chunk_cache[cid]->buf,
												   &dsize, varp->ndim, varp->chunkdim, varp->etype, &ctx);
							if (dsize != varp->chunksize) { printf ("Decompress Error\n"); }
							k++;
							NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_GET_IO_DECOM,
												 NC_CHK_TIMER_GET_IO_CACHE)
						} else {
							memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
						}
					} else {
						ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
					}
				}
				varp->filter_driver->finalize ();
			} else {
				for (j = lo[i]; j < hi[i]; j++) {
					cid = varp->mychunks[j];

					// Allocate chunk cache if not allocated
					if (varp->chunk_cache[cid] == NULL) {
						err =
							ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
						CHK_ERR
						// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

						if (varp->chunk_index[cid].len > 0) {
							NC_CHK_TIMER_START (NC_CHK_TIMER_GET_IO_DECOM)
							memcpy (varp->chunk_cache[cid]->buf, zbufs[k], lens[k]);
							k++;
							NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_GET_IO_DECOM,
												 NC_CHK_TIMER_GET_IO_CACHE)
						} else {
							memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
						}
					} else {
						ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
					}
				}
			}
		}

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO_CACHE)
	} else {
		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_INIT, NC_CHK_TIMER_GET_IO_CACHE)

		for (i = 0; i < nvar; i++) {
			varp = ncchkp->vars.data + varids[i];

			for (j = lo[i]; j < hi[i]; j++) {
				cid = varp->mychunks[j];

				// Allocate chunk cache if not allocated
				if (varp->chunk_cache[cid] == NULL) {
					err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
					CHK_ERR
					// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
					memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
				} else {
					ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
				}
			}
		}

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_GET_IO_CACHE, NC_CHK_TIMER_GET_IO_RD)

		// Follow coll I/O with dummy call
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO_RD)
	}

	// Free buffers
	if (nchunk > 0) { NCI_Free (zbufs[0]); }
	NCI_Free (zbufs);

	NCI_Free (lens);
	NCI_Free (fdisps);

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_GET_IO)

err_out:;
	return err;
}

int ncchkioi_load_var_bg (NC_chk *ncchkp, NC_chk_var *varp, int nchunk, int *cids) {
	int err=NC_NOERR;
	int i;
	int cid;
	int get_size;

	int dsize;
	MPI_Offset bsize;

	int *lens;
	MPI_Aint *fdisps, *mdisps;
	MPI_Status status;
	MPI_Datatype ftype, mtype;	// Memory and file datatype

	char **zbufs;

	NC *ncp = (NC *)(ncchkp->ncp);

	NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG)
	NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_INIT)

	// -1 means all chunks
	if (nchunk < 0) {
		nchunk = varp->nmychunk;
		cids   = varp->mychunks;
	}

	// Allocate buffer for I/O
	lens   = (int *)NCI_Malloc (sizeof (int) * nchunk);
	fdisps = (MPI_Aint *)NCI_Malloc (sizeof (MPI_Aint) * nchunk * 2);
	mdisps = fdisps + nchunk;
	zbufs  = (char **)NCI_Malloc (sizeof (char *) * nchunk);

	/* Carry our coll I/O
	 * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when
	 * count is 0 We use a dummy call inplace of type with 0 count
	 */
	if (nchunk > 0) {
		// Create file type
		bsize = 0;
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			// offset and length of compressed chunks
			lens[i]	  = varp->chunk_index[cid].len;
			fdisps[i] = (MPI_Aint) (varp->chunk_index[cid].off) + ncp->begin_var;
			mdisps[i] = bsize;
			// At the same time, we record the size of buffer we need
			bsize += (MPI_Offset)lens[i];
		}

		// Allocate buffer for compressed data
		zbufs[0] = (char *)NCI_Malloc (bsize);
		for (i = 1; i < nchunk; i++) {
			zbufs[i] = zbufs[i - 1] + varp->chunk_index[cids[i - 1]].len;
		}

		ncchkioi_sort_file_offset (nchunk, fdisps, mdisps, lens);

		MPI_Type_create_hindexed (nchunk, lens, fdisps, MPI_BYTE, &ftype);
		CHK_ERR_TYPE_COMMIT (&ftype);

		MPI_Type_create_hindexed (nchunk, lens, mdisps, MPI_BYTE, &mtype);
		CHK_ERR_TYPE_COMMIT (&mtype);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_INIT)
		NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_RD)

		// Perform MPI-IO
		// Set file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
		// Write data
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, zbufs[0], 1, mtype, &status);
		// Restore file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_RD)

#ifdef _USE_MPI_PUT_COUNT
		MPI_Get_count (&status, MPI_BYTE, &get_size);
#else
		MPI_Type_size (ftype, &get_size);
#endif
		ncchkp->getsize += get_size;

		// Free type
		MPI_Type_free (&ftype);
		MPI_Type_free (&mtype);
	} else {
		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_INIT)
		NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_RD)

		// Follow coll I/O with dummy call
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_RD)
	}

	NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_DECOM)

	// Decompress each chunk
	// Allocate chunk cache if not allocated
	if (varp->filter_driver != NULL) {
		varp->filter_driver->init (MPI_INFO_NULL);
		dsize = varp->chunksize;
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
				// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
					}

		// Prepare variable context for decompression
		NCCHK_var_context ctx = (NCCHK_var_context){0};
		ctx.sz_abs_err_bound = varp->sz_abs_err_bound;
		ctx.sz_rel_bound_ratio = varp->sz_rel_bound_ratio;
		ctx.zlib_level = varp->zlib_level;
		ctx.varid = varp->varid;
#ifdef ENABLE_IPCOMP
		ctx.ipcomp_layers = varp->ipcomp_layers;
		ctx.ipcomp_interp = varp->ipcomp_interp;
		ctx.ipcomp_direction = varp->ipcomp_direction;
		ctx.ipcomp_level_progressive = varp->ipcomp_level_progressive;
		ctx.ipcomp_block_size = varp->ipcomp_block_size;
		ctx.ipcomp_interp_dim_limit = varp->ipcomp_interp_dim_limit;
		ctx.ipcomp_ebs = varp->ipcomp_ebs;
		ctx.ipcomp_num_ebs = varp->ipcomp_num_ebs;
		ctx.ipcomp_data_range = varp->ipcomp_data_range;
		ctx.ipcomp_data_min = varp->ipcomp_data_min;
		ctx.ipcomp_data_max = varp->ipcomp_data_max;
		ctx.ipcomp_has_minmax = varp->ipcomp_has_minmax;
		ctx.ipcomp_has_fill = varp->ipcomp_has_fill;
		ctx.ipcomp_fill_value = varp->ipcomp_fill_value;
#endif

		varp->filter_driver->decompress (zbufs[i], lens[i], varp->chunk_cache[cid]->buf, &dsize,
							   varp->ndim, varp->chunkdim, varp->etype, &ctx);

		if (dsize != varp->chunksize) { printf ("Decompress Error\n"); }
		}
		varp->filter_driver->finalize ();
	} else {
		for (i = 0; i < nchunk; i++) {
			cid = cids[i];
			if (varp->chunk_cache[cid] == NULL) {
				err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
				CHK_ERR
				// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
			} else {
				ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
			}

			memcpy (varp->chunk_cache[cid]->buf, zbufs[i], lens[i]);
		}
	}

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_DECOM)

	// Free buffers
	if (nchunk > 0) { NCI_Free (zbufs[0]); }
	NCI_Free (zbufs);

	NCI_Free (lens);
	NCI_Free (fdisps);

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG)

err_out:;
	return err;
}

int ncchkioi_load_nvar_bg (NC_chk *ncchkp, int nvar, int *varids, int *lo, int *hi) {
	int err=NC_NOERR;
	int i, j, k;
	int cid;
	int get_size;

	int nchunk;

	int dsize;
	MPI_Offset bsize;

	int *lens;
	MPI_Aint *fdisps, *mdisps;
	MPI_Status status;
	MPI_Datatype ftype, mtype;	// Memory and file datatype

	char **zbufs;

	NC *ncp = (NC *)(ncchkp->ncp);
	NC_chk_var *varp;

	NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG)
	NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_INIT)

	// -1 means all chunks
	nchunk = 0;
	for (i = 0; i < nvar; i++) {
		varp = ncchkp->vars.data + varids[i];

		for (j = lo[i]; j < hi[i]; j++) {
			cid = varp->mychunks[j];
			if (varp->chunk_cache[cid] == NULL && varp->chunk_index[cid].len > 0) { nchunk++; }
		}
	}

	// Allocate buffer for I/O
	lens   = (int *)NCI_Malloc (sizeof (int) * nchunk);
	fdisps = (MPI_Aint *)NCI_Malloc (sizeof (MPI_Aint) * nchunk * 2);
	mdisps = fdisps + nchunk;
	zbufs  = (char **)NCI_Malloc (sizeof (char *) * nchunk);

	/* Carry our coll I/O
	 * OpenMPI will fail when set view or do I/O on type created with MPI_Type_create_hindexed when
	 * count is 0 We use a dummy call inplace of type with 0 count
	 */
	if (nchunk > 0) {
		// Create file type
		bsize = 0;
		k	  = 0;
		for (i = 0; i < nvar; i++) {
			varp = ncchkp->vars.data + varids[i];

			for (j = lo[i]; j < hi[i]; j++) {
				cid = varp->mychunks[j];

				// We only need to read when it is not in cache
				if (varp->chunk_cache[cid] == NULL && varp->chunk_index[cid].len > 0) {
					// offset and length of compressed chunks
					lens[k]	  = varp->chunk_index[cid].len;
					fdisps[k] = (MPI_Aint) (varp->chunk_index[cid].off + ncp->begin_var);
					mdisps[k] = bsize;
					// At the same time, we record the size of buffer we need
					bsize += (MPI_Offset)lens[k++];
				}
			}
		}

		// Allocate buffer for compressed data
		// We allocate it continuously so no mem type needed
		zbufs[0] = (char *)NCI_Malloc (bsize);
		for (j = 1; j < nchunk; j++) { zbufs[j] = zbufs[j - 1] + lens[j - 1]; }

		ncchkioi_sort_file_offset (k, fdisps, mdisps, lens);

		MPI_Type_create_hindexed (nchunk, lens, fdisps, MPI_BYTE, &ftype);
		CHK_ERR_TYPE_COMMIT (&ftype);

		MPI_Type_create_hindexed (nchunk, lens, mdisps, MPI_BYTE, &mtype);
		CHK_ERR_TYPE_COMMIT (&mtype);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_INIT)
		NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_RD)

		// Perform MPI-IO
		// Set file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, ftype, "native", MPI_INFO_NULL);
		// Write data
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, zbufs[0], 1, mtype, &status);
		// Restore file view
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

#ifdef _USE_MPI_PUT_COUNT
		MPI_Get_count (&status, MPI_BYTE, &get_size);
#else
		MPI_Type_size (ftype, &get_size);
#endif
		ncchkp->getsize += get_size;

		// Free type
		MPI_Type_free (&ftype);
		MPI_Type_free (&mtype);

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_PUT_BG_RD, NC_CHK_TIMER_PUT_BG_CACHE)

		k = 0;
		for (i = 0; i < nvar; i++) {
			varp  = ncchkp->vars.data + varids[i];
			dsize = varp->chunksize;

			// Decompress each chunk
			if (varp->filter_driver != NULL) {
				varp->filter_driver->init (MPI_INFO_NULL);

				for (j = lo[i]; j < hi[i]; j++) {
					cid = varp->mychunks[j];

					// Allocate chunk cache if not allocated
					if (varp->chunk_cache[cid] == NULL) {
						err =
							ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
						CHK_ERR
						// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

						// Perform decompression
						if (varp->chunk_index[cid].len > 0) {
							// Prepare variable context for decompression
							NCCHK_var_context ctx = (NCCHK_var_context){0};
							ctx.sz_abs_err_bound = varp->sz_abs_err_bound;
							ctx.sz_rel_bound_ratio = varp->sz_rel_bound_ratio;
							ctx.zlib_level = varp->zlib_level;
							ctx.varid = varp->varid;
#ifdef ENABLE_IPCOMP
							ctx.ipcomp_layers = varp->ipcomp_layers;
							ctx.ipcomp_interp = varp->ipcomp_interp;
							ctx.ipcomp_direction = varp->ipcomp_direction;
							ctx.ipcomp_level_progressive = varp->ipcomp_level_progressive;
							ctx.ipcomp_block_size = varp->ipcomp_block_size;
							ctx.ipcomp_interp_dim_limit = varp->ipcomp_interp_dim_limit;
							ctx.ipcomp_ebs = varp->ipcomp_ebs;
							ctx.ipcomp_num_ebs = varp->ipcomp_num_ebs;
							ctx.ipcomp_data_range = varp->ipcomp_data_range;
							ctx.ipcomp_data_min = varp->ipcomp_data_min;
							ctx.ipcomp_data_max = varp->ipcomp_data_max;
							ctx.ipcomp_has_minmax = varp->ipcomp_has_minmax;
							ctx.ipcomp_has_fill = varp->ipcomp_has_fill;
							ctx.ipcomp_fill_value = varp->ipcomp_fill_value;
#endif

							NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_DECOM)
							varp->filter_driver->decompress (zbufs[k], lens[k], varp->chunk_cache[cid]->buf,
												   &dsize, varp->ndim, varp->chunkdim, varp->etype, &ctx);
							if (dsize != varp->chunksize) { printf ("Decompress Error\n"); }
							k++;
							NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_PUT_BG_DECOM,
												 NC_CHK_TIMER_PUT_BG_CACHE)
						} else {
							memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
						}
					} else {
						// Cache is always up to date, no need to read and decompress
						ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
					}
				}
				varp->filter_driver->finalize ();
			} else {
				for (j = lo[i]; j < hi[i]; j++) {
					cid = varp->mychunks[j];

					// Allocate chunk cache if not allocated
					if (varp->chunk_cache[cid] == NULL) {
						err =
							ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
						CHK_ERR
						// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);

						if (varp->chunk_index[cid].len > 0) {
							NC_CHK_TIMER_START (NC_CHK_TIMER_PUT_BG_DECOM)
							memcpy (varp->chunk_cache[cid]->buf, zbufs[k], lens[k]);
							k++;
							NC_CHK_TIMER_STOPEX (NC_CHK_TIMER_PUT_BG_DECOM,
												 NC_CHK_TIMER_PUT_BG_CACHE)
						} else {
							memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
						}
					} else {
						// Cache is always up to date, no need to read and decompress
						ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
					}
				}
			}
		}

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_CACHE)
	} else {
		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_PUT_BG_INIT, NC_CHK_TIMER_PUT_BG_CACHE)

		for (i = 0; i < nvar; i++) {
			varp = ncchkp->vars.data + varids[i];

			for (j = lo[i]; j < hi[i]; j++) {
				cid = varp->mychunks[j];

				// Allocate chunk cache if not allocated
				if (varp->chunk_cache[cid] == NULL) {
					err = ncchkioi_cache_alloc (ncchkp, varp->chunksize, varp->chunk_cache + cid);
					CHK_ERR
					// varp->chunk_cache[cid] = (char*)NCI_Malloc(varp->chunksize);
					memset (varp->chunk_cache[cid]->buf, 0, varp->chunksize);
				} else {
					ncchkioi_cache_visit (ncchkp, varp->chunk_cache[cid]);
				}
			}
		}

		NC_CHK_TIMER_SWAP (NC_CHK_TIMER_PUT_BG_CACHE, NC_CHK_TIMER_PUT_BG_RD)

		// Follow coll I/O with dummy call
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);
		CHK_ERR_READ_AT_ALL (ncp->collective_fh, 0, &i, 0, MPI_BYTE, &status);
		CHK_ERR_SET_VIEW (ncp->collective_fh, 0, MPI_BYTE, MPI_BYTE, "native", MPI_INFO_NULL);

		NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG_RD)
	}

	// Free buffers
	if (nchunk > 0) { NCI_Free (zbufs[0]); }
	NCI_Free (zbufs);

	NCI_Free (lens);
	NCI_Free (fdisps);

	NC_CHK_TIMER_STOP (NC_CHK_TIMER_PUT_BG)

err_out:;
	return err;
}
