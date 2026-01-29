/*
 * 并行读写NetCDF文件的示例：forcing2d_raw2chunk_xy
 * 功能：读取原始NetCDF文件（无chunking/compression），重新写入带有chunking和IPCOMP压缩的新文件
 * 从文件名中自动提取变量名，使用xy checkboard partition进行并行处理
 * 输出文件供forcing2d_average_v1_xy.c等脚本读取
 */

/*mpiexec -n 14 ./forcing2d_raw2chunk_xy \
  -i /N/project/hpc_innovation_slate/ELM_Dataset/clmforc.Daymet4.1km.FLDS.2014-01.nc \
  -o /output/path/clmforc.Daymet4.1km.FLDS.2014-01_chunked.nc*/
  #include <stdio.h>
  #include <stdlib.h>
  #include <string.h>
  #include <libgen.h>
  #include <mpi.h>
  #include <pnetcdf.h>
  #include <unistd.h>  /* 用于getopt */
  
  /* 错误处理宏 */
  #define CHECK_ERR(err) { \
      if (err != NC_NOERR) { \
          printf("Error at line %d: %s\n", __LINE__, ncmpi_strerror(err)); \
          MPI_Abort(MPI_COMM_WORLD, -1); \
          return 1; \
      } \
  }
  
  #define CALC_START_COUNT(len, nprocs, rank, start, count) { \
      count = len / nprocs; \
      start = count * rank; \
      if (rank < len % nprocs) { \
          start += rank; \
          count++; \
      } \
      else { \
          start += len % nprocs; \
      } \
  }
  
  /* 最大文件路径长度 */
  #define MAX_PATH_LEN 1024
  
  /* 从文件名中提取变量名 */
  int extract_variable_name(const char *filename, char *var_name) {
      char *basename_copy = strdup(filename);
      char *base = basename(basename_copy);
      
      /* 查找模式: clmforc.Daymet4.1km.VARNAME.YYYY-MM.nc */
      char *token;
      char *save_ptr = NULL;
      int token_count = 0;
      
      token = strtok_r(base, ".", &save_ptr);
      while (token != NULL) {
          token_count++;
          if (token_count == 4) { /* 第4个部分是变量名 */
              strcpy(var_name, token);
              free(basename_copy);
              return 0;
          }
          token = strtok_r(NULL, ".", &save_ptr);
      }
      
      free(basename_copy);
      return -1; /* 未找到变量名 */
  }
  
  /* 复制变量属性 */
  int copy_variable_attributes(int ncid_in, int varid_in, int ncid_out, int varid_out) {
      int ret;
      int natts;
      
      /* 获取变量属性数量 */
      ret = ncmpi_inq_varnatts(ncid_in, varid_in, &natts);
      CHECK_ERR(ret);
      
      /* 复制每个属性 */
      for (int i = 0; i < natts; i++) {
          char att_name[NC_MAX_NAME+1];
          ret = ncmpi_inq_attname(ncid_in, varid_in, i, att_name);
          CHECK_ERR(ret);
          
          ret = ncmpi_copy_att(ncid_in, varid_in, att_name, ncid_out, varid_out);
          CHECK_ERR(ret);
      }
      
      return 0;
  }
  
  /* 复制全局属性 */
  int copy_global_attributes(int ncid_in, int ncid_out) {
      int ret;
      int ngatts;
      
      /* 获取全局属性数量 */
      ret = ncmpi_inq_natts(ncid_in, &ngatts);
      CHECK_ERR(ret);
      
      /* 复制每个全局属性 */
      for (int i = 0; i < ngatts; i++) {
          char att_name[NC_MAX_NAME+1];
          ret = ncmpi_inq_attname(ncid_in, NC_GLOBAL, i, att_name);
          CHECK_ERR(ret);
          
          ret = ncmpi_copy_att(ncid_in, NC_GLOBAL, att_name, ncid_out, NC_GLOBAL);
          CHECK_ERR(ret);
      }
      
      return 0;
  }
  
  /* 显示使用帮助 */
  void show_usage(const char *program_name) {
      printf("Usage: %s [options]\n", program_name);
      printf("Options:\n");
      printf("  -i <input_file>  指定输入文件路径\n");
      printf("  -o <output_file> 指定输出文件路径\n");
      printf("  -h               显示帮助信息\n");
      printf("Example:\n");
      printf("  mpiexec -n 14 %s -i /path/clmforc.Daymet4.1km.FLDS.2014-01.nc -o /output/clmforc.Daymet4.1km.FLDS.2014-01_chunked.nc\n", program_name);
  }
  
  int main(int argc, char **argv) {
      int ret, i;
      int ncid_in, ncid_out, varid_in, varid_out;
      int *dimids_in, *dimids_out, ndims;
      MPI_Offset *dim_sizes_in, *dim_sizes_out;
      int global_rank, global_size;
      char input_file[MAX_PATH_LEN] = "";
      char output_file[MAX_PATH_LEN] = "";
      char var_name[NC_MAX_NAME+1];
      float *buffer = NULL;
      MPI_Info info;
      
      /* 添加计时变量 */
      double start_time, end_time, process_time;
      double read_start, read_end, read_time;
      double write_start_time, write_end_time, write_time;
      double total_read_time, total_write_time, total_time;
  
      /* 参数相关变量 */
      int opt;
      
      /* xy partition相关变量 */
      int psize[2], err, rank_y, rank_x;
  
      /* 初始化MPI */
      MPI_Init(&argc, &argv);
      MPI_Comm_rank(MPI_COMM_WORLD, &global_rank);
      MPI_Comm_size(MPI_COMM_WORLD, &global_size);
      
      /* 开始计时 - 整个程序开始 */
      start_time = MPI_Wtime();
  
      /* 解析命令行参数 */
      while ((opt = getopt(argc, argv, "i:o:h")) != -1) {
          switch (opt) {
              case 'i':
                  strcpy(input_file, optarg);
                  break;
              case 'o':
                  strcpy(output_file, optarg);
                  break;
              case 'h':
                  if (global_rank == 0) {
                      show_usage(argv[0]);
                  }
                  MPI_Finalize();
                  return 0;
              default:
                  if (global_rank == 0) {
                      fprintf(stderr, "Unknown option: %c\n", opt);
                      show_usage(argv[0]);
                  }
                  MPI_Finalize();
                  return 1;
          }
      }
      
      /* 检查必要参数 */
      if (input_file[0] == '\0' || output_file[0] == '\0') {
          if (global_rank == 0) {
              fprintf(stderr, "Error: Missing required parameters\n");
              show_usage(argv[0]);
          }
          MPI_Finalize();
          return 1;
      }
      
      /* 从文件名中提取变量名 */
      if (extract_variable_name(input_file, var_name) != 0) {
          if (global_rank == 0) {
              fprintf(stderr, "Error: Could not extract variable name from filename: %s\n", input_file);
          }
          MPI_Finalize();
          return 1;
      }
      
      if (global_rank == 0) {
          printf("Input file: %s\n", input_file);
          printf("Output file: %s\n", output_file);
          printf("Variable name: %s\n", var_name);
      }
      
      /* 创建MPI信息对象 */
      MPI_Info_create(&info);
      
      /* 第一阶段：打开输入文件，获取文件结构信息 */
      if (global_rank == 0) {
          printf("Start reading input file structure...\n");
      }
      
      /* 开始读取计时 */
      read_start = MPI_Wtime();
  
      /* 打开输入文件 */
      ret = ncmpi_open(MPI_COMM_WORLD, input_file, NC_NOWRITE, info, &ncid_in);
      CHECK_ERR(ret);
  
      /* 获取变量ID */
      ret = ncmpi_inq_varid(ncid_in, var_name, &varid_in);
      CHECK_ERR(ret);
      
      /* 获取变量维度数 */
      ret = ncmpi_inq_varndims(ncid_in, varid_in, &ndims);
      CHECK_ERR(ret);
      
      if (ndims != 3) {
          printf("Error: Expected 3 dimensions (time, y, x) but found %d dimensions\n", ndims);
          MPI_Abort(MPI_COMM_WORLD, -1);
          return 1;
      }
      
      /* 分配维度ID和大小数组 */
      dimids_in = (int *)malloc(ndims * sizeof(int));
      dim_sizes_in = (MPI_Offset *)malloc(ndims * sizeof(MPI_Offset));
      
      /* 获取变量的维度ID */
      ret = ncmpi_inq_vardimid(ncid_in, varid_in, dimids_in);
      CHECK_ERR(ret);
      
      /* 获取每个维度的大小和名称 */
      char **dim_names = (char **)malloc(ndims * sizeof(char *));
      for (i = 0; i < ndims; i++) {
          dim_names[i] = (char *)malloc((NC_MAX_NAME+1) * sizeof(char));
          ret = ncmpi_inq_dimname(ncid_in, dimids_in[i], dim_names[i]);
          CHECK_ERR(ret);
          
          ret = ncmpi_inq_dimlen(ncid_in, dimids_in[i], &dim_sizes_in[i]);
          CHECK_ERR(ret);
      }
      
      MPI_Offset NTIMES = dim_sizes_in[0];
      MPI_Offset NY = dim_sizes_in[1];
      MPI_Offset NX = dim_sizes_in[2];
  
      if (global_rank == 0) {
          printf("Dimensions: time=%lld, y=%lld, x=%lld\n", NTIMES, NY, NX);
      }
  
      /* 设置xy checkboard partition */
      psize[0] = psize[1] = 0;
      err = MPI_Dims_create(global_size, 2, psize);
      if (global_rank == 0) {
          printf("MPI_Dims_create() 2D: psize=%d %d\n", psize[0], psize[1]);
      }
      rank_y = global_rank / psize[1];
      rank_x = global_rank % psize[1];
  
      /* 分配读取起始位置和计数数组 */
      MPI_Offset start[3], count[3];
  
      CALC_START_COUNT(NY, psize[0], rank_y, start[1], count[1])
      CALC_START_COUNT(NX, psize[1], rank_x, start[2], count[2])
      start[0] = 0;
      count[0] = 1;
  
      if (global_rank == 31) {
          printf("start=%lld, count=%lld\n", start[0], count[0]);
          printf("start=%lld, count=%lld\n", start[1], count[1]);
          printf("start=%lld, count=%lld\n", start[2], count[2]);
      }
  
      /* 第二阶段：创建输出文件 */
      if (global_rank == 0) {
          printf("Start creating output file...\n");
      }
      
      /* 开始写入计时 */
      write_start_time = MPI_Wtime();
  
      /* 设置chunking和compression信息 */
      MPI_Info_free(&info);
      MPI_Info_create(&info);
      MPI_Info_set(info, "nc_chunking", "enable");
      MPI_Info_set(info, "nc_chunk_default_filter", "ipcomp");
  
      /* 创建输出文件 */
      ret = ncmpi_create(MPI_COMM_WORLD, output_file, NC_64BIT_DATA, info, &ncid_out);
      CHECK_ERR(ret);
      
      /* 为输出文件创建维度 */
      dimids_out = (int *)malloc(ndims * sizeof(int));
      dim_sizes_out = (MPI_Offset *)malloc(ndims * sizeof(MPI_Offset));
      
      for (i = 0; i < ndims; i++) {
          dim_sizes_out[i] = dim_sizes_in[i];
          if (i == 0) {
              /* time维度设置为无限维 */
              ret = ncmpi_def_dim(ncid_out, dim_names[i], NC_UNLIMITED, &dimids_out[i]);
          } else {
              ret = ncmpi_def_dim(ncid_out, dim_names[i], dim_sizes_out[i], &dimids_out[i]);
          }
          CHECK_ERR(ret);
      }
  
      /* 创建输出变量 */
      ret = ncmpi_def_var(ncid_out, var_name, NC_FLOAT, ndims, dimids_out, &varid_out);
      CHECK_ERR(ret);
  
      /* 设置chunk大小 */
      int chunk_dim[3];
      chunk_dim[0] = 1; /* time维度chunk为1 */
      chunk_dim[1] = NY / psize[0];
      if (NY % psize[0]) chunk_dim[1]++;
      chunk_dim[2] = NX / psize[1];
      if (NX % psize[1]) chunk_dim[2]++;
  
      if (global_rank == 0) {
          printf("Writing: chunk_dim=%d %d %d\n", chunk_dim[0], chunk_dim[1], chunk_dim[2]);
      }
  
      /* 设置chunking和compression */
      err = ncmpi_var_set_chunk(ncid_out, varid_out, chunk_dim);
      CHECK_ERR(err);
      err = ncmpi_var_set_filter(ncid_out, varid_out, NC_FILTER_IPCOMP);
      CHECK_ERR(err);

#ifdef ENABLE_IPCOMP
      const char *comp_algo = "ipcomp";
      const char *comp_interp = "cubic";
      const char *comp_codec_version = "1.0.0";
      int comp_layers = 1;
      int comp_level_progressive = 0;
      int comp_block_size = 0; /* 0 lets codec decide */
      double comp_data_range = 0.0; /* codec can autotune */

      ret = ncmpi_put_att_text(ncid_out, varid_out, "comp:algo",
                               (MPI_Offset)strlen(comp_algo), comp_algo);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_text(ncid_out, varid_out, "comp:interp",
                               (MPI_Offset)strlen(comp_interp), comp_interp);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_text(ncid_out, varid_out, "comp:codec_version",
                               (MPI_Offset)strlen(comp_codec_version), comp_codec_version);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_int(ncid_out, varid_out, "comp:layers",
                              NC_INT, 1, &comp_layers);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_int(ncid_out, varid_out, "comp:level_progressive",
                              NC_INT, 1, &comp_level_progressive);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_int(ncid_out, varid_out, "comp:block_size",
                              NC_INT, 1, &comp_block_size);
      CHECK_ERR(ret);
      ret = ncmpi_put_att_double(ncid_out, varid_out, "comp:data_range",
                                 NC_DOUBLE, 1, &comp_data_range);
      CHECK_ERR(ret);
#endif
  
      // /* 复制全局属性 */
      // copy_global_attributes(ncid_in, ncid_out);
      
      // /* 复制变量属性 */
      // copy_variable_attributes(ncid_in, varid_in, ncid_out, varid_out);
      
      /* 添加处理信息到全局属性 */
      char processing_info[] = "Converted from raw to chunked+IPCOMP-compressed format";
      ret = ncmpi_put_att_text(ncid_out, NC_GLOBAL, "processing_info", strlen(processing_info), processing_info);
      CHECK_ERR(ret);
  
      /* 结束定义模式 */
      ret = ncmpi_enddef(ncid_out);
      CHECK_ERR(ret);
  
      /* 第三阶段：读取和写入数据 */
      if (global_rank == 0) {
          printf("Start reading and writing data...\n");
      }
      
      /* 分配内存用于读取数据 */
      MPI_Offset local_elements = count[1] * count[2];
      buffer = (float *)malloc(local_elements * sizeof(float));
      if (buffer == NULL) {
          printf("Error: Memory allocation failed for buffer\n");
          MPI_Abort(MPI_COMM_WORLD, -1);
          return 1;
      }
  
      /* 逐时间步读取和写入数据 */
      for (int t = 0; t < NTIMES; t++) {
          start[0] = t;
          count[0] = 1;
          /* 读取数据 */
          ret = ncmpi_get_vara_float_all(ncid_in, varid_in, start, count, buffer);
          CHECK_ERR(ret);
          /* 写入数据 */
          ret = ncmpi_put_vara_float_all(ncid_out, varid_out, start, count, buffer);
          // ret = ncmpi_iput_vara_float(ncid_out, varid_out, start, count, buffer, NULL);
          CHECK_ERR(ret);
      }
      // ret = ncmpi_wait_all(ncid_out, NC_REQ_ALL, NULL, NULL);
      // CHECK_ERR(ret);
      /* 关闭文件 */
      ret = ncmpi_close(ncid_in);
      CHECK_ERR(ret);
      ret = ncmpi_close(ncid_out);
      CHECK_ERR(ret);
      /* 结束读取计时 */
      read_end = MPI_Wtime();
      read_time = read_end - read_start;
      /* 使用MPI_Reduce收集所有进程的读取时间，取最大值 */
      MPI_Reduce(&read_time, &total_read_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
      /* 结束写入计时 */
      write_end_time = MPI_Wtime();
      write_time = write_end_time - write_start_time;
      /* 使用MPI_Reduce收集所有进程的写入时间，取最大值 */
      MPI_Reduce(&write_time, &total_write_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
      /* 释放资源 */
      free(dimids_in);
      free(dimids_out);
      free(dim_sizes_in);
      free(dim_sizes_out);
      
      for (i = 0; i < ndims; i++) {
          free(dim_names[i]);
      }
      free(dim_names);
      
      free(buffer);
      
      MPI_Info_free(&info);
      
      if (global_rank == 0) {
          printf("Successfully completed! Output file: %s\n", output_file);
      }
      
      /* 结束计时 - 整个程序结束*/
      end_time = MPI_Wtime();
      process_time = end_time - start_time;
      /* 使用MPI_Reduce收集所有进程的总时间，取最大值 */
      MPI_Reduce(&process_time, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  
      if (global_rank == 0) {
          printf("===== Performance statistics =====\n");
          printf("Total read time: %.4f seconds\n", total_read_time);
          printf("Total write time: %.4f seconds\n", total_write_time);
          printf("Total execution time: %.4f seconds\n", total_time);
      }
      
      MPI_Finalize();
      return 0;
  }