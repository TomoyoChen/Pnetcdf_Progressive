# Supporting Data Compression in PnetCDF —— 核心实现与 Chunk 机制小结

> 本文档总结 HKL21《Supporting Data Compression in PnetCDF》一文中**核心实现**，特别是基于 **chunk 机制支持压缩** 的设计与实现细节。

---

## 1. 设计目标与整体思路

经典 NetCDF（classic format）本身只支持**连续布局**、不支持压缩。PnetCDF 在保持 **classic NetCDF 格式不变** 的前提下，引入类似 HDF5 的 **chunked storage layout**，并在 PnetCDF 内部完成压缩/解压与并行访问控制。

核心目标：

- 在 **classic NetCDF 文件格式不变** 的前提下支持压缩；
- 通过 **chunk 粒度压缩** 避免“大块数据必须整体重压缩”的问题；
- 在并行环境下保证 **数据一致性**（每个 chunk 只允许一个“owner”直接写）；
- 利用 PnetCDF 的 **non-blocking I/O 接口做 I/O 聚合**，缓解压缩导致的可扩展性瓶颈。

---

## 2. Chunk 与压缩的元数据设计

由于 classic NetCDF 头部没有 chunk/压缩相关条目，作者使用了一组以 `_` 开头的 **特殊属性** 来描述 chunk/压缩元数据：

### 2.1 关键属性

- **`_chunk_dims`**  
  - 类型：1D int 数组，长度 = 变量维度数  
  - 内容：每个维度对应的 **chunk 尺寸** 所用的 dimension ID（通过 `ncmpi_def_dim()` 定义）  
  - 作用：
    - 标记该变量为 **chunked 变量**（是否存在该属性作为判据）；
    - 间接说明每个 chunk 的逻辑形状。

- **`_chunk_refs`**  
  - 存储 **chunk reference table 的起始偏移**：
  - 对固定大小变量：
    - 类型：单个 `int64`，指向该变量 chunk reference table 在文件中的起始 offset。
  - 对 record 变量：
    - 类型：1D `int64` 数组，长度 = 记录数；
    - 每个元素指向该记录自己的 chunk reference table 在文件中的 offset。

- **`_chunk_ext_ndims`**  
  - 类型：单个 `int64`；
  - 含义：该变量的“有效 record 数”（已经写入的记录数），小于等于文件中记录维度的值。

- **`_filters`**  
  - 类型：int 数组；
  - 内容：对 chunk 数据应用的 **filter ID 列表**（可以是 deflate、zstd、SZ、ZFP 等），数组顺序即数据流中 filter 的流水线顺序。

- **各压缩算法相关的附加属性**  
  - 如压缩等级、误差容忍度（lossy）、等。

这些属性全部挂在 **变量级别**，因此原始 classic NetCDF 阅读器仍能看到普通变量（只是看不懂这些私有属性）。

---

## 3. Chunk Reference Table 与数据块布局

### 3.1 Chunk Reference Table 结构

作者为每个 chunked 变量（或 record）引入一种新的逻辑数据对象：**chunk reference table**。

- 对于 **固定大小变量**：  
  - reference table = 两个长度为 `num_chunks` 的 1D `int64` 数组：
    1. **offset 数组**：每个元素记录一个 chunk 在文件中的起始偏移；
    2. **size 数组**：每个元素记录压缩后该 chunk 的大小。
- 对于 **record 变量**：  
  - 每条 record 有各自的 reference table；
  - 变量属性 `_chunk_refs` 数组中每个条目指向对应 record 的 reference table 起始偏移。

**注意：**

- reference table **本质是数据区中的”隐藏变量“**，而不是 classic header 的一部分；
- **表项下标 = chunk ID**，按 row-major 顺序对应逻辑 chunk 的顺序。

### 3.2 Data Chunks 存储

- 每个 chunk 压缩后的数据称为 **data chunk**：
  - 每个 chunk 在文件中占用 **连续空间**，但不同 chunk 之间不要求连续；
  - 支持 **复用旧空间**（如果新压缩后大小不超过原空间），否则在新的“空洞”位置重新分配；
  - 当前实现不做“碎片整理/空洞回收”（作者认为修改不频繁）。

### 3.3 利用 NetCDF padding 作为“容器空间”

classic NetCDF 要求变量数据按 4 字节或文件系统条带对齐，因此变量之间会有 **对齐填充（padding）空洞**。PnetCDF 在不修改 header 结构的前提下，把这些 padding 与额外对齐空间视作 **“free space”**：

- 在这些 free space 中存放：
  - 固定大小变量的 **chunk reference tables**；
  - 所有变量的 **压缩 chunk 数据**。

这样既：

- 遵守了 classic NetCDF 对 `begin` 和对齐的规范；
- 又能在不破坏格式的前提下“藏”下新对象。

---

## 4. 固定变量与 Record 变量的 Chunk 布局

### 4.1 固定大小变量

对固定大小变量：

- 在 `ncmpi_enddef` 时：
  1. 对所有标记为 chunked 的固定变量，计算其 `num_chunks`；
  2. 在数据节开头为每个变量分配 reference table 空间；
  3. 必要时调整后续变量的 `begin` 以腾出空间；
  4. 初始化 offset 数组为 `-1`（表示尚未分配 chunk ）。

- 压缩数据写入时：
  - 真正为 chunk 分配文件空间，并在 reference table 中填入 offset 和 size。

### 4.2 Record 变量 —— “伪装成 chunked fixed-size”

classic NetCDF 中 record 变量的正规布局会把所有变量在同一 record index 上的数据放在一起，并且记录段必须排在所有固定变量之后。若在其间插 padding，会导致频繁移动后续记录，性能灾难。

为避免这一点，作者采取策略：

1. **不再使用传统 record 变量布局**；
2. 而是将 record 变量等价地视作：
   - 第一维是 record 维；  
   - **chunk 大小沿 record 维固定为 1 条 record 的大小**；  
   - 于是每个 record 变成一个“固定大小变量的 chunked 版本”，具有自己的 reference table；
3. `_chunk_refs` 属性数组中的每个元素存的是该 record 的 reference table 的 offset，未写过的记录设置为 NULL。

效果：

- 每条 record 相当于“独立的 chunked 变量”，只支持 append；
- 避免了频繁移动旧记录；
- 与 NetCDF 数据模型（只有一个共享 record 维）很好契合。

---

## 5. 新增 API：定义 chunk 与 filter

实现层面对外暴露了两个核心 API（PnetCDF 扩展）：

```c
/* 设置 chunk 尺寸（按维度） */
int ncmpi_var_set_chunk(int ncid, int varid,
                        const MPI_Offset *chunk_counts);

/* 为变量设置压缩 filter（可级联） */
int ncmpi_var_set_filter(int ncid, int varid,
                         int nfilters,
                         const int *filter_ids,
                         const void *filter_params);
```

- `ncmpi_var_set_chunk`：指定每个维度的 chunk 尺寸；
- `ncmpi_var_set_filter`：关联一个或多个压缩 filter：
  - 支持 deflate、zstd、SZ、ZFP 等插件式压缩器；
  - 对应参数用 filter-specific 属性写入文件。

---

## 6. 并行访问策略：Chunk Owner 分配

### 6.1 Owner 模型

为保证一个 chunk 在压缩/写入时的一致性：

- 每个 chunk 只允许一个 **owner process** 直接读/写文件；
- 其他进程对该 chunk 的写入请求需要发送给 owner，由 owner 负责：
  - 解压旧数据；
  - 合并写入；
  - 重新压缩；
  - 写回文件。

因此，写入压缩变量必须是 **collective** 操作。

### 6.2 访问量与负载均衡

- 定义 `access_size(p, chunk)`：
  - = 进程 p 本地所有 I/O 请求与该 chunk 的交集字节数之和；
- HDF5 策略：
  - 只考虑**最小化通信**：owner = access_size 最大的进程，负载均衡只在 tie 时考虑；
- 本文策略：
  - 引入一个 **per-process workload penalty**：
    - 与该进程已经拥有的 chunk 总大小成正比；
  - 对每个 (p, chunk) 计算：
    - `score(p,chunk) = access_size(p, chunk) - penalty(p)`；
  - owner = score 最大的进程；
  - 当某进程被分配更多 chunk 时，其 penalty 增大，**自然抑制过度集中**，实现通信与负载之间的折中。

---

## 7. 并行写入流程（压缩变量）

以 collective 写为例：

1. **计算交集 & 构造请求**
   - 每个进程根据自己的写入子数组求出涉及的 chunk 及其子区域；
   - 对同一个 owner 的多个 chunk 请求，尽量通过 **一个 MPI 消息 + 派生 datatype**（`MPI_Type_create_subarray` 或 contiguous）打包。

2. **全局通信：确定消息数量**
   - collective 操作（如 `MPI_Alltoall`）让每个 owner 知道来自每个进程的请求数量／大小。

3. **Owner 侧：管理 chunk buffer**
   - 对每个自己拥有的 chunk：
     - 分配一个 **未压缩 chunk buffer**；
     - 如果 chunk 已存在：
       - 用 MPI-IO 从文件读出压缩块；
       - 解压到 chunk buffer；
     - 如果是第一次写入：
       - 用 fill value 初始化整个 buffer。

4. **合并写入数据**
   - owner 接收来自其他进程的请求（包含 chunk ID + chunk 内部子数组 + 数据）；
   - 用派生 datatype 把数据 scatter 到 chunk buffer 的正确位置；
   - owner 自己的本地写入同样 merge 进去。

5. **压缩与分配文件空间**
   - 处理完所有请求后，owner 对 chunk buffer 做压缩；
   - 若是新 chunk：
     - 在 free space 中按 chunk ID 顺序分配空间；
     - 基于已知各 chunk size 计算文件 offset；
   - 若是修改已有 chunk：
     - 若新压缩大小 <= 旧空间：原位覆盖；
     - 否则当作新 chunk，重新分配空间（旧空间暂不回收）。

6. **写回文件 & 更新 reference table**
   - Owner 使用 MPI-IO：
     - 通过文件视图将自己拥有的所有 chunk 一次 collective 写；
   - 对每个变量：
     - 指定一个进程（通常为第一个 chunk 的 owner）负责：
       - 写/覆盖 chunk reference table（offset & size 数组）；
       - 更新 `_chunk_refs` 属性（若是首次分配）。

---

## 8. 并行读取流程（压缩变量）

读取过程基本是写入的“反向数据流”：

1. 每个进程对所需数据区域求与 chunk 的交集；
2. 构建 **read request**：
   - 包含：chunk ID、chunk 内偏移/子数组信息；
3. 发送请求给相应 chunk owner；
4. Owner：
   - 如尚未解压该 chunk，则：
     - 从文件读压缩块；
     - 解压到 chunk buffer；
   - 使用 MPI 派生类型从 chunk buffer 中选出所需子区域，发送回请求进程；
5. 请求进程：
   - 使用 MPI 派生类型把收到的数据直接 scatter 到用户 buffer 的相应位置；
   - 如用户请求类型与变量本身类型不一致，在本地做类型转换。

与 HDF5 的一大区别：

- HDF5：每个进程 **自己读+自己解压** 所需 chunk，无需额外 owner 通信，但可能对同一 chunk 多次解压；
- PnetCDF：通过 owner 共享 decompression 结果，避免重复解压，但需要额外进程间通信。

---

## 9. I/O 聚合与 non-blocking API

### 9.1 为什么需要 I/O 聚合

Chunk + 压缩的并行 I/O 有个硬约束：

- 在一个变量内部：**最大并行度 ≈ chunk 数**；
- 如果变量少、chunk 大，则并行度受限；
- 但真实应用往往同时访问 **多个变量**，甚至不同文件。

PnetCDF 的 non-blocking I/O 接口允许：

- 应用先调用大量 `ncmpi_iput_var*` / `ncmpi_iget_var*`（只登记请求，不立刻执行）；
- 最后一次性 `ncmpi_wait_all` / `ncmpi_wait` 触发实际 I/O。

这样，库可以：

- **跨变量汇总所有请求** 一起做 chunk owner 分配与通信；
- 大幅增加可同时调度的 chunk 数量；
- 通过 penalty 机制，在更多 chunk 间实现更好的 **负载均衡**。

### 9.2 实现上的关键点

- 在 chunk 访问请求中增加 **`varid` 元数据**：
  - 使 owner 能够区分不同变量的 chunk；
- 将 **owner 分配延迟到 non-blocking I/O flush** 时：
  - 获得全局视角（所有变量所有请求）；
- 在多变量 owner 分配时：
  - 一边为某些变量进行 owner 分配通信；
  - 一边对其他变量并行计算 access_size，形成 **通信与计算重叠**。

---

## 10. 小结：Chunk 机制下的压缩支持的核心要点

1. **格式兼容**：  
   - 不修改 classic NetCDF 头部结构，只利用变量属性和数据区 padding 空间；
   - 生成的文件仍是合法 classic NetCDF 文件，但只有支持该扩展的 PnetCDF 能理解压缩与 chunk 语义。

2. **Chunk 粒度压缩**：  
   - 通过 chunk reference table 管理每个 chunk 的位置与大小；
   - 支持独立压缩/解压每个 chunk，避免整体重压缩。

3. **Owner 模型 + 负载均衡**：  
   - 每个 chunk 只由一个 owner 承担压缩+I/O；
   - 通过 access_size + workload penalty 平衡通信与负载。

4. **MPI-IO + 派生类型优化**：  
   - 使用 MPI 派生类型合并请求、文件视图合并写入；
   - 尽量减少消息数量与小 I/O 操作。

5. **I/O 聚合作为“增幅器”**：  
   - 跨变量汇总所有非阻塞 I/O 请求；
   - 在 chunk 并行度受限时，通过更多变量的 chunk 扩展整体并行度与可扩展性。
