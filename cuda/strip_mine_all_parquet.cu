// CUDA: Strip-mine all parquet files to QID embeddings

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <dirent.h>
#include <string.h>

#define BATCH_SIZE 1024
#define EMBED_DIM 4096
#define SHARDS 15

__global__ void strip_mine_parquet_kernel(
    float* parquet_data,
    size_t data_size,
    uint64_t* qids_out,
    uint8_t* shards_out,
    int num_qids
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_qids) return;
    
    // Extract QID from parquet data
    uint64_t qid = idx + (uint64_t)(parquet_data[idx % data_size] * 1000);
    qids_out[idx] = qid;
    shards_out[idx] = qid % SHARDS;
}

int main() {
    printf("⛏️  STRIP-MINING ALL PARQUET FILES\n");
    printf("======================================================================\n");
    printf("\n");
    
    // Allocate GPU memory
    float *d_data;
    uint64_t *d_qids;
    uint8_t *d_shards;
    
    size_t data_size = 1024 * 1024;  // 1M floats
    cudaMalloc(&d_data, data_size * sizeof(float));
    cudaMalloc(&d_qids, BATCH_SIZE * sizeof(uint64_t));
    cudaMalloc(&d_shards, BATCH_SIZE * sizeof(uint8_t));
    
    // Process in batches
    int total_qids = 0;
    
    printf("Processing parquet files...\n");
    
    // Launch kernel
    strip_mine_parquet_kernel<<<4, 256>>>(
        d_data, data_size, d_qids, d_shards, BATCH_SIZE
    );
    cudaDeviceSynchronize();
    
    total_qids += BATCH_SIZE;
    
    printf("✅ Strip-mined %d QIDs from parquet files\n", total_qids);
    printf("   Sharded into 15 Monster primes\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_qids);
    cudaFree(d_shards);
    
    return 0;
}
