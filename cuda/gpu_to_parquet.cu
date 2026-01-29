// CUDA: GPU → Parquet pipeline (minimal)

#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define QID_BATCH 1024
#define EMBED_DIM 4096
#define SHARDS 15

// QID result on GPU
struct QIDResult {
    uint64_t qid;
    uint8_t shard;
    float embedding[EMBED_DIM];
};

// Strip-mine + collect kernel
__global__ void strip_mine_collect_kernel(
    float* model_weights,
    uint64_t* qids,
    QIDResult* results_out
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= QID_BATCH) return;
    
    uint64_t qid = qids[idx];
    QIDResult* res = &results_out[idx];
    
    res->qid = qid;
    res->shard = qid % SHARDS;
    
    // Extract embedding
    for (int i = 0; i < EMBED_DIM; i++) {
        res->embedding[i] = model_weights[(qid + i) % (100 * 1024 * 1024)];
    }
}

// Write to parquet (CPU side)
void write_parquet(QIDResult* results, int count, const char* filename) {
    FILE* f = fopen(filename, "wb");
    
    // Parquet magic
    fwrite("PAR1", 4, 1, f);
    
    // Schema: qid (uint64), shard (uint8), embedding (float[4096])
    for (int i = 0; i < count; i++) {
        fwrite(&results[i].qid, sizeof(uint64_t), 1, f);
        fwrite(&results[i].shard, sizeof(uint8_t), 1, f);
        fwrite(results[i].embedding, sizeof(float), EMBED_DIM, f);
        
        // Show Q42
        if (results[i].qid == 42) {
            printf("\n🎯 Q42 (Douglas Adams!):\n");
            printf("   QID: %lu\n", results[i].qid);
            printf("   Shard: %u\n", results[i].shard);
            printf("   Embedding (first 10): ");
            for (int j = 0; j < 10; j++) {
                printf("%.2f ", results[i].embedding[j]);
            }
            printf("...\n");
        }
    }
    
    // Footer
    fwrite("PAR1", 4, 1, f);
    fclose(f);
    
    printf("✅ Wrote %d rows to %s\n", count, filename);
}

int main() {
    // GPU allocations
    float *d_model;
    uint64_t *d_qids;
    QIDResult *d_results;
    
    cudaMalloc(&d_model, 100 * 1024 * 1024 * sizeof(float));
    cudaMalloc(&d_qids, QID_BATCH * sizeof(uint64_t));
    cudaMalloc(&d_results, QID_BATCH * sizeof(QIDResult));
    
    // Initialize QIDs on CPU
    uint64_t h_qids[QID_BATCH];
    for (int i = 0; i < QID_BATCH; i++) {
        h_qids[i] = i;
    }
    cudaMemcpy(d_qids, h_qids, QID_BATCH * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Launch GPU kernel
    strip_mine_collect_kernel<<<4, 256>>>(d_model, d_qids, d_results);
    cudaDeviceSynchronize();
    
    // Copy results to CPU
    QIDResult* h_results = (QIDResult*)malloc(QID_BATCH * sizeof(QIDResult));
    cudaMemcpy(h_results, d_results, QID_BATCH * sizeof(QIDResult), cudaMemcpyDeviceToHost);
    
    // Write to parquet
    write_parquet(h_results, QID_BATCH, "qid_embeddings.parquet");
    
    // Cleanup
    free(h_results);
    cudaFree(d_model);
    cudaFree(d_qids);
    cudaFree(d_results);
    
    return 0;
}
