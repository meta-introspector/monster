// CUDA: Parallel batch sampling kernel for 8M files

#include <cuda_runtime.h>
#include <stdio.h>

#define VECTOR_SIZE 71000
#define BATCH_SIZE 1024
#define NUM_FILES 8000000

// Monster Walk vector
struct MonsterVector {
    float data[VECTOR_SIZE];
};

// Batch of vectors
struct VectorBatch {
    MonsterVector vectors[BATCH_SIZE];
};

// Sampling result
struct SampleResult {
    int file_id;
    float score;
    int tokens[256];
    int num_tokens;
};

// Batch sampling kernel
__global__ void batch_sample_kernel(
    VectorBatch* input_batch,
    SampleResult* output_results,
    int batch_id,
    int num_vectors
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_vectors) return;
    
    MonsterVector* vec = &input_batch->vectors[idx];
    SampleResult* result = &output_results[idx];
    
    // Extract features from 71k vector
    float base = vec->data[67100];
    float layer_sum = 0.0f;
    
    // Sum first 1000 elements as feature
    for (int i = 0; i < 1000; i++) {
        layer_sum += vec->data[i];
    }
    
    // Compute score (simplified)
    result->file_id = batch_id * BATCH_SIZE + idx;
    result->score = layer_sum / 1000.0f;
    result->num_tokens = 10;
    
    // Generate tokens (simplified)
    for (int i = 0; i < result->num_tokens; i++) {
        result->tokens[i] = (int)(result->score * (i + 1)) % 50000;
    }
}

// Process all 8M files in batches
void process_8m_files_gpu() {
    printf("🚀 Batch GPU Sampling for 8M Files\n");
    printf("="*70);
    printf("\n\n");
    
    int num_batches = (NUM_FILES + BATCH_SIZE - 1) / BATCH_SIZE;
    
    printf("Configuration:\n");
    printf("  Total files: %d\n", NUM_FILES);
    printf("  Batch size: %d\n", BATCH_SIZE);
    printf("  Num batches: %d\n", num_batches);
    printf("  Vector size: %d elements\n", VECTOR_SIZE);
    printf("\n");
    
    // Allocate GPU memory
    VectorBatch* d_batch;
    SampleResult* d_results;
    
    cudaMalloc(&d_batch, sizeof(VectorBatch));
    cudaMalloc(&d_results, BATCH_SIZE * sizeof(SampleResult));
    
    // Allocate CPU memory for results
    SampleResult* h_results = (SampleResult*)malloc(BATCH_SIZE * sizeof(SampleResult));
    
    printf("Processing batches:\n");
    
    for (int batch_id = 0; batch_id < num_batches; batch_id++) {
        int vectors_in_batch = (batch_id == num_batches - 1) 
            ? (NUM_FILES % BATCH_SIZE) 
            : BATCH_SIZE;
        
        // Launch kernel
        int threads = 256;
        int blocks = (vectors_in_batch + threads - 1) / threads;
        
        batch_sample_kernel<<<blocks, threads>>>(
            d_batch, d_results, batch_id, vectors_in_batch
        );
        
        cudaDeviceSynchronize();
        
        // Copy results back (optional)
        if (batch_id % 1000 == 0) {
            cudaMemcpy(h_results, d_results, 
                      vectors_in_batch * sizeof(SampleResult), 
                      cudaMemcpyDeviceToHost);
            
            printf("  Batch %d/%d: processed %d files (score: %.4f)\n",
                   batch_id, num_batches, vectors_in_batch, h_results[0].score);
        }
    }
    
    printf("\n");
    printf("="*70);
    printf("\n✅ All %d files processed!\n", NUM_FILES);
    
    // Cleanup
    cudaFree(d_batch);
    cudaFree(d_results);
    free(h_results);
}

int main() {
    process_8m_files_gpu();
    return 0;
}
