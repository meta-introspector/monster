// Example: Using libzkprologml
// Read kernel samples, generate zkSNARK proofs, export to Prolog

#include "zkprologml.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    printf("🔐 libzkprologml Example\n");
    printf("========================\n\n");
    
    // Initialize library
    zkprologml_ctx_t *ctx = zkprologml_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize libzkprologml\n");
        return 1;
    }
    
    // Create batch
    zkprologml_batch_t *batch = zkprologml_batch_create(1000);
    if (!batch) {
        fprintf(stderr, "Failed to create batch\n");
        zkprologml_free(ctx);
        return 1;
    }
    
    // Read samples from kernel
    printf("📖 Reading samples from kernel...\n");
    int ret = zkprologml_read_samples(ctx, batch->samples, 1000, &batch->count);
    if (ret < 0) {
        fprintf(stderr, "Failed to read samples: %s\n", zkprologml_get_error(ctx));
        zkprologml_batch_free(batch);
        zkprologml_free(ctx);
        return 1;
    }
    printf("   Read %zu samples\n\n", batch->count);
    
    // Process batch (generate proofs and facts)
    printf("🔐 Generating zkSNARK proofs...\n");
    ret = zkprologml_batch_process(ctx, batch);
    if (ret < 0) {
        fprintf(stderr, "Failed to process batch\n");
        zkprologml_batch_free(batch);
        zkprologml_free(ctx);
        return 1;
    }
    printf("   Generated %zu proofs\n\n", batch->count);
    
    // Verify batch
    printf("✅ Verifying proofs...\n");
    bool verified = zkprologml_batch_verify(ctx, batch);
    printf("   Verification: %s\n\n", verified ? "PASS" : "FAIL");
    
    // Export samples
    printf("💾 Exporting data...\n");
    zkprologml_export_samples(ctx, "samples.bin", batch->samples, batch->count);
    printf("   Samples: samples.bin\n");
    
    zkprologml_export_proofs(ctx, "proofs.bin", batch->proofs, batch->count);
    printf("   Proofs: proofs.bin\n");
    
    zkprologml_export_prolog(ctx, "facts.pl", batch->facts, batch->count);
    printf("   Prolog: facts.pl\n\n");
    
    // Print statistics
    printf("📊 Statistics:\n");
    for (int i = 0; i < MONSTER_PRIMES_COUNT; i++) {
        size_t count, capacity;
        zkprologml_get_ring_stats(ctx, i, &count, &capacity);
        printf("   Ring %2d (prime %2d): %zu/%zu\n",
               i, MONSTER_PRIMES[i], count, capacity);
    }
    printf("\n");
    
    // Convert to tensor
    printf("🎯 Converting to ML tensor...\n");
    zkprologml_tensor_t tensor;
    ret = zkprologml_to_tensor(ctx, batch->samples, batch->count, &tensor);
    if (ret == 0) {
        printf("   Tensor shape: [%zu, %zu, %zu]\n",
               tensor.dims[0], tensor.dims[1], tensor.dims[2]);
        printf("   Total elements: %zu\n", tensor.total_size);
        free(tensor.data);
    }
    printf("\n");
    
    // Cleanup
    zkprologml_batch_free(batch);
    zkprologml_free(ctx);
    
    printf("✅ Complete!\n");
    return 0;
}
