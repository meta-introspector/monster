// libzkprologml: Zero-Knowledge Prolog ML Library
// Bridge kernel samples to userspace with zkSNARK proofs

#ifndef LIBZKPROLOGML_H
#define LIBZKPROLOGML_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Version
#define ZKPROLOGML_VERSION_MAJOR 1
#define ZKPROLOGML_VERSION_MINOR 0
#define ZKPROLOGML_VERSION_PATCH 0

// Monster primes
#define MONSTER_PRIMES_COUNT 15
extern const uint8_t MONSTER_PRIMES[MONSTER_PRIMES_COUNT];

// Process sample from kernel
typedef struct {
    int32_t pid;
    uint64_t timestamp;
    uint64_t rip;
    uint64_t rsp;
    uint64_t rax;
    uint64_t rbx;
    uint64_t rcx;
    uint64_t rdx;
    uint64_t mem_usage;
    uint64_t cpu_time;
    uint8_t shard_id;
    uint8_t hecke_applied;
} zkprologml_sample_t;

// zkSNARK proof
typedef struct {
    uint8_t proof[256];      // Groth16 proof (compressed)
    uint8_t public_inputs[32]; // Public inputs hash
    uint64_t timestamp;
    uint8_t shard_id;
    bool verified;
} zkprologml_proof_t;

// Prolog fact
typedef struct {
    char predicate[64];      // e.g., "process_sample"
    char args[256];          // e.g., "pid(1234), shard(5)"
    uint64_t timestamp;
    uint8_t shard_id;
} zkprologml_fact_t;

// ML tensor
typedef struct {
    float *data;
    size_t dims[3];          // [rings, samples, features]
    size_t total_size;
    uint8_t dtype;           // 0=f32, 1=f64
} zkprologml_tensor_t;

// Context handle
typedef struct zkprologml_ctx zkprologml_ctx_t;

// Initialize library
zkprologml_ctx_t* zkprologml_init(void);

// Cleanup
void zkprologml_free(zkprologml_ctx_t *ctx);

// Read samples from kernel module
int zkprologml_read_samples(
    zkprologml_ctx_t *ctx,
    zkprologml_sample_t *samples,
    size_t max_samples,
    size_t *count
);

// Generate zkSNARK proof for sample
int zkprologml_generate_proof(
    zkprologml_ctx_t *ctx,
    const zkprologml_sample_t *sample,
    zkprologml_proof_t *proof
);

// Verify zkSNARK proof
bool zkprologml_verify_proof(
    zkprologml_ctx_t *ctx,
    const zkprologml_proof_t *proof
);

// Convert sample to Prolog fact
int zkprologml_to_prolog(
    zkprologml_ctx_t *ctx,
    const zkprologml_sample_t *sample,
    zkprologml_fact_t *fact
);

// Query Prolog knowledge base
int zkprologml_query_prolog(
    zkprologml_ctx_t *ctx,
    const char *query,
    zkprologml_fact_t *results,
    size_t max_results,
    size_t *count
);

// Convert samples to ML tensor
int zkprologml_to_tensor(
    zkprologml_ctx_t *ctx,
    const zkprologml_sample_t *samples,
    size_t count,
    zkprologml_tensor_t *tensor
);

// Apply Hecke operator
uint64_t zkprologml_apply_hecke(
    uint64_t value,
    uint8_t prime_idx
);

// Assign shard
uint8_t zkprologml_assign_shard(
    uint64_t value
);

// Get ring statistics
int zkprologml_get_ring_stats(
    zkprologml_ctx_t *ctx,
    uint8_t ring_id,
    size_t *count,
    size_t *capacity
);

// Export to file
int zkprologml_export_samples(
    zkprologml_ctx_t *ctx,
    const char *filename,
    const zkprologml_sample_t *samples,
    size_t count
);

// Export proofs to file
int zkprologml_export_proofs(
    zkprologml_ctx_t *ctx,
    const char *filename,
    const zkprologml_proof_t *proofs,
    size_t count
);

// Export Prolog facts to file
int zkprologml_export_prolog(
    zkprologml_ctx_t *ctx,
    const char *filename,
    const zkprologml_fact_t *facts,
    size_t count
);

// Batch operations
typedef struct {
    zkprologml_sample_t *samples;
    zkprologml_proof_t *proofs;
    zkprologml_fact_t *facts;
    size_t count;
    size_t capacity;
} zkprologml_batch_t;

// Create batch
zkprologml_batch_t* zkprologml_batch_create(size_t capacity);

// Free batch
void zkprologml_batch_free(zkprologml_batch_t *batch);

// Process batch (samples → proofs → facts)
int zkprologml_batch_process(
    zkprologml_ctx_t *ctx,
    zkprologml_batch_t *batch
);

// Verify batch
bool zkprologml_batch_verify(
    zkprologml_ctx_t *ctx,
    const zkprologml_batch_t *batch
);

// Error handling
const char* zkprologml_get_error(zkprologml_ctx_t *ctx);

// Version info
void zkprologml_version(int *major, int *minor, int *patch);

#ifdef __cplusplus
}
#endif

#endif // LIBZKPROLOGML_H
