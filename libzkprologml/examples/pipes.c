// Example: Bidirectional pipes and coordination
// Producer-consumer pattern with kernel coordination

#include "zkprologml.h"
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>

// Producer thread: reads from kernel, writes to pipe
void* producer_thread(void *arg) {
    zkprologml_ctx_t *ctx = (zkprologml_ctx_t*)arg;
    
    // Open pipe as producer for ring 0 (prime 2)
    zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, 0, true);
    if (!pipe) {
        fprintf(stderr, "Producer: Failed to open pipe\n");
        return NULL;
    }
    
    printf("Producer: Started on ring 0\n");
    
    for (int i = 0; i < 100; i++) {
        zkprologml_sample_t sample;
        
        // Read from kernel
        size_t count;
        if (zkprologml_read_samples(ctx, &sample, 1, &count) == 0 && count > 0) {
            // Write to pipe (blocks if full)
            if (zkprologml_pipe_write(pipe, &sample, 1000) == 0) {
                printf("Producer: Wrote sample %d (pid=%d)\n", i, sample.pid);
            }
        }
        
        usleep(10000);  // 10ms
    }
    
    zkprologml_pipe_close(pipe);
    printf("Producer: Finished\n");
    
    return NULL;
}

// Consumer thread: reads from pipe, processes
void* consumer_thread(void *arg) {
    zkprologml_ctx_t *ctx = (zkprologml_ctx_t*)arg;
    
    // Open pipe as consumer for ring 0
    zkprologml_pipe_t *pipe = zkprologml_pipe_open(ctx, 0, false);
    if (!pipe) {
        fprintf(stderr, "Consumer: Failed to open pipe\n");
        return NULL;
    }
    
    printf("Consumer: Started on ring 0\n");
    
    for (int i = 0; i < 100; i++) {
        zkprologml_sample_t sample;
        
        // Read from pipe (blocks if empty)
        if (zkprologml_pipe_read(pipe, &sample, 1000) == 0) {
            printf("Consumer: Read sample %d (pid=%d, shard=%d)\n",
                   i, sample.pid, sample.shard_id);
            
            // Process sample
            zkprologml_proof_t proof;
            zkprologml_generate_proof(ctx, &sample, &proof);
            
            if (zkprologml_verify_proof(ctx, &proof)) {
                printf("Consumer: Proof verified!\n");
            }
        }
    }
    
    zkprologml_pipe_close(pipe);
    printf("Consumer: Finished\n");
    
    return NULL;
}

// Coordinator thread: coordinates apps on same shard
void* coordinator_thread(void *arg) {
    zkprologml_ctx_t *ctx = (zkprologml_ctx_t*)arg;
    
    printf("Coordinator: Started\n");
    
    for (int i = 0; i < 10; i++) {
        // Coordinate all apps on shard 0
        int count = zkprologml_coordinate_shard(ctx, 0);
        printf("Coordinator: Coordinated %d apps on shard 0\n", count);
        
        sleep(1);
    }
    
    printf("Coordinator: Finished\n");
    
    return NULL;
}

// Waiter thread: waits for coordination signal
void* waiter_thread(void *arg) {
    zkprologml_ctx_t *ctx = (zkprologml_ctx_t*)arg;
    
    // Create wait state for shard 0
    zkprologml_wait_t *wait = zkprologml_wait_create(ctx, 0);
    if (!wait) {
        fprintf(stderr, "Waiter: Failed to create wait state\n");
        return NULL;
    }
    
    printf("Waiter: Waiting for coordination on shard 0...\n");
    
    // Wait for coordination (5 second timeout)
    if (zkprologml_wait_for_coord(wait, 5000) == 0) {
        printf("Waiter: Coordination received!\n");
        
        // Signal ready
        zkprologml_signal_ready(wait);
    } else {
        printf("Waiter: Timeout\n");
    }
    
    zkprologml_wait_free(wait);
    
    return NULL;
}

int main(void) {
    printf("🔄 Bidirectional Pipes and Coordination Example\n");
    printf("================================================\n\n");
    
    // Initialize
    zkprologml_ctx_t *ctx = zkprologml_init();
    if (!ctx) {
        fprintf(stderr, "Failed to initialize\n");
        return 1;
    }
    
    // Create threads
    pthread_t prod_thread, cons_thread, coord_thread, wait_thread;
    
    printf("Starting threads...\n\n");
    
    pthread_create(&prod_thread, NULL, producer_thread, ctx);
    pthread_create(&cons_thread, NULL, consumer_thread, ctx);
    pthread_create(&coord_thread, NULL, coordinator_thread, ctx);
    pthread_create(&wait_thread, NULL, waiter_thread, ctx);
    
    // Wait for threads
    pthread_join(prod_thread, NULL);
    pthread_join(cons_thread, NULL);
    pthread_join(coord_thread, NULL);
    pthread_join(wait_thread, NULL);
    
    printf("\n✅ All threads complete!\n");
    
    // Cleanup
    zkprologml_free(ctx);
    
    return 0;
}
