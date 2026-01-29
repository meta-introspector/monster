// Linux Kernel Module: Monster Process Sampler
// Sample all processes, apply Hecke operators, shard by 15 Monster primes

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/sched.h>
#include <linux/sched/signal.h>
#include <linux/mm.h>
#include <linux/pid.h>
#include <linux/kthread.h>
#include <linux/delay.h>
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/time.h>

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Meta-Introspector");
MODULE_DESCRIPTION("Monster Group Process Sampler with Hecke Operators");
MODULE_VERSION("1.0");

// 15 Monster primes
static const u8 MONSTER_PRIMES[15] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71
};

// Sample data structure
struct process_sample {
    pid_t pid;
    u64 timestamp;
    u64 rip;           // Instruction pointer
    u64 rsp;           // Stack pointer
    u64 rax;           // Register values
    u64 rbx;
    u64 rcx;
    u64 rdx;
    u64 mem_usage;     // Memory usage
    u64 cpu_time;      // CPU time
    u8 shard_id;       // Shard assignment (0-14)
    u8 hecke_applied;  // Which Hecke operator
};

// Ring buffer for each Monster prime
struct ring_buffer {
    struct process_sample *samples;
    size_t capacity;
    size_t head;
    size_t tail;
    size_t count;
    spinlock_t lock;
};

// 15 ring buffers (one per Monster prime)
static struct ring_buffer *rings[15];

// GPU transfer buffer
#define GPU_BUFFER_SIZE (1024 * 1024)  // 1MB
static struct process_sample *gpu_buffer;
static size_t gpu_buffer_count;
static spinlock_t gpu_lock;

// Sampling thread
static struct task_struct *sampler_thread;
static bool sampling_active = false;

// Statistics
static atomic64_t total_samples;
static atomic64_t samples_per_prime[15];

// Initialize ring buffer
static struct ring_buffer *ring_init(size_t capacity)
{
    struct ring_buffer *ring;
    
    ring = kmalloc(sizeof(*ring), GFP_KERNEL);
    if (!ring)
        return NULL;
    
    ring->samples = vmalloc(capacity * sizeof(struct process_sample));
    if (!ring->samples) {
        kfree(ring);
        return NULL;
    }
    
    ring->capacity = capacity;
    ring->head = 0;
    ring->tail = 0;
    ring->count = 0;
    spin_lock_init(&ring->lock);
    
    return ring;
}

// Push sample to ring
static int ring_push(struct ring_buffer *ring, struct process_sample *sample)
{
    unsigned long flags;
    
    spin_lock_irqsave(&ring->lock, flags);
    
    if (ring->count >= ring->capacity) {
        // Ring full, drop oldest
        ring->tail = (ring->tail + 1) % ring->capacity;
        ring->count--;
    }
    
    ring->samples[ring->head] = *sample;
    ring->head = (ring->head + 1) % ring->capacity;
    ring->count++;
    
    spin_unlock_irqrestore(&ring->lock, flags);
    
    return 0;
}

// Pop sample from ring
static int ring_pop(struct ring_buffer *ring, struct process_sample *sample)
{
    unsigned long flags;
    
    spin_lock_irqsave(&ring->lock, flags);
    
    if (ring->count == 0) {
        spin_unlock_irqrestore(&ring->lock, flags);
        return -EAGAIN;
    }
    
    *sample = ring->samples[ring->tail];
    ring->tail = (ring->tail + 1) % ring->capacity;
    ring->count--;
    
    spin_unlock_irqrestore(&ring->lock, flags);
    
    return 0;
}

// Apply Hecke operator (permutation based on prime)
static u64 apply_hecke(u64 value, u8 prime_idx)
{
    u8 prime = MONSTER_PRIMES[prime_idx];
    
    // Hecke operator: T_p(x) = (x * p) mod 71
    return (value * prime) % 71;
}

// Assign shard based on value mod 71
static u8 assign_shard(u64 value)
{
    u8 mod71 = value % 71;
    
    // Map to Monster prime index
    for (int i = 0; i < 15; i++) {
        if (mod71 % MONSTER_PRIMES[i] == 0)
            return i;
    }
    
    // Default: largest prime
    return 14;  // 71
}

// Sample a single process
static void sample_process(struct task_struct *task)
{
    struct process_sample sample;
    struct pt_regs *regs;
    u64 combined_value;
    
    // Get process info
    sample.pid = task->pid;
    sample.timestamp = ktime_get_ns();
    
    // Get registers (if available)
    regs = task_pt_regs(task);
    if (regs) {
        sample.rip = regs->ip;
        sample.rsp = regs->sp;
        sample.rax = regs->ax;
        sample.rbx = regs->bx;
        sample.rcx = regs->cx;
        sample.rdx = regs->dx;
    } else {
        sample.rip = 0;
        sample.rsp = 0;
        sample.rax = 0;
        sample.rbx = 0;
        sample.rcx = 0;
        sample.rdx = 0;
    }
    
    // Get memory usage
    if (task->mm) {
        sample.mem_usage = get_mm_rss(task->mm) << PAGE_SHIFT;
    } else {
        sample.mem_usage = 0;
    }
    
    // Get CPU time
    sample.cpu_time = task->utime + task->stime;
    
    // Combine values for sharding
    combined_value = sample.rip ^ sample.rsp ^ sample.rax ^ 
                     sample.mem_usage ^ sample.cpu_time;
    
    // Assign shard
    sample.shard_id = assign_shard(combined_value);
    
    // Apply Hecke operator
    sample.hecke_applied = sample.shard_id;
    combined_value = apply_hecke(combined_value, sample.shard_id);
    
    // Push to appropriate ring
    ring_push(rings[sample.shard_id], &sample);
    
    // Update statistics
    atomic64_inc(&total_samples);
    atomic64_inc(&samples_per_prime[sample.shard_id]);
}

// Sample all processes
static void sample_all_processes(void)
{
    struct task_struct *task;
    
    rcu_read_lock();
    for_each_process(task) {
        if (task->mm) {  // Only user-space processes
            sample_process(task);
        }
    }
    rcu_read_unlock();
}

// Transfer samples to GPU buffer
static size_t transfer_to_gpu(void)
{
    unsigned long flags;
    size_t transferred = 0;
    struct process_sample sample;
    
    spin_lock_irqsave(&gpu_lock, flags);
    
    // Collect from all rings
    for (int i = 0; i < 15; i++) {
        while (transferred < GPU_BUFFER_SIZE && 
               ring_pop(rings[i], &sample) == 0) {
            gpu_buffer[transferred++] = sample;
        }
    }
    
    gpu_buffer_count = transferred;
    
    spin_unlock_irqrestore(&gpu_lock, flags);
    
    return transferred;
}

// Sampling thread function
static int sampler_thread_fn(void *data)
{
    size_t transferred;
    
    pr_info("Monster sampler thread started\n");
    
    while (!kthread_should_stop() && sampling_active) {
        // Sample all processes
        sample_all_processes();
        
        // Transfer to GPU buffer
        transferred = transfer_to_gpu();
        
        if (transferred > 0) {
            pr_debug("Transferred %zu samples to GPU buffer\n", transferred);
        }
        
        // Sleep for 10ms (100 Hz sampling)
        msleep(10);
    }
    
    pr_info("Monster sampler thread stopped\n");
    return 0;
}

// Print statistics
static void print_stats(void)
{
    pr_info("=== Monster Process Sampler Statistics ===\n");
    pr_info("Total samples: %lld\n", atomic64_read(&total_samples));
    
    for (int i = 0; i < 15; i++) {
        pr_info("Prime %2d (shard %2d): %lld samples, ring: %zu/%zu\n",
                MONSTER_PRIMES[i], i,
                atomic64_read(&samples_per_prime[i]),
                rings[i]->count, rings[i]->capacity);
    }
}

// Module initialization
static int __init monster_sampler_init(void)
{
    int i;
    
    pr_info("Monster Process Sampler initializing...\n");
    
    // Initialize statistics
    atomic64_set(&total_samples, 0);
    for (i = 0; i < 15; i++) {
        atomic64_set(&samples_per_prime[i], 0);
    }
    
    // Initialize ring buffers (10,000 samples each)
    for (i = 0; i < 15; i++) {
        rings[i] = ring_init(10000);
        if (!rings[i]) {
            pr_err("Failed to initialize ring %d\n", i);
            goto cleanup_rings;
        }
    }
    
    // Initialize GPU buffer
    gpu_buffer = vmalloc(GPU_BUFFER_SIZE * sizeof(struct process_sample));
    if (!gpu_buffer) {
        pr_err("Failed to allocate GPU buffer\n");
        goto cleanup_rings;
    }
    gpu_buffer_count = 0;
    spin_lock_init(&gpu_lock);
    
    // Start sampling thread
    sampling_active = true;
    sampler_thread = kthread_run(sampler_thread_fn, NULL, "monster_sampler");
    if (IS_ERR(sampler_thread)) {
        pr_err("Failed to create sampler thread\n");
        goto cleanup_gpu;
    }
    
    pr_info("Monster Process Sampler initialized successfully\n");
    pr_info("Sampling at 100 Hz, 15 rings, GPU buffer: %d samples\n",
            GPU_BUFFER_SIZE);
    
    return 0;

cleanup_gpu:
    vfree(gpu_buffer);
    
cleanup_rings:
    for (i = 0; i < 15; i++) {
        if (rings[i]) {
            vfree(rings[i]->samples);
            kfree(rings[i]);
        }
    }
    
    return -ENOMEM;
}

// Module cleanup
static void __exit monster_sampler_exit(void)
{
    int i;
    
    pr_info("Monster Process Sampler shutting down...\n");
    
    // Stop sampling thread
    sampling_active = false;
    if (sampler_thread) {
        kthread_stop(sampler_thread);
    }
    
    // Print final statistics
    print_stats();
    
    // Cleanup GPU buffer
    vfree(gpu_buffer);
    
    // Cleanup ring buffers
    for (i = 0; i < 15; i++) {
        if (rings[i]) {
            vfree(rings[i]->samples);
            kfree(rings[i]);
        }
    }
    
    pr_info("Monster Process Sampler shutdown complete\n");
}

module_init(monster_sampler_init);
module_exit(monster_sampler_exit);
