// zkPerf eBPF Kernel Plugin
#include <linux/bpf.h>
#include <linux/ptrace.h>
#include <linux/perf_event.h>

// Monster primes for scoring
#define PRIME_71 71
#define PRIME_59 59
#define PRIME_47 47
#define PRIME_7  7

// Per-process performance counters
BPF_HASH(cpu_cycles, u32, u64);
BPF_HASH(cache_misses, u32, u64);
BPF_HASH(branch_mispredicts, u32, u64);
BPF_HASH(slowness_scores, u32, u64);

// Trace CPU cycles
int trace_cpu_cycles(struct bpf_perf_event_data *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = cpu_cycles.lookup(&pid);
    
    if (count) {
        (*count)++;
    } else {
        u64 init = 1;
        cpu_cycles.update(&pid, &init);
    }
    
    return 0;
}

// Trace cache misses
int trace_cache_miss(struct bpf_perf_event_data *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = cache_misses.lookup(&pid);
    
    if (count) {
        (*count)++;
    } else {
        u64 init = 1;
        cache_misses.update(&pid, &init);
    }
    
    return 0;
}

// Trace branch mispredictions
int trace_branch_mispredict(struct bpf_perf_event_data *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = branch_mispredicts.lookup(&pid);
    
    if (count) {
        (*count)++;
    } else {
        u64 init = 1;
        branch_mispredicts.update(&pid, &init);
    }
    
    // Compute slowness score using Monster primes
    u64 *cycles = cpu_cycles.lookup(&pid);
    u64 *misses = cache_misses.lookup(&pid);
    u64 *mispredicts = branch_mispredicts.lookup(&pid);
    
    if (cycles && misses && mispredicts) {
        u64 score = (*cycles / 1000) + (*misses * PRIME_7) + (*mispredicts * PRIME_71);
        slowness_scores.update(&pid, &score);
        
        // If score > threshold, trigger optimization
        if (score > 10000) {
            bpf_trace_printk("Slow process detected: PID %d, score %llu\n", pid, score);
        }
    }
    
    return 0;
}
