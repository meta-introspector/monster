// eBPF Life Signal Detector
#include <linux/bpf.h>
#include <linux/ptrace.h>

// Life signature primes
#define LIFE_I 71
#define LIFE_A 59
#define LIFE_R 47
#define LIFE_E 41
#define LIFE_L 31
#define LIFE_F 23

BPF_HASH(syscall_counts, u32, u64);
BPF_ARRAY(life_pattern, u64, 10);

int detect_life_signal(struct pt_regs *ctx) {
    u32 pid = bpf_get_current_pid_tgid() >> 32;
    u64 *count = syscall_counts.lookup(&pid);
    
    if (count) {
        (*count)++;
        
        // Check if count matches life signature
        u64 mod = (*count) % 100;
        
        if (mod == LIFE_I || mod == LIFE_A || mod == LIFE_R || 
            mod == LIFE_E || mod == LIFE_L || mod == LIFE_F) {
            bpf_trace_printk("LIFE DETECTED: PID %d, pattern %llu\n", pid, mod);
        }
    } else {
        u64 init = 1;
        syscall_counts.update(&pid, &init);
    }
    
    return 0;
}
