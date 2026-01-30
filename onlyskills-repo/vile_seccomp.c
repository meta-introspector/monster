// Seccomp filter for vile code - minimal syscall whitelist
#include <stdio.h>
#include <seccomp.h>
#include <unistd.h>

int apply_vile_code_filter() {
    scmp_filter_ctx ctx;
    
    // Default: KILL process on any syscall
    ctx = seccomp_init(SCMP_ACT_KILL);
    if (ctx == NULL) return -1;
    
    // ONLY allow these syscalls:
    
    // Read (stdin only)
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(read), 1,
        SCMP_A0(SCMP_CMP_EQ, STDIN_FILENO));
    
    // Write (stdout/stderr only)
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
        SCMP_A0(SCMP_CMP_EQ, STDOUT_FILENO));
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(write), 1,
        SCMP_A0(SCMP_CMP_EQ, STDERR_FILENO));
    
    // Exit
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit), 0);
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(exit_group), 0);
    
    // Minimal memory (no exec)
    seccomp_rule_add(ctx, SCMP_ACT_ALLOW, SCMP_SYS(brk), 0);
    
    // Load filter
    if (seccomp_load(ctx) < 0) {
        seccomp_release(ctx);
        return -1;
    }
    
    seccomp_release(ctx);
    return 0;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <vile_code_binary>\n", argv[0]);
        return 1;
    }
    
    printf("Applying vile code seccomp filter...\n");
    
    if (apply_vile_code_filter() < 0) {
        fprintf(stderr, "Failed to apply seccomp filter\n");
        return 1;
    }
    
    printf("Filter applied. Executing vile code (contained)...\n");
    
    // Execute vile code with filter active
    execv(argv[1], &argv[1]);
    
    // Should never reach here
    perror("execv failed");
    return 1;
}
