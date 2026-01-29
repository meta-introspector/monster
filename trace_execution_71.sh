#!/usr/bin/env bash
# Trace execution along prime 71 - see what's computable

set -euo pipefail

echo "🎯 Tracing Execution Along Prime 71"
echo "===================================="
echo ""

# Create tracer
cat > trace_71.c << 'C'
#include <stdio.h>
#include <stdint.h>
#include <time.h>

#define PRIME_71 71

// Monster primes
uint64_t monster_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71};
uint32_t weights[] = {46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1};

// Trace point structure
typedef struct {
    uint64_t value;
    int step;
    int divisible_by_71;
    int monster_score;
    double resonance;
} TracePoint;

int is_divisible_by_71(uint64_t n) {
    return (n % PRIME_71) == 0;
}

int count_monster_factors(uint64_t n) {
    int count = 0;
    for (int i = 0; i < 15; i++) {
        if (n % monster_primes[i] == 0) count++;
    }
    return count;
}

double compute_resonance(uint64_t n) {
    uint32_t weighted_sum = 0;
    uint32_t total_weight = 0;
    
    for (int i = 0; i < 15; i++) {
        total_weight += weights[i];
        if (n % monster_primes[i] == 0) {
            weighted_sum += weights[i];
        }
    }
    
    return (double)weighted_sum / total_weight;
}

// Graded multiplication along 71
uint64_t graded_mul_71(uint64_t a, uint64_t b) {
    // Simulate graded multiplication at precedence 71
    uint64_t result = a * b;
    
    // Extract Monster prime factors
    uint64_t monster_part = 1;
    for (int i = 0; i < 15; i++) {
        if (result % monster_primes[i] == 0) {
            monster_part *= monster_primes[i];
        }
    }
    
    return monster_part;
}

int main() {
    printf("🎯 Execution Trace Along Prime 71\n\n");
    
    // Test values that interact with 71
    uint64_t test_values[] = {
        71,           // Prime 71 itself
        71 * 2,       // 71 with smallest Monster prime
        71 * 3,       // 71 with second Monster prime
        71 * 71,      // 71 squared
        2 * 3 * 71,   // 71 with first two primes
        2 * 3 * 5 * 7 * 11 * 71,  // 71 with first 5 primes
    };
    
    printf("Step-by-step trace:\n");
    printf("%-6s %-12s %-8s %-8s %-10s\n", "Step", "Value", "Div71?", "Score", "Resonance");
    printf("%-6s %-12s %-8s %-8s %-10s\n", "----", "-----", "------", "-----", "---------");
    
    for (int i = 0; i < 6; i++) {
        uint64_t val = test_values[i];
        int div71 = is_divisible_by_71(val);
        int score = count_monster_factors(val);
        double res = compute_resonance(val);
        
        printf("%-6d %-12lu %-8s %-8d %-10.4f\n", 
               i, val, div71 ? "YES" : "NO", score, res);
    }
    
    printf("\n🔬 Graded multiplication along 71:\n\n");
    
    // Test graded multiplication
    uint64_t a = 71;
    uint64_t b = 2 * 3 * 5;  // 30
    uint64_t result = graded_mul_71(a, b);
    
    printf("  a = %lu (prime 71)\n", a);
    printf("  b = %lu (2×3×5)\n", b);
    printf("  a ** b = %lu (graded mul)\n", result);
    printf("  Monster factors extracted: ");
    for (int i = 0; i < 15; i++) {
        if (result % monster_primes[i] == 0) {
            printf("%lu ", monster_primes[i]);
        }
    }
    printf("\n");
    
    printf("\n📊 Resonance along 71:\n\n");
    
    // Scan values around multiples of 71
    int high_resonance_count = 0;
    for (uint64_t n = 71; n <= 71 * 100; n += 71) {
        double res = compute_resonance(n);
        if (res > 0.5) {
            high_resonance_count++;
            if (high_resonance_count <= 5) {
                printf("  %lu: resonance %.4f (score %d)\n", 
                       n, res, count_monster_factors(n));
            }
        }
    }
    printf("  ... (%d total with resonance > 0.5)\n", high_resonance_count);
    
    return 0;
}
C

gcc -O2 -o trace_71 trace_71.c
./trace_71

echo ""
echo "✅ Trace complete!"
