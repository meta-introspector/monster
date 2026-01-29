#include <stdio.h>
#include <stdint.h>

uint64_t monster_primes[] = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71};
uint32_t weights[] = {46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1};

int count_monster_factors(uint64_t n) {
    int count = 0;
    for (int i = 0; i < 15; i++) {
        if (n % monster_primes[i] == 0) {
            count++;
        }
    }
    return count;
}

double check_resonance(uint64_t n) {
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

int main() {
    printf("🎯 Testing Monster algorithm (1M iterations)\n\n");
    
    uint64_t total_score = 0;
    double total_resonance = 0.0;
    
    // Test 1 million values
    for (uint64_t n = 1; n < 1000000; n++) {
        int score = count_monster_factors(n);
        double resonance = check_resonance(n);
        total_score += score;
        total_resonance += resonance;
    }
    
    printf("Total score: %lu\n", total_score);
    printf("Average resonance: %.6f\n", total_resonance / 1000000.0);
    
    return 0;
}
