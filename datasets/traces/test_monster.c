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
    uint64_t test_values[] = {
        12345,
        2310,    // 2*3*5*7*11
        1024,    // 2^10
        4189,    // 59*71
        30030,   // 2*3*5*7*11*13
    };
    
    printf("🎯 Testing Monster algorithm\n\n");
    
    for (int i = 0; i < 5; i++) {
        uint64_t val = test_values[i];
        int score = count_monster_factors(val);
        double resonance = check_resonance(val);
        printf("Value: %lu, Score: %d, Resonance: %.4f\n", val, score, resonance);
    }
    
    return 0;
}
