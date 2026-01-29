// C: Monster Walk with Prime Factorizations in All Bases

#include <stdio.h>
#include <stdint.h>
#include <gmp.h>

#define NUM_PRIMES 15

// Prime factorization structure
typedef struct {
    uint32_t primes[NUM_PRIMES];
    uint32_t exponents[NUM_PRIMES];
} PrimeFactorization;

// Monster primes
const uint32_t MONSTER_PRIMES[NUM_PRIMES] = {
    2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71
};

// Monster factorization
PrimeFactorization monster_factorization() {
    PrimeFactorization f;
    for (int i = 0; i < NUM_PRIMES; i++) {
        f.primes[i] = MONSTER_PRIMES[i];
    }
    f.exponents[0] = 46;  // 2^46
    f.exponents[1] = 20;  // 3^20
    f.exponents[2] = 9;   // 5^9
    f.exponents[3] = 6;   // 7^6
    f.exponents[4] = 2;   // 11^2
    f.exponents[5] = 3;   // 13^3
    for (int i = 6; i < NUM_PRIMES; i++) {
        f.exponents[i] = 1;
    }
    return f;
}

// Step 4: Remove 8 factors
PrimeFactorization step4_remaining() {
    PrimeFactorization f = monster_factorization();
    f.exponents[3] = 0;   // Remove 7^6
    f.exponents[4] = 0;   // Remove 11^2
    f.exponents[6] = 0;   // Remove 17
    f.exponents[7] = 0;   // Remove 19
    f.exponents[9] = 0;   // Remove 29
    f.exponents[10] = 0;  // Remove 31
    f.exponents[11] = 0;  // Remove 41
    f.exponents[13] = 0;  // Remove 59
    return f;
}

// Compute value from factorization
void compute_value(PrimeFactorization *f, mpz_t result) {
    mpz_set_ui(result, 1);
    mpz_t temp;
    mpz_init(temp);
    
    for (int i = 0; i < NUM_PRIMES; i++) {
        if (f->exponents[i] > 0) {
            mpz_ui_pow_ui(temp, f->primes[i], f->exponents[i]);
            mpz_mul(result, result, temp);
        }
    }
    
    mpz_clear(temp);
}

// Convert to base
void to_base(mpz_t n, uint32_t base, uint8_t *digits, size_t *len) {
    mpz_t temp;
    mpz_init_set(temp, n);
    
    *len = 0;
    while (mpz_cmp_ui(temp, 0) > 0) {
        digits[*len] = mpz_fdiv_ui(temp, base);
        mpz_fdiv_q_ui(temp, temp, base);
        (*len)++;
    }
    
    // Reverse
    for (size_t i = 0; i < *len / 2; i++) {
        uint8_t tmp = digits[i];
        digits[i] = digits[*len - 1 - i];
        digits[*len - 1 - i] = tmp;
    }
    
    mpz_clear(temp);
}

// Print factorization
void print_factorization(PrimeFactorization *f) {
    int first = 1;
    for (int i = 0; i < NUM_PRIMES; i++) {
        if (f->exponents[i] > 0) {
            if (!first) printf(" × ");
            if (f->exponents[i] == 1) {
                printf("%u", f->primes[i]);
            } else {
                printf("%u^%u", f->primes[i], f->exponents[i]);
            }
            first = 0;
        }
    }
}

int main() {
    printf("🔢 Monster Walk with Primes - All Bases (C)\n");
    printf("============================================\n\n");
    
    mpz_t monster_val, step4_val;
    mpz_init(monster_val);
    mpz_init(step4_val);
    
    // Step 1: Full Monster
    PrimeFactorization monster = monster_factorization();
    compute_value(&monster, monster_val);
    
    printf("Step 1: Full Monster\n");
    printf("  Primes: ");
    print_factorization(&monster);
    printf("\n");
    printf("  Decimal: ");
    mpz_out_str(stdout, 10, monster_val);
    printf("\n");
    printf("  Hex: 0x");
    mpz_out_str(stdout, 16, monster_val);
    printf("\n\n");
    
    // Step 4: Remove 8 factors
    PrimeFactorization step4 = step4_remaining();
    compute_value(&step4, step4_val);
    
    printf("Step 4: Remove 8 factors (Group 1) ⭐\n");
    printf("  Remaining: ");
    print_factorization(&step4);
    printf("\n");
    printf("  Decimal: ");
    mpz_out_str(stdout, 10, step4_val);
    printf("\n");
    printf("  Hex: 0x");
    mpz_out_str(stdout, 16, step4_val);
    printf("\n\n");
    
    // Test in all bases
    printf("Testing all bases 2-71:\n");
    for (uint32_t base = 2; base <= 71; base++) {
        uint8_t digits[256];
        size_t len;
        to_base(step4_val, base, digits, &len);
        
        if (base == 2 || base == 10 || base == 16 || base == 71) {
            printf("  Base %2u: %zu digits\n", base, len);
        }
    }
    
    printf("\n✅ All bases computed\n");
    
    mpz_clear(monster_val);
    mpz_clear(step4_val);
    
    return 0;
}
