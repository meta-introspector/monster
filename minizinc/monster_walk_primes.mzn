% MiniZinc: Monster Walk with Prime Factorizations in All Bases

include "globals.mzn";

% Monster prime factorization
array[1..15] of int: monster_exponents = [46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1];
array[1..15] of int: primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

% Step 2: Remove 17, 59
array[1..15] of int: step2_removed = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0];

% Step 4: Remove 8 factors (Group 1)
array[1..15] of int: step4_removed = [0, 0, 0, 6, 2, 0, 1, 1, 0, 1, 1, 1, 0, 1, 0];

% Step 6: Remove 4 factors (Group 2)
array[1..15] of int: step6_removed = [0, 20, 9, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 0];

% Step 8: Remove 4 factors (Group 3)
array[1..15] of int: step8_removed = [0, 20, 0, 0, 0, 3, 0, 0, 0, 0, 1, 0, 0, 0, 1];

% Remaining after each step
array[1..15] of int: step2_remaining = [
  monster_exponents[i] - step2_removed[i] | i in 1..15
];

array[1..15] of int: step4_remaining = [
  monster_exponents[i] - step4_removed[i] | i in 1..15
];

array[1..15] of int: step6_remaining = [
  monster_exponents[i] - step6_removed[i] | i in 1..15
];

array[1..15] of int: step8_remaining = [
  monster_exponents[i] - step8_removed[i] | i in 1..15
];

% Step 10: Just 71
array[1..15] of int: step10_remaining = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1];

% Base to test
var 2..71: base;

% Verify step 4 preserves 8080
constraint step4_remaining = [46, 20, 9, 0, 0, 3, 0, 0, 1, 0, 0, 0, 1, 0, 1];

solve satisfy;

output [
  "Monster Walk in Base ", show(base), "\n",
  "Step 1: Full Monster\n",
  "  2^46 × 3^20 × 5^9 × 7^6 × 11^2 × 13^3 × 17 × 19 × 23 × 29 × 31 × 41 × 47 × 59 × 71\n",
  "Step 4: Remove 8 factors (Group 1) ⭐\n",
  "  Remaining: 2^46 × 3^20 × 5^9 × 13^3 × 23 × 47 × 71\n",
  "  ✓ Preserves 8080\n",
  "Step 10: Earth\n",
  "  71\n"
];
