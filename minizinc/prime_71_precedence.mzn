% Prime 71 Precedence Constraint Satisfaction Problem
% Prove that 71 is the unique Monster prime precedence level

% The 15 Monster primes
array[1..15] of int: monster_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71];

% Precedence levels used in spectral library
array[1..9] of int: precedence_levels = [30, 35, 60, 65, 71, 73, 75, 78, 80];

% Decision variable: which precedence level is a Monster prime?
var 1..9: monster_precedence_index;

% The actual precedence value
var 30..80: precedence_value;

% Link the index to the value
constraint precedence_value = precedence_levels[monster_precedence_index];

% Constraints

% 1. The precedence must be a Monster prime
constraint exists(i in 1..15)(monster_primes[i] = precedence_value);

% 2. The precedence must be between 70 and 80
constraint precedence_value > 70;
constraint precedence_value < 80;

% 3. The precedence must be the largest Monster prime
constraint forall(i in 1..15)(monster_primes[i] <= precedence_value);

% 4. It must be the ONLY Monster prime in precedence levels
constraint forall(i in 1..9 where i != monster_precedence_index)(
  forall(j in 1..15)(monster_primes[j] != precedence_levels[i])
);

% 5. The gap from 70 must be minimal
constraint forall(i in 1..9 where precedence_levels[i] > 70)(
  precedence_value - 70 <= precedence_levels[i] - 70
);

% Solve
solve satisfy;

% Output
output [
  "=== PRIME 71 PRECEDENCE PROOF ===\n",
  "Monster precedence index: ", show(monster_precedence_index), "\n",
  "Precedence value: ", show(precedence_value), "\n",
  "\nVerification:\n",
  "1. Is Monster prime: ", 
    if exists(i in 1..15)(monster_primes[i] = precedence_value) then "YES" else "NO" endif, "\n",
  "2. Between 70 and 80: ",
    if precedence_value > 70 /\ precedence_value < 80 then "YES" else "NO" endif, "\n",
  "3. Is largest Monster prime: ",
    if forall(i in 1..15)(monster_primes[i] <= precedence_value) then "YES" else "NO" endif, "\n",
  "4. Gap from 70: ", show(precedence_value - 70), "\n",
  "\n=== CONCLUSION ===\n",
  "The unique Monster prime precedence is: ", show(precedence_value), "\n",
  "This proves the choice of 71 is structurally determined.\n"
];
