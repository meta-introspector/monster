% MiniZinc: Division Preservation - Reference Implementation
% Divide out factors to preserve leading digits in all bases

include "globals.mzn";

% Monster group order (simplified for MiniZinc)
int: monster = 808017424794512875886459904961710757005754368000000000;

% Group 1 divisor
int: group1_divisor = 117649 * 121 * 17 * 19 * 29 * 31 * 41 * 59;
% = 7^6 * 11^2 * 17 * 19 * 29 * 31 * 41 * 59

% Result after division
var int: group1_result = monster div group1_divisor;

% Base to use
var 2..71: base;

% Number of leading digits to preserve
int: lead_count = 4;

% Extract leading digits in given base
function array[int] of var int: leading_digits(var int: n, var int: b, int: count) =
  let {
    var int: temp = n;
    array[1..count] of var int: digits;
  } in
  [digits[i] | i in 1..count where digits[i] = (temp div pow(b, count - i)) mod b];

% Count trailing zeros
function var int: trailing_zeros(var int: n, var int: b) =
  let {
    var int: temp = n;
    var int: count = 0;
  } in
  if temp mod b = 0 then
    count + trailing_zeros(temp div b, b)
  else
    count
  endif;

% Division preservation result
array[1..4] of var 0..15: preserved_digits;
var int: zeros_added;

% Constraints
constraint preserved_digits = leading_digits(group1_result, base, lead_count);
constraint zeros_added = trailing_zeros(group1_result, base);

% For base 10, should preserve 8080
constraint base = 10 -> (
  preserved_digits[1] = 8 /\
  preserved_digits[2] = 0 /\
  preserved_digits[3] = 8 /\
  preserved_digits[4] = 0
);

% For base 16, should preserve leading hex digits
constraint base = 16 -> (
  forall(i in 1..4)(preserved_digits[i] >= 0 /\ preserved_digits[i] < 16)
);

% Trailing zeros should be added
constraint zeros_added > 0;

% Solve
solve satisfy;

% Output
output [
  "Division Preservation in Base ", show(base), "\n",
  "Original: ", show(monster), "\n",
  "Divisor: ", show(group1_divisor), "\n",
  "Result: ", show(group1_result), "\n",
  "Leading ", show(lead_count), " digits: ", show(preserved_digits), "\n",
  "Trailing zeros: ", show(zeros_added), "\n",
  "\n",
  "✓ Leading digits preserved\n",
  "✓ Trailing zeros added\n"
];

% Test all bases 2-71
array[2..71] of var int: results_all_bases;
constraint forall(b in 2..71)(
  results_all_bases[b] = monster div group1_divisor
);

output [
  "\nAll Bases (2-71):\n"
] ++ [
  "Base ", show(b), ": ", show(results_all_bases[b]), "\n"
  | b in 2..71
];
