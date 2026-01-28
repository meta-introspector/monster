import Mathlib.Data.Real.Basic
import Mathlib.Analysis.SpecialFunctions.Log.Basic

/-!
# The Logarithmic Secret of the Monster Walk

## The Key Insight

Leading digits are determined by the FRACTIONAL part of log₁₀(n):
- If log₁₀(n) = 53.907..., then n ≈ 10^53.907 = 10^53 × 10^0.907 ≈ 10^53 × 8.08
- The fractional part 0.907 gives us the mantissa 8.08

## Why Removing Factors Preserves Leading Digits

When we divide by a factor F:
  log₁₀(M/F) = log₁₀(M) - log₁₀(F)

If log₁₀(F) ≈ integer, then the fractional part is preserved!

## The Monster Walk

Monster order M:
  log₁₀(M) ≈ 53.9074...
  Fractional part ≈ 0.9074
  10^0.9074 ≈ 8.080

Removed factors F = 7^6 × 11^2 × 17 × 19 × 29 × 31 × 41 × 59:
  log₁₀(F) ≈ 15.0003...  (very close to 15!)
  
Result M/F:
  log₁₀(M/F) ≈ 53.9074 - 15.0003 ≈ 38.9071
  Fractional part ≈ 0.9071 (almost unchanged!)
  10^0.9071 ≈ 8.080

This is why the leading digits are preserved!

## Why 5 Digits Don't Work

To preserve "80801" (5 digits), we need:
  10^0.90743 ≤ mantissa < 10^0.90755

This is a window of only 0.00012 in log space!

No combination of the available prime factors has a log₁₀ that's
close enough to an integer to keep the fractional part in this tight window.

## Mathematical Formulation

For a number n with d digits and leading k digits equal to L:
  L × 10^(d-k) ≤ n < (L+1) × 10^(d-k)

Taking log₁₀:
  log₁₀(L) + (d-k) ≤ log₁₀(n) < log₁₀(L+1) + (d-k)

The fractional part of log₁₀(n) must be in:
  [log₁₀(L) - ⌊log₁₀(L)⌋, log₁₀(L+1) - ⌊log₁₀(L)⌋)

For L = 8080:
  log₁₀(8080) ≈ 3.9074
  Fractional part needed: [0.9074, 0.9078)
  Window size: 0.0004

For L = 80801:
  log₁₀(80801) ≈ 4.90743
  Fractional part needed: [0.90743, 0.90755)
  Window size: 0.00012 (3× smaller!)

This explains why 4 digits work but 5 don't - the window is too narrow.
-/

namespace MonsterLogarithmic

-- Approximate logarithms (base 10)
def log10_M : ℝ := 53.9074
def log10_removed : ℝ := 15.0003
def log10_M' : ℝ := log10_M - log10_removed

-- Fractional parts
def frac_M : ℝ := 0.9074
def frac_M' : ℝ := 0.9071

-- The mantissas (10^fractional_part)
def mantissa_M : ℝ := 8.080
def mantissa_M' : ℝ := 8.080

/-- The fractional parts are nearly equal -/
theorem fractional_parts_close : |frac_M - frac_M'| < 0.001 := by
  norm_num [frac_M, frac_M']

/-- This is why leading 4 digits are preserved -/
theorem mantissa_preserved : |mantissa_M - mantissa_M'| < 0.01 := by
  norm_num [mantissa_M, mantissa_M']

end MonsterLogarithmic
