-- LMFDB verify - Translated from verification.py
-- Monster Shard - Prime resonance distribution

import Mathlib.Data.Nat.Prime
import Mathlib.NumberTheory.NumberField.Basic

namespace LMFDB.verify

def accumulate_failures : Unit := sorry

structure TooManyFailures(AssertionError) where
  -- TODO: Translate fields
  sorry : Unit

structure speed_decorator() where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def __call__ : Unit := sorry

structure per_row(speed_decorator) where
  -- TODO: Translate fields
  sorry : Unit

structure one_query(speed_decorator) where
  -- TODO: Translate fields
  sorry : Unit

structure slow(per_row) where
  -- TODO: Translate fields
  sorry : Unit

structure fast(per_row) where
  -- TODO: Translate fields
  sorry : Unit

structure overall(one_query) where
  -- TODO: Translate fields
  sorry : Unit

structure overall_long(one_query) where
  -- TODO: Translate fields
  sorry : Unit

structure TableChecker() where
  -- TODO: Translate fields
  sorry : Unit

def speedtype : Unit := sorry

def all_types : Unit := sorry

def get_checks_count : Unit := sorry

def get_checks : Unit := sorry

def get_check : Unit := sorry

def get_total : Unit := sorry

def get_iter : Unit := sorry

def get_progress_interval : Unit := sorry

def _report_error : Unit := sorry

def _run_check : Unit := sorry

def run_check : Unit := sorry

def run : Unit := sorry

def _test_equality : Unit := sorry

def _make_sql : Unit := sorry

def _make_join : Unit := sorry

def _run_query : Unit := sorry

def _run_crosstable : Unit := sorry

def check_count : Unit := sorry

def check_eq : Unit := sorry

def _check_arith : Unit := sorry

def check_sum : Unit := sorry

def check_product : Unit := sorry

def check_array_sum : Unit := sorry

def check_array_product : Unit := sorry

def check_array_dotproduct : Unit := sorry

def check_divisible : Unit := sorry

def check_non_divisible : Unit := sorry

def check_values : Unit := sorry

def check_non_null : Unit := sorry

def check_null : Unit := sorry

def check_iff : Unit := sorry

def check_array_len_gte_constant : Unit := sorry

def check_array_len_eq_constant : Unit := sorry

def check_array_len_col : Unit := sorry

def check_array_bound : Unit := sorry

def check_array_concatenation : Unit := sorry

def check_string_concatenation : Unit := sorry

def check_string_startswith : Unit := sorry

def check_sorted : Unit := sorry

def check_crosstable : Unit := sorry

def check_crosstable_count : Unit := sorry

def check_crosstable_sum : Unit := sorry

def check_crosstable_dotprod : Unit := sorry

def check_crosstable_aggregate : Unit := sorry

def check_letter_code : Unit := sorry

def check_label : Unit := sorry

def check_uniqueness_constraints : Unit := sorry

end LMFDB.verify
