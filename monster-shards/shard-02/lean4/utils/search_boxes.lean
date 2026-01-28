-- LMFDB utils - Translated from search_boxes.py
-- Monster Shard - Prime resonance distribution

import Mathlib.Data.Nat.Prime
import Mathlib.NumberTheory.NumberField.Basic

namespace LMFDB.utils

structure TdElt() where
  -- TODO: Translate fields
  sorry : Unit

def _add_class : Unit := sorry

def _wrap : Unit := sorry

def td : Unit := sorry

structure Spacer(TdElt) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def input_html : Unit := sorry

def label_html : Unit := sorry

def example_html : Unit := sorry

def has_label : Unit := sorry

structure RowSpacer(Spacer) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def tr : Unit := sorry

def html : Unit := sorry

structure BasicSpacer(Spacer) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def input_html : Unit := sorry

structure CheckboxSpacer(Spacer) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def html : Unit := sorry

structure SearchBox(TdElt) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def _label : Unit := sorry

def has_label : Unit := sorry

def label_html : Unit := sorry

def input_html : Unit := sorry

def example_html : Unit := sorry

structure TextBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def _input : Unit := sorry

structure SelectBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def _input : Unit := sorry

structure NoEg(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def example_html : Unit := sorry

structure TextBoxNoEg(NoEg, where
  -- TODO: Translate fields
  sorry : Unit

structure SelectBoxNoEg(NoEg, where
  -- TODO: Translate fields
  sorry : Unit

structure HiddenBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def _input : Unit := sorry

structure CheckBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def _input : Unit := sorry

structure SneakyBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

structure SneakyTextBox(TextBox, where
  -- TODO: Translate fields
  sorry : Unit

structure SneakySelectBox(SelectBox, where
  -- TODO: Translate fields
  sorry : Unit

structure SkipBox(TextBox) where
  -- TODO: Translate fields
  sorry : Unit

def _input : Unit := sorry

def _label : Unit := sorry

structure TextBoxWithSelect(TextBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def label_html : Unit := sorry

structure DoubleSelectBox(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def _input : Unit := sorry

structure ExcludeOnlyBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure YesNoBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure YesNoMaybeBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure ParityBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure ParityMod(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure SubsetBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure SubsetNoExcludeBox(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

structure CountBox(TextBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

structure ColumnController(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def _label : Unit := sorry

def _input : Unit := sorry

structure SortController(SelectBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

structure SearchButton(SearchBox) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def td : Unit := sorry

def _input : Unit := sorry

structure SearchButtonWithSelect(SearchButton) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def label_html : Unit := sorry

structure SearchArray(UniqueRepresentation) where
  -- TODO: Translate fields
  sorry : Unit

def sort_order : Unit := sorry

def _search_again : Unit := sorry

def search_types : Unit := sorry

def hidden : Unit := sorry

def main_array : Unit := sorry

def _print_table : Unit := sorry

def _st : Unit := sorry

def dynstats_array : Unit := sorry

def hidden_inputs : Unit := sorry

def main_table : Unit := sorry

def has_advanced_inputs : Unit := sorry

def _buttons : Unit := sorry

def buttons : Unit := sorry

def html : Unit := sorry

def jump_box : Unit := sorry

structure EmbeddedSearchArray(SearchArray) where
  -- TODO: Translate fields
  sorry : Unit

def __init__ : Unit := sorry

def buttons : Unit := sorry

end LMFDB.utils
