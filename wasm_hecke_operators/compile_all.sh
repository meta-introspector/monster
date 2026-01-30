#!/bin/bash
# Compile all WAT to WASM

for wat in wasm_hecke_operators/*.wat; do
  wasm="${wat%.wat}.wasm"
  echo "Compiling $wat → $wasm"
  wat2wasm "$wat" -o "$wasm"
done
