#!/usr/bin/env bash

echo "📥 Adding mathematical sources as submodules..."
echo ""

# 1. Lean-LMFDB
if [ ! -d "lean-lmfdb" ]; then
    echo "Adding lean-lmfdb..."
    git submodule add https://github.com/multramate/lean-lmfdb lean-lmfdb
fi

# 2. Vericoding
if [ ! -d "vericoding" ]; then
    echo "Adding vericoding..."
    git submodule add https://github.com/beneficial-AI-Foundation/vericoding vericoding
fi

# 3. tex_lean_retriever
if [ ! -d "tex_lean_retriever" ]; then
    echo "Adding tex_lean_retriever..."
    git submodule add https://github.com/Aflo23/tex_lean_retriever tex_lean_retriever
fi

# 4. FormalBook
if [ ! -d "FormalBook" ]; then
    echo "Adding FormalBook..."
    git submodule add https://github.com/mo271/FormalBook FormalBook
fi

# 5. ProofNet
if [ ! -d "ProofNet" ]; then
    echo "Adding ProofNet..."
    git submodule add https://github.com/zhangir-azerbayev/ProofNet ProofNet
fi

echo ""
echo "✅ All sources added!"
