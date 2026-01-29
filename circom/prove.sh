#!/bin/bash
# Build and prove Monster Walk Music zkSNARK

set -e

echo "🔐 Monster Walk Music zkSNARK Proof"
echo "===================================="

# Compile circuit
echo "📦 Compiling circuit..."
circom circom/monster_walk_music.circom --r1cs --wasm --sym -o circom/build/

# Generate witness
echo "🔍 Generating witness..."
cd circom/build/monster_walk_music_js
node generate_witness.js monster_walk_music.wasm ../../input.json witness.wtns
cd ../../..

# Setup (Powers of Tau)
echo "⚡ Powers of Tau ceremony..."
snarkjs powersoftau new bn128 12 circom/build/pot12_0000.ptau -v
snarkjs powersoftau contribute circom/build/pot12_0000.ptau circom/build/pot12_0001.ptau --name="Monster" -v
snarkjs powersoftau prepare phase2 circom/build/pot12_0001.ptau circom/build/pot12_final.ptau -v

# Generate proving key
echo "🔑 Generating proving key..."
snarkjs groth16 setup circom/build/monster_walk_music.r1cs circom/build/pot12_final.ptau circom/build/monster_walk_music_0000.zkey
snarkjs zkey contribute circom/build/monster_walk_music_0000.zkey circom/build/monster_walk_music_final.zkey --name="Monster Walk" -v
snarkjs zkey export verificationkey circom/build/monster_walk_music_final.zkey circom/build/verification_key.json

# Generate proof
echo "✨ Generating proof..."
snarkjs groth16 prove circom/build/monster_walk_music_final.zkey circom/build/monster_walk_music_js/witness.wtns circom/build/proof.json circom/build/public.json

# Verify proof
echo "✅ Verifying proof..."
snarkjs groth16 verify circom/build/verification_key.json circom/build/public.json circom/build/proof.json

echo ""
echo "🎵 Monster Walk Music proven in zero-knowledge!"
echo "   - 10 steps verified"
echo "   - 8/8 time signature verified"
echo "   - 80 BPM tempo verified"
echo "   - All Monster primes verified"
echo "   - Frequency ordering verified"
echo ""
echo "📄 Proof: circom/build/proof.json"
echo "🔑 Verification key: circom/build/verification_key.json"
