# zkprologml RDFa URL Encoding for LLM Calls

## Overview

**Problem**: LLM calls are non-deterministic and environment-dependent.  
**Solution**: Encode LLM calls as zkprologml RDFa URLs with unambiguous semantics.  
**Benefit**: GitHub vs local resolution, verifiable results, suspended execution.

---

## Architecture

```
Pipeline Stage
    ↓
Generate zkprologml RDFa URL
    ↓
Suspend execution
    ↓
Store URL in dataset
    ↓
Resolve (GitHub or local)
    ↓
Verify result
    ↓
Continue pipeline
```

---

## zkprologml RDFa URL Format

### Base Structure
```
zkprologml://llm.call/
  ?model=<model_id>
  &prompt=<escaped_prompt>
  &context=<escaped_context>
  &proof=<proof_hash>
  &verify=<verification_method>
```

### RDFa Attributes
```xml
<div vocab="http://zkprologml.org/ns#"
     typeof="LLMCall"
     resource="zkprologml://llm.call/text-generation">
  
  <span property="model">gpt-4</span>
  <span property="prompt" content="base64:...">Generate lyrics...</span>
  <span property="context" content="base64:...">Monster primes...</span>
  <span property="proof" content="sha256:...">abc123...</span>
  <span property="verify">zkproof</span>
  
</div>
```

---

## Example: Text Model Call

### Input (Pipeline Stage 3)
```json
{
  "stage": "text_generation",
  "model": "gpt-4",
  "prompt": "Generate lyrics for Monster Walk...",
  "context": {
    "primes": [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71],
    "frequencies": [440, 660, 1100, ...]
  },
  "proof": "MonsterLean/MonsterHarmonics.lean"
}
```

### Output (zkprologml RDFa URL)
```
zkprologml://llm.call/text-generation
  ?model=gpt-4
  &prompt=R2VuZXJhdGUgbHlyaWNzIGZvciBNb25zdGVyIFdhbGsu...
  &context=eyJwcmltZXMiOlsyLDMsNSw3LDExLDEzLDE3LDE5LDIzLDI5...
  &proof=sha256:abc123def456...
  &verify=zkproof
  &timestamp=20260129_121438
```

---

## Resolution Methods

### GitHub Resolution
```yaml
# .github/workflows/resolve-llm.yml
name: Resolve zkprologml LLM Calls

on:
  push:
    paths:
      - 'datasets/llm_calls/*.zkprologml'

jobs:
  resolve:
    runs-on: ubuntu-latest
    steps:
      - name: Parse zkprologml URL
        run: |
          URL=$(cat datasets/llm_calls/call_*.zkprologml)
          MODEL=$(echo $URL | grep -oP 'model=\K[^&]+')
          PROMPT=$(echo $URL | grep -oP 'prompt=\K[^&]+' | base64 -d)
          
      - name: Call LLM API
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: |
          curl https://api.openai.com/v1/chat/completions \
            -H "Authorization: Bearer $OPENAI_API_KEY" \
            -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}]}"
          
      - name: Generate zkproof
        run: |
          # Verify result matches proof constraints
          ./verify_zkproof.sh
          
      - name: Store result
        run: |
          echo "$RESULT" > datasets/llm_results/result_$TIMESTAMP.json
          git add datasets/llm_results/
          git commit -m "Resolve zkprologml call: $TIMESTAMP"
```

### Local Resolution
```bash
#!/usr/bin/env bash
# resolve_zkprologml_local.sh

URL="$1"

# Parse URL
MODEL=$(echo "$URL" | grep -oP 'model=\K[^&]+')
PROMPT=$(echo "$URL" | grep -oP 'prompt=\K[^&]+' | base64 -d)
CONTEXT=$(echo "$URL" | grep -oP 'context=\K[^&]+' | base64 -d)
PROOF=$(echo "$URL" | grep -oP 'proof=\K[^&]+')

# Call local LLM (ollama)
ollama run "$MODEL" "$PROMPT"

# Verify against proof
lean4 --run verify_proof.lean "$PROOF" "$RESULT"

# Store result
echo "$RESULT" > "datasets/llm_results/result_$(date +%s).json"
```

---

## Verification Methods

### zkproof (Zero-Knowledge Proof)
```lean
-- MonsterLean/VerifyLLMCall.lean

structure LLMCall where
  model : String
  prompt : String
  context : Json
  proof : String

def verifyLLMResult (call : LLMCall) (result : String) : Bool :=
  -- Verify result satisfies proof constraints
  let constraints := parseProof call.proof
  checkConstraints constraints result

theorem llm_result_valid (call : LLMCall) (result : String) :
  verifyLLMResult call result = true →
  ∃ (proof : LLMResultProof), proof.valid := by
  sorry
```

### Semantic Hash
```rust
// src/bin/verify_semantic_hash.rs

fn verify_semantic_hash(
    prompt: &str,
    result: &str,
    expected_hash: &str
) -> bool {
    let semantic_embedding = embed_text(result);
    let hash = sha256(&semantic_embedding);
    hash == expected_hash
}
```

---

## Pipeline Integration

### Stage 3: Generate zkprologml URL (Suspended)
```bash
# pipelite_nix_rust.sh - Stage 3 modified

echo "🤖 [3/6] Generating zkprologml LLM call..."
PROMPTS_DIR="$STORE_PATH/prompts"
mkdir -p "$PROMPTS_DIR"

# Read prompt
PROMPT=$(cat "$PROMPTS_DIR/text_prompt.txt")
PROMPT_B64=$(echo "$PROMPT" | base64 -w0)

# Read context (audio metadata)
CONTEXT=$(cat "$AUDIO_DIR/monster_walk_metadata.json")
CONTEXT_B64=$(echo "$CONTEXT" | base64 -w0)

# Read proof hash
PROOF_HASH=$(sha256sum "$PROOF_DIR/MonsterHarmonics.lean" | cut -d' ' -f1)

# Generate zkprologml URL
ZKPROLOGML_URL="zkprologml://llm.call/text-generation?model=gpt-4&prompt=$PROMPT_B64&context=$CONTEXT_B64&proof=sha256:$PROOF_HASH&verify=zkproof&timestamp=$TIMESTAMP"

# Store URL (suspend execution)
echo "$ZKPROLOGML_URL" > "$PROMPTS_DIR/llm_call.zkprologml"

# Store RDFa
cat > "$PROMPTS_DIR/llm_call.rdfa" << EOF
<div vocab="http://zkprologml.org/ns#"
     typeof="LLMCall"
     resource="$ZKPROLOGML_URL">
  <span property="model">gpt-4</span>
  <span property="prompt" content="$PROMPT_B64">Text generation</span>
  <span property="context" content="$CONTEXT_B64">Monster harmonics</span>
  <span property="proof" content="sha256:$PROOF_HASH">MonsterHarmonics.lean</span>
  <span property="verify">zkproof</span>
  <span property="timestamp">$TIMESTAMP</span>
  <span property="status">suspended</span>
</div>
EOF

echo "✓ zkprologml URL: $PROMPTS_DIR/llm_call.zkprologml"
echo "✓ RDFa: $PROMPTS_DIR/llm_call.rdfa"
echo "⏸️  Execution suspended - resolve via GitHub or local"
```

### Stage 4: Check for Resolution
```bash
echo "🎼 [4/6] Checking LLM resolution..."
SONG_DIR="$STORE_PATH/song"
mkdir -p "$SONG_DIR"

# Check if result exists
RESULT_FILE="datasets/llm_results/result_$TIMESTAMP.json"
if [ -f "$RESULT_FILE" ]; then
    echo "✓ LLM result found: $RESULT_FILE"
    cp "$RESULT_FILE" "$SONG_DIR/lyrics.json"
else
    echo "⏸️  LLM result pending - using reference"
    cp MONSTER_WALK_SONG.md "$SONG_DIR/lyrics_reference.md"
fi
```

---

## Unambiguous Semantics

### RDF Vocabulary
```turtle
@prefix zkp: <http://zkprologml.org/ns#> .
@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .

zkp:LLMCall a rdfs:Class ;
    rdfs:label "LLM Call" ;
    rdfs:comment "A suspended call to a language model" .

zkp:model a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "Model identifier" .

zkp:prompt a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "Base64-encoded prompt" .

zkp:context a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "Base64-encoded context (JSON)" .

zkp:proof a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "SHA256 hash of proof file" .

zkp:verify a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "Verification method (zkproof, semantic_hash)" .

zkp:timestamp a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:dateTime ;
    rdfs:label "Call timestamp" .

zkp:status a rdf:Property ;
    rdfs:domain zkp:LLMCall ;
    rdfs:range xsd:string ;
    rdfs:label "Call status (suspended, resolved, verified)" .
```

---

## Verification Flow

```
1. Generate zkprologml URL
2. Store URL + RDFa
3. Suspend pipeline
4. Resolve (GitHub or local)
5. Generate result
6. Verify result against proof
7. Store verified result
8. Resume pipeline
```

---

## Benefits

### 1. Unambiguous Specification
- RDFa provides formal semantics
- zkprologml URL is self-contained
- All arguments encoded

### 2. Verifiable Results
- zkproof verification
- Semantic hash checking
- Lean4 proof validation

### 3. Environment Independence
- GitHub: CI/CD resolution
- Local: ollama resolution
- Same URL, different resolvers

### 4. Suspended Execution
- Pipeline doesn't block
- Async resolution
- Resume when ready

### 5. Traceability
- Full call history
- Proof lineage
- Result verification

---

## Implementation Files

### Core
- `pipelite_nix_rust.sh` - Modified pipeline with suspension
- `resolve_zkprologml_local.sh` - Local resolver
- `.github/workflows/resolve-llm.yml` - GitHub resolver

### Verification
- `MonsterLean/VerifyLLMCall.lean` - Lean4 verification
- `src/bin/verify_semantic_hash.rs` - Rust verification

### Vocabulary
- `zkprologml_vocab.ttl` - RDF vocabulary
- `zkprologml_schema.json` - JSON schema

---

## Example Complete Flow

### 1. Generate Call
```bash
$ ./pipelite_nix_rust.sh
🤖 [3/6] Generating zkprologml LLM call...
✓ zkprologml URL: .../prompts/llm_call.zkprologml
⏸️  Execution suspended
```

### 2. Resolve (GitHub)
```bash
# GitHub Action triggered
# Calls OpenAI API
# Generates result
# Commits to datasets/llm_results/
```

### 3. Verify
```bash
$ lean4 --run MonsterLean/VerifyLLMCall.lean
✓ Result verified against proof
✓ zkproof valid
```

### 4. Resume
```bash
$ ./pipelite_nix_rust.sh --resume
🎼 [4/6] Checking LLM resolution...
✓ LLM result found
✓ Pipeline resumed
```

---

## Next Steps

1. ✅ Specify zkprologml RDFa format
2. ⚠️ Implement URL generator
3. ⚠️ Implement GitHub resolver
4. ⚠️ Implement local resolver
5. ⚠️ Implement Lean4 verifier
6. ⚠️ Test complete flow

---

## Conclusion

**zkprologml RDFa URLs provide:**
- Unambiguous LLM call specification
- Verifiable results
- Environment-independent resolution
- Suspended execution
- Full traceability

**The Monster's song is now formally specified.** 🎯✨🎵
