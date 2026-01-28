#!/bin/bash
# Generate directory structure for all 71 shards

echo "🎪 Creating 71 Monster Shards"
echo "=============================="

mkdir -p monster-shards

for n in {1..71}; do
    echo "Creating shard-$(printf "%02d" $n)..."
    
    SHARD_DIR="monster-shards/shard-$(printf "%02d" $n)"
    
    # Create directory structure
    mkdir -p "$SHARD_DIR"/{paper,code/src,data,zk,demo}
    
    # Create paper template
    cat > "$SHARD_DIR/paper/shard-$(printf "%02d" $n).html" << EOF
<!DOCTYPE html>
<html vocab="http://schema.org/" typeof="ScholarlyArticle"
      resource="https://monster-shards.io/shard-$n">
<head>
    <meta charset="UTF-8">
    <meta property="identifier" content="urn:monster:shard:$n"/>
    <meta property="isPartOf" content="urn:monster:lattice:71"/>
    <meta property="primeNumber" content="$n"/>
    <title property="name">Shard $n: Monster Lattice Decomposition</title>
</head>
<body>
    <article>
        <h1 property="name">Shard $n: Monster Lattice Decomposition</h1>
        
        <div property="author" typeof="Organization">
            <span property="name">Monster Group Walk Project</span>
        </div>
        
        <section property="abstract">
            <h2>Abstract</h2>
            <p>This paper presents Shard $n of the 71-piece Monster lattice decomposition...</p>
        </section>
        
        <section property="hasPart" typeof="SoftwareSourceCode">
            <h2>Implementation</h2>
            <meta property="programmingLanguage" content="Rust"/>
            <meta property="runtimePlatform" content="WebAssembly"/>
            <link property="codeRepository" href="https://github.com/monster-lean/shard-$n"/>
            <p><a href="../demo/index.html">Run Interactive Demo</a></p>
        </section>
        
        <section property="hasPart" typeof="Dataset">
            <h2>Data</h2>
            <meta property="encodingFormat" content="application/x-gguf"/>
            <link property="distribution" href="../data/shard-$n.gguf"/>
        </section>
        
        <section property="hasPart" typeof="Proof">
            <h2>Zero-Knowledge Proof</h2>
            <meta property="proofType" content="ZK-SNARK"/>
            <link property="verificationCircuit" href="../zk/circuit.circom"/>
            <p><a href="../zk/verify.html">Verify Proof</a></p>
        </section>
    </article>
</body>
</html>
EOF

    # Create Cargo.toml
    cat > "$SHARD_DIR/code/Cargo.toml" << EOF
[package]
name = "monster-shard-$n"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"

[profile.release]
opt-level = "z"
lto = true
EOF

    # Create lib.rs
    cat > "$SHARD_DIR/code/src/lib.rs" << EOF
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct MonsterShard {
    number: u32,
    neurons: Vec<f32>,
}

#[wasm_bindgen]
impl MonsterShard {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            number: $n,
            neurons: vec![],
        }
    }
    
    pub fn get_number(&self) -> u32 {
        self.number
    }
    
    pub fn forward(&self, input: Vec<f32>) -> Vec<f32> {
        // Apply Hecke operator
        input.iter()
            .map(|&x| x * (self.number as f32 / 10.0))
            .collect()
    }
}
EOF

    # Create demo
    cat > "$SHARD_DIR/demo/index.html" << EOF
<!DOCTYPE html>
<html>
<head>
    <title>Shard $n Demo</title>
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
        h1 { color: #4ecdc4; }
        button { background: #4ecdc4; color: #000; border: none; padding: 10px 20px; cursor: pointer; }
    </style>
</head>
<body>
    <h1>🎪 Monster Shard $n</h1>
    <p>Interactive demonstration of Shard $n</p>
    <button onclick="runShard()">Run Shard</button>
    <div id="output"></div>
    
    <script type="module">
        async function runShard() {
            document.getElementById('output').textContent = 'Running Shard $n...';
            // WASM module would be loaded here
        }
        window.runShard = runShard;
    </script>
</body>
</html>
EOF

    # Create ZK circuit
    cat > "$SHARD_DIR/zk/circuit.circom" << EOF
pragma circom 2.0.0;

template MonsterShard$n() {
    signal input neurons[100];
    signal output valid;
    
    // Verify all neurons divisible by $n
    var sum = 0;
    for (var i = 0; i < 100; i++) {
        sum += neurons[i] % $n;
    }
    
    valid <== (sum == 0) ? 1 : 0;
}

component main = MonsterShard$n();
EOF

    # Create metadata
    cat > "$SHARD_DIR/metadata.json" << EOF
{
  "shard": $n,
  "type": "$(if [ $n -eq 2 ] || [ $n -eq 3 ] || [ $n -eq 5 ] || [ $n -eq 7 ] || [ $n -eq 11 ] || [ $n -eq 13 ] || [ $n -eq 17 ] || [ $n -eq 19 ] || [ $n -eq 23 ] || [ $n -eq 29 ] || [ $n -eq 31 ] || [ $n -eq 41 ] || [ $n -eq 47 ] || [ $n -eq 59 ] || [ $n -eq 71 ]; then echo "prime"; else echo "composite"; fi)",
  "godel": "$n^$n",
  "status": "planned",
  "release_week": $(( (n - 1) / 4 + 1 ))
}
EOF

done

# Create index
cat > monster-shards/index.html << 'EOF'
<!DOCTYPE html>
<html vocab="http://schema.org/">
<head>
    <meta charset="UTF-8">
    <title>Monster Shards: 71-Piece Lattice Decomposition</title>
    <style>
        body { font-family: monospace; background: #0a0a0a; color: #e0e0e0; padding: 20px; }
        h1 { color: #ff6b6b; text-align: center; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(100px, 1fr)); gap: 10px; }
        .shard { background: #1a1a1a; border: 2px solid #333; padding: 10px; text-align: center; cursor: pointer; }
        .shard.prime { border-color: #4ecdc4; }
        .shard:hover { border-color: #ff6b6b; }
    </style>
</head>
<body>
    <h1>🎪 Monster Shards</h1>
    <p style="text-align: center;">71-Piece Lattice Decomposition of Neural Networks</p>
    <div class="grid" id="shard-grid"></div>
    
    <script>
        const PRIMES = [2,3,5,7,11,13,17,19,23,29,31,41,47,59,71];
        const grid = document.getElementById('shard-grid');
        
        for (let n = 1; n <= 71; n++) {
            const div = document.createElement('div');
            div.className = 'shard' + (PRIMES.includes(n) ? ' prime' : '');
            div.innerHTML = \`<strong>\${n}</strong><br><small>\${PRIMES.includes(n) ? '★' : ''}</small>\`;
            div.onclick = () => window.location.href = \`shard-\${String(n).padStart(2, '0')}/paper/shard-\${String(n).padStart(2, '0')}.html\`;
            grid.appendChild(div);
        }
    </script>
</body>
</html>
EOF

echo ""
echo "✅ Created 71 shards in monster-shards/"
echo "📄 Index: monster-shards/index.html"
echo ""
echo "Next steps:"
echo "  1. cd monster-shards"
echo "  2. Open index.html in browser"
echo "  3. Start filling in content for each shard"
