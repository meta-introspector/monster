#!/usr/bin/env bash
set -e

echo "🎪 Monster Walk - Complete Test Suite (Pure Rust)"
echo "=================================================="
echo ""

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

step() {
    echo -e "${BLUE}▶ $1${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

# Step 1: Build papers with Nix
step "Building papers with Nix..."
nix build .#paper -L
nix build .#paper-v2 -L
success "Papers built"

# Step 2: Build WASM
step "Building WASM module..."
cd wasm
cargo build --target wasm32-unknown-unknown --release
wasm-pack build --target web --out-dir ../test-site/wasm
cd ..
success "WASM built"

# Step 3: Setup test site
step "Setting up test site..."
mkdir -p test-results
mkdir -p test-site
cp web/index.html test-site/
cp web/style.css test-site/
cp result/*.html test-site/ 2>/dev/null || true
cp result/*.pdf test-site/ 2>/dev/null || true
success "Test site ready"

# Step 4: Run Rust integration tests with headless Chrome
step "Running Rust integration tests..."
cd tests
CHROME_BIN=$(which chromium || which google-chrome || which chrome) cargo test --release -- --nocapture --test-threads=1
cd ..
success "All Rust tests passed"

# Step 5: Test with nektos/act
step "Testing GitHub Actions locally with act..."
if command -v act &> /dev/null; then
    act -l
    echo "Run full workflow with: act -j build-and-test"
else
    echo "Install act: curl https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash"
fi

# Step 6: Display AI report
step "AI Accessibility Report..."
if [ -f test-results/ai-accessibility-report.json ]; then
    echo "Report generated successfully!"
    cat test-results/ai-accessibility-report.json | head -30
    success "AI can read the site"
fi

# Step 7: Start local server (Rust)
step "Building local server..."
cat > test-site/server.rs << 'EOF'
use std::net::TcpListener;
use std::io::{Read, Write};
use std::fs;

fn main() {
    let listener = TcpListener::bind("127.0.0.1:8000").unwrap();
    println!("Server running at http://localhost:8000");
    
    for stream in listener.incoming() {
        if let Ok(mut stream) = stream {
            let mut buffer = [0; 1024];
            stream.read(&mut buffer).unwrap();
            
            let request = String::from_utf8_lossy(&buffer);
            let path = request.lines().next()
                .and_then(|line| line.split_whitespace().nth(1))
                .unwrap_or("/");
            
            let file_path = if path == "/" { "index.html" } else { &path[1..] };
            
            let (status, content) = if let Ok(content) = fs::read(file_path) {
                ("200 OK", content)
            } else {
                ("404 NOT FOUND", b"Not Found".to_vec())
            };
            
            let response = format!("HTTP/1.1 {}\r\n\r\n", status);
            stream.write_all(response.as_bytes()).unwrap();
            stream.write_all(&content).unwrap();
        }
    }
}
EOF

echo ""
echo "======================================"
echo "🎉 All tests passed!"
echo "======================================"
echo ""
echo "Outputs:"
echo "  📄 Papers: result/monster_walk*.pdf"
echo "  🌐 HTML: result/monster_walk*.html"
echo "  🧪 Tests: test-results/"
echo "  🤖 AI Report: test-results/ai-accessibility-report.json"
echo ""
echo "To serve locally (Rust):"
echo "  cd test-site && rustc server.rs && ./server"
