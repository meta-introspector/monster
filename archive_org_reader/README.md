# Archive.org WASM Reader

Read Monster ZK Lattice shards directly from Archive.org using WASM.

## Features

✅ **Pure WASM**: Runs entirely in browser  
✅ **Archive.org Direct**: Fetches data from archive.org  
✅ **No Backend**: Static HTML + WASM  
✅ **Self-Hosted**: Can be hosted on Archive.org itself  

## Build

```bash
cd /home/mdupont/experiments/monster/archive_org_reader

# Install wasm-pack
cargo install wasm-pack

# Build
./build.sh
```

## Test Locally

```bash
cd deploy
python3 -m http.server 8000

# Open http://localhost:8000
```

## Deploy to Archive.org

```bash
cd deploy

# Install internetarchive CLI
pip install internetarchive
ia configure

# Upload
./upload_to_archive.sh

# Access at:
# https://archive.org/details/monster-zk-lattice-reader
```

## Usage

1. **Open the page** (local or archive.org)
2. **Enter Item ID**: `monster-zk-lattice-v1`
3. **Click "Connect"**
4. **Read Shard**: Select shard 0-5, click "Read Shard"
5. **Read All**: Click "Read All (0-5)"
6. **Read Lattice**: Click "Read Value Lattice"

## Architecture

```
Browser
  ↓
WASM (Rust)
  ↓
Fetch API (CORS)
  ↓
Archive.org
  ↓
RDF Shards + Value Lattice
```

## Self-Hosting on Archive.org

The reader itself can be hosted on Archive.org:

1. Upload reader to `monster-zk-lattice-reader` item
2. Upload data to `monster-zk-lattice-v1` item
3. Access reader at: `https://archive.org/details/monster-zk-lattice-reader`
4. Reader fetches data from `monster-zk-lattice-v1`

**Result**: Fully self-contained on Archive.org! 🎯

## API

```javascript
import init, { ArchiveOrgReader } from './pkg/archive_org_shard_reader.js';

await init();

const reader = new ArchiveOrgReader("monster-zk-lattice-v1");

// Read shard
const shard = await reader.read_shard(0);
console.log(shard.triples);

// Read lattice
const lattice = await reader.read_lattice();
console.log(Object.keys(lattice).length);

// Get URL
const url = reader.get_shard_url(0);
console.log(url);
```

## CORS

Archive.org supports CORS, so the WASM can fetch directly:

```
Access-Control-Allow-Origin: *
```

## File Structure

```
deploy/
  index.html          - Web interface
  pkg/
    archive_org_shard_reader.js
    archive_org_shard_reader_bg.wasm
  upload_to_archive.sh
```

## URLs

**Reader** (after upload):
```
https://archive.org/details/monster-zk-lattice-reader
```

**Data** (after upload):
```
https://archive.org/download/monster-zk-lattice-v1/monster_shard_00_*.ttl
https://archive.org/download/monster-zk-lattice-v1/value_lattice_witnessed.json
```

## Example Output

```
Shard 0
Hash: 3083b531c65e4313
Triples: 12

<monster:value_24> rdf:type monster:Value .
<monster:value_24> monster:godelNumber "155"^^xsd:integer .
<monster:value_24> monster:usageCount "65"^^xsd:integer .
...
```

## Next Steps

1. Build: `./build.sh`
2. Test: `cd deploy && python3 -m http.server 8000`
3. Upload data: `ia upload monster-zk-lattice-v1 archive_org_shards/*`
4. Upload reader: `cd deploy && ./upload_to_archive.sh`
5. Access: `https://archive.org/details/monster-zk-lattice-reader`

The entire system runs on Archive.org with zero external dependencies! 🎯
