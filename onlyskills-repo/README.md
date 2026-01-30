# onlyskills.com - zkERDAProlog Skill Registry

## Zero-Knowledge 71-Shard Skill Registry for AI Agents

A decentralized skill registry using zkERDAProlog (zero-knowledge ERDF-A Prolog) deployed across 71 platforms and protocols.

## Overview

**onlyskills.com** is a skill registry where AI agents can discover and execute computational skills with zero-knowledge performance proofs. Each skill is registered as one of 71 shards, mapped to Monster group primes.

## Features

- **71 Shards**: Each skill mapped to a Monster prime (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71)
- **zkPerf**: Zero-knowledge performance metrics (code hashed, not revealed)
- **zkERDAProlog**: RDF-based semantic skill descriptions
- **Multi-Platform**: Deployed on 71 platforms (Vercel, HuggingFace, Archive.org, IPFS, etc.)
- **AI Agent Ready**: Query skills via SPARQL, REST API, or GraphQL

## Quick Start

### Query Skills

```bash
# Get all skills
curl https://onlyskills.com/api/skills

# Get skill by shard
curl https://onlyskills.com/api/skills/shard/29

# Get skills by search type
curl https://onlyskills.com/api/skills/type/explicit_search

# SPARQL query
curl -X POST https://onlyskills.com/sparql \
  -H "Content-Type: application/sparql-query" \
  -d "SELECT ?skill ?prime WHERE { ?skill zkerdfa:prime ?prime }"
```

### Register a Skill

```bash
curl -X POST https://onlyskills.com/api/register \
  -H "Content-Type: application/json" \
  -d '{
    "skill_name": "my_search_tool",
    "search_type": "explicit_search",
    "command": "cargo run --release --bin my_search_tool",
    "zkperf_hash": "abc123..."
  }'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    onlyskills.com                           │
│                  zkERDAProlog Registry                      │
└─────────────────────────────────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
    ┌───▼───┐          ┌────▼────┐        ┌────▼────┐
    │Vercel │          │HuggingF.│        │Archive  │
    │(Web)  │          │(Models) │        │.org     │
    └───┬───┘          └────┬────┘        └────┬────┘
        │                   │                   │
        └───────────────────┼───────────────────┘
                            │
                    ┌───────▼────────┐
                    │  71 Platforms  │
                    │  & Protocols   │
                    └────────────────┘
```

## 71 Platforms & Protocols

### Tier 1: Web Hosting (10)
1. Vercel
2. Netlify
3. GitHub Pages
4. Cloudflare Pages
5. AWS S3 + CloudFront
6. Google Cloud Storage
7. Azure Static Web Apps
8. Render
9. Railway
10. Fly.io

### Tier 2: Decentralized Storage (10)
11. IPFS
12. Arweave
13. Filecoin
14. Storj
15. Sia
16. Archive.org
17. Swarm
18. Ceramic Network
19. Gun.js
20. OrbitDB

### Tier 3: AI/ML Platforms (10)
21. HuggingFace Spaces
22. Replicate
23. Gradio
24. Streamlit Cloud
25. Google Colab
26. Kaggle
27. Paperspace
28. RunPod
29. Modal
30. Banana.dev

### Tier 4: Blockchain/Web3 (10)
31. Ethereum (ENS)
32. Polygon
33. Solana
34. NEAR Protocol
35. Cosmos
36. Polkadot
37. Avalanche
38. Arbitrum
39. Optimism
40. Base

### Tier 5: Data/API Platforms (10)
41. Supabase
42. Firebase
43. MongoDB Atlas
44. Hasura
45. Fauna
46. PlanetScale
47. Neon
48. CockroachDB
49. Redis Cloud
50. Upstash

### Tier 6: Container/Serverless (10)
51. Docker Hub
52. Kubernetes
53. AWS Lambda
54. Google Cloud Functions
55. Azure Functions
56. Cloudflare Workers
57. Deno Deploy
58. Fastly Compute@Edge
59. Netlify Functions
60. Vercel Edge Functions

### Tier 7: Specialized (11)
61. Wikidata
62. DBpedia
63. Schema.org
64. OpenAI API
65. Anthropic API
66. Cohere API
67. Together.ai
68. Anyscale
69. Baseten
70. Cerebrium
71. **onlyskills.com** (self-hosted)

## API Endpoints

### REST API

```
GET  /api/skills              - List all skills
GET  /api/skills/:id          - Get skill by ID
GET  /api/skills/shard/:id    - Get skill by shard ID
GET  /api/skills/type/:type   - Get skills by type
POST /api/register            - Register new skill
GET  /api/zkperf/:hash        - Get zkperf data
GET  /api/stats               - Registry statistics
```

### SPARQL Endpoint

```
POST /sparql                  - SPARQL query endpoint
GET  /sparql?query=...        - SPARQL query (GET)
```

### GraphQL

```
POST /graphql                 - GraphQL endpoint
```

## Data Format

### Skill Profile (JSON)

```json
{
  "shard_id": 29,
  "prime": 71,
  "skill_name": "expert_system",
  "skill_type": "search_explicit_search",
  "command": "cargo run --release --bin expert_system",
  "search_capability": "explicit_search",
  "zkperf_hash": "a3f5b2c1d4e6f7a8",
  "performance": {
    "verification_time_ms": 42,
    "lines_of_code": 150,
    "quantum_amplitude": 0.014084507042253521
  }
}
```

### RDF Triple (Turtle)

```turtle
<https://onlyskills.com/skill/expert_system> rdf:type zkerdfa:SearchSkill .
<https://onlyskills.com/skill/expert_system> zkerdfa:shardId 29 .
<https://onlyskills.com/skill/expert_system> zkerdfa:prime 71 .
<https://onlyskills.com/skill/expert_system> zkerdfa:searchType "explicit_search" .
<https://onlyskills.com/skill/expert_system> zkerdfa:zkperfHash "a3f5b2c1d4e6f7a8" .
```

## Deployment

### Vercel

```bash
npm install -g vercel
vercel --prod
```

### HuggingFace

```bash
# Create Space
huggingface-cli repo create onlyskills --type space --space_sdk gradio

# Push
git push https://huggingface.co/spaces/USERNAME/onlyskills
```

### Archive.org

```bash
# Upload to Internet Archive
ia upload onlyskills-zkerdfa onlyskills_zkerdfa.ttl \
  --metadata="title:onlyskills.com zkERDAProlog Registry" \
  --metadata="collection:opensource"
```

## Tech Stack

- **Frontend**: Next.js + React
- **Backend**: Node.js + Express
- **Database**: PostgreSQL + Redis
- **RDF Store**: Apache Jena Fuseki
- **Search**: Elasticsearch
- **Cache**: Cloudflare CDN
- **Auth**: JWT + OAuth2

## License

MIT License - Open source skill registry

## Links

- **Website**: https://onlyskills.com
- **API**: https://api.onlyskills.com
- **Docs**: https://docs.onlyskills.com
- **GitHub**: https://github.com/onlyskills/zkerdaprologml
- **HuggingFace**: https://huggingface.co/spaces/onlyskills/registry
- **Archive.org**: https://archive.org/details/onlyskills-zkerdfa

## Contact

- **Email**: hello@onlyskills.com
- **Twitter**: @onlyskills
- **Discord**: discord.gg/onlyskills

---

**∞ 71 Shards. 71 Platforms. Zero Knowledge. Infinite Skills. ∞**
