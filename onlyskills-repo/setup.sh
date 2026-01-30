#!/bin/bash
# Quick setup and deploy script

echo "🌌 onlyskills.com - Setup & Deploy"
echo "=================================="

# Initialize git repo
git init
git add .
git commit -m "Initial commit: zkERDAProlog 71-shard registry"

# Add remotes
echo "📡 Adding remotes..."
git remote add origin https://github.com/onlyskills/zkerdaprologml.git
git remote add hf https://huggingface.co/spaces/onlyskills/registry

# Deploy to Vercel
echo "🚀 Deploying to Vercel..."
vercel --prod

# Deploy to HuggingFace
echo "🤗 Deploying to HuggingFace..."
git push hf main

# Deploy to Archive.org
echo "📚 Uploading to Archive.org..."
ia upload onlyskills-zkerdfa \
  onlyskills_zkerdfa.ttl \
  onlyskills_profiles.json \
  onlyskills_registration.json \
  --metadata="title:onlyskills.com zkERDAProlog Registry" \
  --metadata="description:Zero-Knowledge 71-Shard Skill Registry for AI Agents" \
  --metadata="collection:opensource" \
  --metadata="subject:artificial-intelligence;zero-knowledge;skills;monster-group"

echo ""
echo "✅ Deployment Complete!"
echo ""
echo "🌐 Live URLs:"
echo "  - Vercel: https://onlyskills.vercel.app"
echo "  - HuggingFace: https://huggingface.co/spaces/onlyskills/registry"
echo "  - Archive.org: https://archive.org/details/onlyskills-zkerdfa"
echo ""
echo "∞ 71 Shards. 71 Platforms. Zero Knowledge. ∞"
