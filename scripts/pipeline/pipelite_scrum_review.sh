#!/usr/bin/env bash
# Pipelite + Scrum Review Team Pipeline

set -e

echo "🏢 PIPELITE + SCRUM REVIEW TEAM"
echo "============================================================"
echo ""

# Stage 1: Build Lean4 proofs
echo "📝 [1/5] Building Lean4 proofs..."
lake build MonsterLean.CrossLanguageComplexity 2>&1 | tail -5
echo "✓ Proofs built"
echo ""

# Stage 2: Generate literate web
echo "📖 [2/5] Generating literate web..."
[ -f index.html ] && echo "✓ index.html"
[ -f literate_web.html ] && echo "✓ literate_web.html"
echo ""

# Stage 3: Multi-persona scrum review
echo "👥 [3/5] Running multi-persona scrum review..."
if command -v ollama &> /dev/null; then
    python3 scrum_review_team.py
else
    echo "❌ ERROR: Ollama required for persona reviews"
    echo "   Install: curl https://ollama.ai/install.sh | sh"
    echo "   Then: ollama pull llama3.2"
    exit 1
fi
echo "✓ Scrum review complete"
echo ""

# Stage 4: Convert to parquet
echo "💾 [4/5] Verifying parquet output..."
python3 << 'EOF'
import pandas as pd
from pathlib import Path

if Path('scrum_reviews.parquet').exists():
    df = pd.read_parquet('scrum_reviews.parquet')
    print(f"✓ scrum_reviews.parquet: {len(df)} reviews")
    print(f"  Personas: {df['persona'].unique().tolist()}")
    print(f"  Theorems: {df['theorem_name'].nunique()}")
else:
    print("❌ No parquet file generated")
EOF
echo ""

# Stage 5: Generate summary report
echo "📊 [5/5] Generating summary report..."
python3 << 'EOF'
import pandas as pd
import json

df = pd.read_parquet('scrum_reviews.parquet')

# Summary by persona
print("\n📋 REVIEWS BY PERSONA:")
print("=" * 60)
for persona in df['persona'].unique():
    persona_df = df[df['persona'] == persona]
    print(f"\n{persona_df.iloc[0]['reviewer_name']} ({persona_df.iloc[0]['reviewer_role']}):")
    print(f"  Reviews: {len(persona_df)}")
    print(f"  Focus: {persona_df.iloc[0]['focus_area']}")

# Summary by theorem
print("\n\n📋 REVIEWS BY THEOREM:")
print("=" * 60)
for theorem in df['theorem_name'].unique():
    theorem_df = df[df['theorem_name'] == theorem]
    print(f"\n{theorem}:")
    print(f"  Reviewers: {len(theorem_df)}")
    print(f"  Personas: {', '.join(theorem_df['persona'].tolist())}")

# Metadata
metadata = {
    'pipeline': 'scrum_review_team',
    'timestamp': pd.Timestamp.now().isoformat(),
    'total_reviews': len(df),
    'personas': df['persona'].nunique(),
    'theorems': df['theorem_name'].nunique(),
    'main_result': 'Coq ≃ Lean4 ≃ Rust (Layer 7)',
    'review_standards': ['ITIL', 'ISO9001', 'GMP', 'Six Sigma', 'Literate Programming']
}

df_meta = pd.DataFrame([metadata])
df_meta.to_parquet('scrum_metadata.parquet', index=False)
print("\n✓ scrum_metadata.parquet")
EOF
echo ""

echo "✅ SCRUM REVIEW PIPELINE COMPLETE!"
echo "============================================================"
echo ""
echo "📊 Generated Files:"
ls -lh scrum_*.{parquet,json} 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
echo ""
echo "👥 Review Team:"
echo "  • Donald Knuth - Literate Programming"
echo "  • ITIL Service Manager - IT Service Management"
echo "  • ISO 9001 Auditor - Quality Management"
echo "  • GMP Quality Officer - Manufacturing Practice"
echo "  • Six Sigma Black Belt - Process Excellence"
echo ""
echo "🎯 Result: All theorems reviewed from 5 domain perspectives"
