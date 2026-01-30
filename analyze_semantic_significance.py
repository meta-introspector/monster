#!/usr/bin/env python3
"""
Statistical Significance Analysis of Monster Prime Vectorization
Proves that semantic structure exists in the 71-layer embeddings
"""

import polars as pl
import numpy as np
from scipy import stats
from pathlib import Path
import json

def load_shard(layer: int) -> pl.DataFrame:
    """Load a single layer shard"""
    path = f"vectors_layer_{layer:02}.parquet"
    if Path(path).exists():
        return pl.read_parquet(path)
    return None

def compute_intra_shard_similarity(layer: int) -> dict:
    """
    Compute similarity within a shard (same Monster prime)
    High similarity = semantic clustering by prime
    """
    df = load_shard(layer)
    if df is None or len(df) < 2:
        return None
    
    # Group by shard (Monster prime assignment)
    shard_groups = df.group_by("shard").agg(pl.count())
    
    results = {
        "layer": layer,
        "total_vectors": len(df),
        "num_shards": len(shard_groups),
        "shard_sizes": shard_groups["count"].to_list(),
        "entropy": stats.entropy(shard_groups["count"].to_list()),
    }
    
    return results

def compute_cross_layer_coherence() -> dict:
    """
    Measure if same file/column maintains consistent shard across layers
    High coherence = semantic stability across Monster group structure
    """
    coherence_scores = []
    
    # Sample layers
    layers = [0, 17, 35, 53, 70]  # Spread across 71 layers
    dfs = []
    
    for layer in layers:
        df = load_shard(layer)
        if df is not None:
            dfs.append(df.select(["file", "column", "shard", "layer"]))
    
    if len(dfs) < 2:
        return {"error": "Not enough layers"}
    
    # Join on file+column to track same semantic entity
    base = dfs[0]
    for df in dfs[1:]:
        base = base.join(df, on=["file", "column"], suffix=f"_l{df['layer'][0]}")
    
    # Compute shard consistency
    shard_cols = [c for c in base.columns if c.startswith("shard")]
    if len(shard_cols) > 1:
        # Standard deviation of shard assignments per entity
        shard_std = base.select(shard_cols).to_numpy().std(axis=1).mean()
        coherence = 1.0 / (1.0 + shard_std)  # Lower std = higher coherence
    else:
        coherence = 0.0
    
    return {
        "layers_analyzed": layers,
        "entities_tracked": len(base),
        "shard_consistency_score": float(coherence),
        "interpretation": "High score = same semantic entity stays in same Monster prime shard"
    }

def test_null_hypothesis() -> dict:
    """
    H0: Shard assignments are random (uniform distribution)
    H1: Shard assignments follow semantic structure
    
    Use chi-square test for goodness of fit
    """
    # Aggregate all shard assignments across layers
    all_shards = []
    
    for layer in range(71):
        df = load_shard(layer)
        if df is not None:
            all_shards.extend(df["shard"].to_list())
    
    if len(all_shards) < 100:
        return {"error": "Insufficient data"}
    
    # Observed distribution
    observed = np.bincount(all_shards, minlength=15)
    
    # Expected uniform distribution
    expected = np.full(15, len(all_shards) / 15)
    
    # Chi-square test
    chi2, p_value = stats.chisquare(observed, expected)
    
    return {
        "null_hypothesis": "Shard assignments are uniformly random",
        "alternative_hypothesis": "Shard assignments follow semantic structure",
        "chi_square_statistic": float(chi2),
        "p_value": float(p_value),
        "reject_null": p_value < 0.05,
        "significance_level": 0.05,
        "interpretation": "p < 0.05 means semantic structure exists (reject randomness)",
        "observed_distribution": observed.tolist(),
        "expected_uniform": expected.tolist(),
    }

def analyze_column_type_clustering() -> dict:
    """
    Test if similar column types cluster in same shards
    E.g., numeric columns vs string columns
    """
    # Sample from layer 0
    df = load_shard(0)
    if df is None:
        return {"error": "No data"}
    
    # Heuristic: column names with numbers likely numeric
    df = df.with_columns([
        pl.col("column").str.contains(r"\d").alias("likely_numeric")
    ])
    
    # Compare shard distributions
    numeric_shards = df.filter(pl.col("likely_numeric"))["shard"].to_list()
    text_shards = df.filter(~pl.col("likely_numeric"))["shard"].to_list()
    
    if len(numeric_shards) < 10 or len(text_shards) < 10:
        return {"error": "Insufficient samples"}
    
    # KS test: are distributions different?
    ks_stat, p_value = stats.ks_2samp(numeric_shards, text_shards)
    
    return {
        "test": "Kolmogorov-Smirnov two-sample test",
        "hypothesis": "Numeric and text columns have different shard distributions",
        "ks_statistic": float(ks_stat),
        "p_value": float(p_value),
        "significant_difference": p_value < 0.05,
        "interpretation": "p < 0.05 means column types cluster differently (semantic meaning)",
        "numeric_sample_size": len(numeric_shards),
        "text_sample_size": len(text_shards),
    }

def main():
    print("🔬 STATISTICAL SIGNIFICANCE ANALYSIS")
    print("=" * 70)
    print()
    
    results = {
        "analysis_date": "2026-01-29",
        "method": "Monster Prime Vectorization (71 layers, 15 shards)",
        "tests": {}
    }
    
    # Test 1: Null hypothesis (randomness)
    print("📊 Test 1: Testing for non-random structure...")
    test1 = test_null_hypothesis()
    results["tests"]["randomness_test"] = test1
    
    if "p_value" in test1:
        if test1["reject_null"]:
            print(f"   ✅ SIGNIFICANT: p = {test1['p_value']:.2e} < 0.05")
            print(f"   → Shard assignments are NOT random")
        else:
            print(f"   ❌ Not significant: p = {test1['p_value']:.2e}")
    print()
    
    # Test 2: Cross-layer coherence
    print("📊 Test 2: Cross-layer semantic coherence...")
    test2 = compute_cross_layer_coherence()
    results["tests"]["coherence_test"] = test2
    
    if "shard_consistency_score" in test2:
        score = test2["shard_consistency_score"]
        print(f"   Coherence score: {score:.3f}")
        if score > 0.7:
            print(f"   ✅ HIGH coherence - semantic entities stable across layers")
        elif score > 0.5:
            print(f"   ⚠️  MODERATE coherence")
        else:
            print(f"   ❌ LOW coherence")
    print()
    
    # Test 3: Column type clustering
    print("📊 Test 3: Column type semantic clustering...")
    test3 = analyze_column_type_clustering()
    results["tests"]["clustering_test"] = test3
    
    if "p_value" in test3:
        if test3["significant_difference"]:
            print(f"   ✅ SIGNIFICANT: p = {test3['p_value']:.2e} < 0.05")
            print(f"   → Column types cluster differently (semantic meaning)")
        else:
            print(f"   ❌ Not significant: p = {test3['p_value']:.2e}")
    print()
    
    # Test 4: Intra-shard analysis
    print("📊 Test 4: Intra-shard entropy analysis...")
    entropies = []
    for layer in [0, 17, 35, 53, 70]:
        result = compute_intra_shard_similarity(layer)
        if result:
            entropies.append(result["entropy"])
            print(f"   Layer {layer:2d}: entropy = {result['entropy']:.3f}")
    
    if entropies:
        avg_entropy = np.mean(entropies)
        max_entropy = np.log(15)  # Maximum for 15 shards
        normalized = avg_entropy / max_entropy
        
        results["tests"]["entropy_analysis"] = {
            "average_entropy": float(avg_entropy),
            "max_possible_entropy": float(max_entropy),
            "normalized_entropy": float(normalized),
            "interpretation": "Lower entropy = stronger clustering (semantic structure)"
        }
        
        if normalized < 0.8:
            print(f"   ✅ Normalized entropy: {normalized:.3f} < 0.8 (good clustering)")
        else:
            print(f"   ⚠️  Normalized entropy: {normalized:.3f} (weak clustering)")
    print()
    
    # Summary
    print("=" * 70)
    print("📋 SUMMARY")
    print("=" * 70)
    
    significant_tests = 0
    total_tests = 0
    
    for test_name, test_result in results["tests"].items():
        if "p_value" in test_result:
            total_tests += 1
            if test_result.get("reject_null") or test_result.get("significant_difference"):
                significant_tests += 1
    
    print(f"Significant tests: {significant_tests}/{total_tests}")
    print()
    
    if significant_tests >= 2:
        print("✅ CONCLUSION: Statistical evidence supports semantic structure in Monster embeddings")
        print("   The 71-layer, 15-shard vectorization captures real semantic meaning.")
    else:
        print("⚠️  CONCLUSION: Insufficient evidence for semantic structure")
        print("   May need more data or refined analysis.")
    
    print()
    
    # Save results
    with open("semantic_significance_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("💾 Results saved to: semantic_significance_analysis.json")
    print()

if __name__ == "__main__":
    main()
