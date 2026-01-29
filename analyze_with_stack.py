#!/usr/bin/env python3
"""
Compare Monster project code against The Stack v2 dataset.
Analyzes code patterns, complexity, and similarity.
"""

from datasets import load_dataset
import pandas as pd
from pathlib import Path
import os

def load_our_code():
    """Load Monster project code."""
    code = []
    
    # Rust files
    for f in Path('src').rglob('*.rs'):
        with open(f) as fp:
            content = fp.read()
            code.append({
                'file': str(f),
                'language': 'Rust',
                'content': content,
                'size': len(content),
                'lines': content.count('\n')
            })
    
    # Lean4 files
    for f in Path('MonsterLean').rglob('*.lean'):
        with open(f) as fp:
            content = fp.read()
            code.append({
                'file': str(f),
                'language': 'Lean',
                'content': content,
                'size': len(content),
                'lines': content.count('\n')
            })
    
    return pd.DataFrame(code)

def sample_stack_v2(language='Rust', n_samples=100):
    """Sample The Stack v2 for comparison."""
    print(f"Loading The Stack v2 ({language})...")
    
    ds = load_dataset(
        "bigcode/the-stack-v2",
        language,
        split="train",
        streaming=True
    )
    
    samples = []
    for i, item in enumerate(ds):
        if i >= n_samples:
            break
        
        content = item.get('content', '')
        samples.append({
            'repo': item.get('repository_name', 'unknown'),
            'language': language,
            'size': len(content),
            'lines': content.count('\n'),
            'stars': item.get('stars', 0)
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Loaded {i + 1} samples...")
    
    return pd.DataFrame(samples)

def analyze_complexity(df):
    """Analyze code complexity metrics."""
    return {
        'total_files': len(df),
        'total_size': df['size'].sum(),
        'total_lines': df['lines'].sum(),
        'avg_size': df['size'].mean(),
        'avg_lines': df['lines'].mean(),
        'median_size': df['size'].median(),
        'median_lines': df['lines'].median()
    }

def main():
    print("=== Monster Project vs The Stack v2 ===\n")
    
    # Load our code
    print("Loading Monster project code...")
    our_code = load_our_code()
    
    print(f"  Found {len(our_code)} files")
    print(f"  Rust: {len(our_code[our_code['language'] == 'Rust'])}")
    print(f"  Lean: {len(our_code[our_code['language'] == 'Lean'])}")
    
    # Sample The Stack v2
    stack_rust = sample_stack_v2('Rust', n_samples=100)
    
    # Analyze
    our_metrics = analyze_complexity(our_code)
    stack_metrics = analyze_complexity(stack_rust)
    
    print("\n=== Our Code ===")
    for k, v in our_metrics.items():
        print(f"  {k}: {v:.0f}")
    
    print("\n=== The Stack v2 (Rust) ===")
    for k, v in stack_metrics.items():
        print(f"  {k}: {v:.0f}")
    
    # Save results
    our_code.to_parquet('our_code_analysis.parquet', index=False)
    stack_rust.to_parquet('stack_comparison.parquet', index=False)
    
    # Summary
    summary = pd.DataFrame([
        {'source': 'Monster', **our_metrics},
        {'source': 'The Stack v2', **stack_metrics}
    ])
    summary.to_parquet('stack_summary.parquet', index=False)
    
    print("\n✓ Saved parquet files")
    print("  - our_code_analysis.parquet")
    print("  - stack_comparison.parquet")
    print("  - stack_summary.parquet")

if __name__ == '__main__':
    main()
