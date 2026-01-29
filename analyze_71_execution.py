#!/usr/bin/env python3
"""
Trace execution along prime 71 - understand what's computable
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Dict
import pandas as pd

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]
WEIGHTS = [46, 20, 9, 6, 2, 3, 1, 1, 1, 1, 1, 1, 1, 1, 1]
PRIME_71 = 71

@dataclass
class ExecutionTrace:
    step: int
    value: int
    operation: str
    divisible_by_71: bool
    monster_factors: List[int]
    score: int
    resonance: float
    precedence_level: int  # 70, 71, or 80

def count_monster_factors(n: int) -> List[int]:
    """Return list of Monster primes dividing n"""
    return [p for p in MONSTER_PRIMES if n % p == 0]

def compute_resonance(n: int) -> float:
    """Compute Monster resonance score"""
    if n == 0:
        return 0.0
    
    factors = count_monster_factors(n)
    weighted_sum = sum(WEIGHTS[MONSTER_PRIMES.index(p)] for p in factors)
    total_weight = sum(WEIGHTS)
    
    return weighted_sum / total_weight

def regular_mul(a: int, b: int) -> int:
    """Regular multiplication (precedence 70)"""
    return a * b

def graded_mul_71(a: int, b: int) -> int:
    """Graded multiplication (precedence 71)"""
    result = a * b
    # Extract Monster prime factors
    monster_part = 1
    for p in MONSTER_PRIMES:
        if result % p == 0:
            monster_part *= p
    return monster_part

def exp_op(a: int, b: int) -> int:
    """Exponentiation (precedence 80)"""
    return a ** b

def trace_computation(start_value: int, steps: int) -> List[ExecutionTrace]:
    """Trace computation starting from a value"""
    traces = []
    current = start_value
    
    for step in range(steps):
        # Alternate between operations
        if step % 3 == 0:
            # Regular multiplication
            next_val = regular_mul(current, 2)
            op = "regular_mul(*)"
            prec = 70
        elif step % 3 == 1:
            # Graded multiplication
            next_val = graded_mul_71(current, 3)
            op = "graded_mul(**)"
            prec = 71
        else:
            # Modular operation (simpler than full exp)
            next_val = (current + PRIME_71) % 10000
            op = "shift(+71)"
            prec = 71
        
        trace = ExecutionTrace(
            step=step,
            value=current,
            operation=op,
            divisible_by_71=(current % PRIME_71 == 0),
            monster_factors=count_monster_factors(current),
            score=len(count_monster_factors(current)),
            resonance=compute_resonance(current),
            precedence_level=prec
        )
        traces.append(trace)
        current = next_val
    
    return traces

def analyze_71_path(traces: List[ExecutionTrace]) -> Dict:
    """Analyze what happens along the 71 path"""
    
    # Count operations at each precedence level
    prec_counts = {}
    for t in traces:
        prec_counts[t.precedence_level] = prec_counts.get(t.precedence_level, 0) + 1
    
    # Find high resonance points
    high_res = [t for t in traces if t.resonance > 0.5]
    
    # Find 71-divisible points
    div_71 = [t for t in traces if t.divisible_by_71]
    
    # Compute statistics
    avg_resonance = sum(t.resonance for t in traces) / len(traces)
    max_resonance = max(t.resonance for t in traces)
    
    return {
        'total_steps': len(traces),
        'precedence_counts': prec_counts,
        'high_resonance_count': len(high_res),
        'div_71_count': len(div_71),
        'avg_resonance': avg_resonance,
        'max_resonance': max_resonance,
        'high_resonance_steps': [t.step for t in high_res[:5]],
        'div_71_steps': [t.step for t in div_71[:5]]
    }

def main():
    print("🎯 Execution Trace Along Prime 71 - Deep Analysis\n")
    print("="*60)
    
    # Test 1: Start from 71
    print("\n📊 Test 1: Starting from prime 71")
    traces_71 = trace_computation(PRIME_71, 20)
    
    print(f"\nFirst 10 steps:")
    print(f"{'Step':<6} {'Value':<10} {'Operation':<16} {'Div71':<8} {'Score':<8} {'Resonance':<10}")
    print("-" * 60)
    for t in traces_71[:10]:
        print(f"{t.step:<6} {t.value:<10} {t.operation:<16} {'YES' if t.divisible_by_71 else 'NO':<8} {t.score:<8} {t.resonance:<10.4f}")
    
    analysis_71 = analyze_71_path(traces_71)
    print(f"\nAnalysis:")
    print(f"  High resonance points: {analysis_71['high_resonance_count']}")
    print(f"  Divisible by 71: {analysis_71['div_71_count']}")
    print(f"  Average resonance: {analysis_71['avg_resonance']:.4f}")
    print(f"  Max resonance: {analysis_71['max_resonance']:.4f}")
    
    # Test 2: Start from 2*3*5*7*11*71
    print("\n📊 Test 2: Starting from 2×3×5×7×11×71 = 164010")
    start_val = 2 * 3 * 5 * 7 * 11 * 71
    traces_multi = trace_computation(start_val, 20)
    
    print(f"\nFirst 10 steps:")
    print(f"{'Step':<6} {'Value':<10} {'Operation':<16} {'Div71':<8} {'Score':<8} {'Resonance':<10}")
    print("-" * 60)
    for t in traces_multi[:10]:
        print(f"{t.step:<6} {t.value:<10} {t.operation:<16} {'YES' if t.divisible_by_71 else 'NO':<8} {t.score:<8} {t.resonance:<10.4f}")
    
    analysis_multi = analyze_71_path(traces_multi)
    print(f"\nAnalysis:")
    print(f"  High resonance points: {analysis_multi['high_resonance_count']}")
    print(f"  Divisible by 71: {analysis_multi['div_71_count']}")
    print(f"  Average resonance: {analysis_multi['avg_resonance']:.4f}")
    print(f"  Max resonance: {analysis_multi['max_resonance']:.4f}")
    
    # Test 3: Precedence comparison
    print("\n📊 Test 3: Precedence Level Analysis")
    print("\nOperations at each precedence level:")
    for prec, count in sorted(analysis_71['precedence_counts'].items()):
        print(f"  Precedence {prec}: {count} operations")
    
    # Save traces
    df_71 = pd.DataFrame([asdict(t) for t in traces_71])
    df_multi = pd.DataFrame([asdict(t) for t in traces_multi])
    
    df_71.to_parquet('trace_71_from_71.parquet', index=False)
    df_multi.to_parquet('trace_71_from_multi.parquet', index=False)
    
    print(f"\n💾 Saved traces:")
    print(f"  trace_71_from_71.parquet")
    print(f"  trace_71_from_multi.parquet")
    
    # Key findings
    print("\n🔑 Key Findings:")
    print(f"  1. Starting from 71: {analysis_71['div_71_count']}/{analysis_71['total_steps']} steps divisible by 71")
    print(f"  2. Starting from multi-prime: {analysis_multi['div_71_count']}/{analysis_multi['total_steps']} steps divisible by 71")
    print(f"  3. Graded multiplication (precedence 71) extracts Monster factors")
    print(f"  4. High resonance maintained: {analysis_multi['avg_resonance']:.4f} average")
    
    print("\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
