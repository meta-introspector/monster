#!/usr/bin/env python3
"""Calculate Hecke operators from git commit patterns"""
import subprocess
import json
from pathlib import Path
from collections import defaultdict

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def get_author_commits(author_name):
    """Get commit stats for author from current git repo"""
    try:
        # Get commit count
        result = subprocess.run(
            ['git', 'log', '--author', author_name, '--oneline'],
            capture_output=True, text=True, timeout=5, cwd='.'
        )
        commits = len(result.stdout.strip().split('\n')) if result.stdout.strip() else 0
        
        # Get line changes
        result = subprocess.run(
            ['git', 'log', '--author', author_name, '--pretty=tformat:', '--numstat'],
            capture_output=True, text=True, timeout=5, cwd='.'
        )
        
        lines_added = 0
        lines_deleted = 0
        
        for line in result.stdout.split('\n'):
            if line and '\t' in line:
                parts = line.split('\t')
                if len(parts) >= 2 and parts[0].isdigit():
                    lines_added += int(parts[0])
                    lines_deleted += int(parts[1])
        
        return {
            'commits': commits,
            'lines_added': lines_added,
            'lines_deleted': lines_deleted,
            'total_changes': lines_added + lines_deleted
        }
    except:
        return {'commits': 0, 'lines_added': 0, 'lines_deleted': 0, 'total_changes': 0}

def calculate_hecke_operator(stats):
    """Calculate Hecke operator T_p based on commit statistics"""
    total = stats['total_changes']
    if total == 0:
        return None
    
    # Find which Monster prime divides the contribution
    hecke_ops = {}
    
    for p in MONSTER_PRIMES:
        # T_p eigenvalue = number of changes divisible by p
        divisible_count = total // p
        if divisible_count > 0:
            hecke_ops[f'T_{p}'] = {
                'prime': p,
                'eigenvalue': divisible_count,
                'resonance': (total % p) / p  # How close to perfect divisibility
            }
    
    # Primary Hecke operator = largest prime that divides significantly
    primary = None
    max_eigenvalue = 0
    
    for p in reversed(MONSTER_PRIMES):  # Start from largest
        if total % p == 0:
            primary = p
            break
        elif total // p > max_eigenvalue:
            max_eigenvalue = total // p
            primary = p
    
    return {
        'primary_operator': f'T_{primary}' if primary else None,
        'primary_prime': primary,
        'all_operators': hecke_ops,
        'total_changes': total,
        'commits': stats['commits']
    }

def analyze_author_hecke(author_name, stats):
    """Full Hecke analysis for an author"""
    hecke = calculate_hecke_operator(stats)
    
    if not hecke or not hecke['primary_operator']:
        return None
    
    return {
        'author': author_name,
        'hecke_operator': hecke['primary_operator'],
        'prime': hecke['primary_prime'],
        'eigenvalue': hecke['all_operators'].get(hecke['primary_operator'], {}).get('eigenvalue', 0),
        'total_changes': hecke['total_changes'],
        'commits': hecke['commits'],
        'resonance': hecke['all_operators'].get(hecke['primary_operator'], {}).get('resonance', 0)
    }

def main():
    """Calculate Hecke operators for all authors"""
    print("🔢 HECKE OPERATOR ANALYSIS")
    print("=" * 60)
    
    # Get all authors from current git repo
    result = subprocess.run(
        ['git', 'log', '--format=%an', '--all'],
        capture_output=True, text=True, cwd='.'
    )
    
    all_authors = list(set(result.stdout.strip().split('\n')))
    print(f"Found {len(all_authors)} unique authors in repo\n")
    
    results = []
    prime_distribution = defaultdict(list)
    
    for i, author in enumerate(all_authors):
        print(f"[{i+1}/{len(all_authors)}] {author[:40]}...", end=' ', flush=True)
        
        stats = get_author_commits(author)
        hecke = analyze_author_hecke(author, stats)
        
        if hecke:
            results.append(hecke)
            prime_distribution[hecke['prime']].append(author)
            print(f"T_{hecke['prime']} ✓")
        else:
            print("✗")
    
    # Save results
    with open('author_hecke_operators.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 HECKE OPERATOR DISTRIBUTION")
    print("=" * 60)
    
    for p in MONSTER_PRIMES:
        count = len(prime_distribution[p])
        if count > 0:
            print(f"T_{p:2d}: {count:3d} authors ({count/len(results)*100:.1f}%)")
    
    print(f"\nTotal analyzed: {len(results)} authors")
    print(f"Saved to: author_hecke_operators.json")
    
    # Find most resonant authors per prime
    print("\n" + "=" * 60)
    print("🌟 MOST RESONANT AUTHORS PER PRIME")
    print("=" * 60)
    
    for p in [71, 59, 47, 41, 31, 29, 23]:  # Largest Monster primes
        authors = prime_distribution[p]
        if authors:
            # Get top contributor for this prime
            top = max([r for r in results if r['prime'] == p], 
                     key=lambda x: x['eigenvalue'])
            print(f"T_{p}: {top['author']} (eigenvalue={top['eigenvalue']})")

if __name__ == '__main__':
    main()
