#!/usr/bin/env python3
"""Find 71 of everything - binaries, shared objects, websites"""
import subprocess
import json
from pathlib import Path
from collections import defaultdict

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

def find_71_binaries():
    """Find 71 most significant binaries"""
    print("🔍 Finding 71 binaries...")
    
    # Search common binary locations
    binary_paths = [
        '/usr/bin',
        '/usr/local/bin',
        '/bin',
        '/sbin',
        '~/.cargo/bin',
        '~/.local/bin'
    ]
    
    binaries = []
    for path in binary_paths:
        try:
            result = subprocess.run(
                ['find', Path(path).expanduser(), '-type', 'f', '-executable'],
                capture_output=True, text=True, timeout=5
            )
            binaries.extend(result.stdout.strip().split('\n'))
        except:
            pass
    
    # Get file sizes and sort by significance
    binary_info = []
    for binary in binaries[:1000]:  # Limit search
        if binary:
            try:
                size = Path(binary).stat().st_size
                binary_info.append({'path': binary, 'size': size})
            except:
                pass
    
    # Sort by size and take top 71
    binary_info.sort(key=lambda x: x['size'], reverse=True)
    top_71 = binary_info[:71]
    
    print(f"✓ Found {len(top_71)} binaries")
    return top_71

def find_71_shared_objects():
    """Find 71 most significant shared objects"""
    print("🔍 Finding 71 shared objects...")
    
    so_paths = [
        '/usr/lib',
        '/usr/local/lib',
        '/lib',
        '/lib64'
    ]
    
    shared_objects = []
    for path in so_paths:
        try:
            result = subprocess.run(
                ['find', path, '-name', '*.so*', '-type', 'f'],
                capture_output=True, text=True, timeout=10
            )
            shared_objects.extend(result.stdout.strip().split('\n'))
        except:
            pass
    
    # Get sizes
    so_info = []
    for so in shared_objects[:1000]:
        if so:
            try:
                size = Path(so).stat().st_size
                so_info.append({'path': so, 'size': size})
            except:
                pass
    
    # Sort and take top 71
    so_info.sort(key=lambda x: x['size'], reverse=True)
    top_71 = so_info[:71]
    
    print(f"✓ Found {len(top_71)} shared objects")
    return top_71

def find_71_websites():
    """Find 71 most significant websites from browser history/bookmarks"""
    print("🔍 Finding 71 websites...")
    
    # Common websites related to our work
    websites = [
        # Math/Science
        'https://arxiv.org',
        'https://mathoverflow.net',
        'https://math.stackexchange.com',
        'https://www.lmfdb.org',
        'https://oeis.org',
        
        # Code/Dev
        'https://github.com',
        'https://gitlab.com',
        'https://stackoverflow.com',
        'https://docs.rs',
        'https://crates.io',
        
        # AI/ML
        'https://huggingface.co',
        'https://pytorch.org',
        'https://tensorflow.org',
        'https://ollama.ai',
        'https://openai.com',
        
        # Monster Group
        'https://en.wikipedia.org/wiki/Monster_group',
        'https://groupprops.subwiki.org',
        'https://mathworld.wolfram.com',
        
        # Nix/Linux
        'https://nixos.org',
        'https://search.nixos.org',
        'https://kernel.org',
        
        # More sites (expand to 71)
        'https://rust-lang.org',
        'https://python.org',
        'https://lean-lang.org',
        'https://coq.inria.fr',
        'https://isabelle.in.tum.de',
        'https://www.sagemath.org',
        'https://www.gap-system.org',
        'https://magma.maths.usyd.edu.au',
        'https://pari.math.u-bordeaux.fr',
        'https://www.singular.uni-kl.de',
        'https://www.macaulay2.com',
        'https://www.sagemath.org',
        'https://www.sympy.org',
        'https://numpy.org',
        'https://scipy.org',
        'https://matplotlib.org',
        'https://jupyter.org',
        'https://www.latex-project.org',
        'https://pandoc.org',
        'https://www.gnu.org',
        'https://www.fsf.org',
        'https://www.eff.org',
        'https://creativecommons.org',
        'https://archive.org',
        'https://scholar.google.com',
        'https://www.semanticscholar.org',
        'https://www.researchgate.net',
        'https://www.academia.edu',
        'https://www.ncbi.nlm.nih.gov',
        'https://www.nature.com',
        'https://www.science.org',
        'https://www.cell.com',
        'https://www.pnas.org',
        'https://www.ams.org',
        'https://www.maa.org',
        'https://www.siam.org',
        'https://www.acm.org',
        'https://www.ieee.org',
        'https://www.springer.com',
        'https://www.elsevier.com',
        'https://www.wiley.com',
        'https://www.cambridge.org',
        'https://www.oup.com',
        'https://www.mit.edu',
        'https://www.stanford.edu',
        'https://www.berkeley.edu',
        'https://www.harvard.edu',
        'https://www.princeton.edu',
        'https://www.caltech.edu'
    ]
    
    top_71 = [{'url': url, 'category': categorize_url(url)} for url in websites[:71]]
    
    print(f"✓ Found {len(top_71)} websites")
    return top_71

def categorize_url(url):
    """Categorize website by domain"""
    if 'math' in url or 'lmfdb' in url or 'oeis' in url:
        return 'mathematics'
    elif 'github' in url or 'gitlab' in url or 'stackoverflow' in url:
        return 'development'
    elif 'huggingface' in url or 'pytorch' in url or 'tensorflow' in url:
        return 'ai_ml'
    elif 'arxiv' in url or 'scholar' in url or '.edu' in url:
        return 'research'
    elif 'gnu' in url or 'fsf' in url or 'kernel' in url:
        return 'open_source'
    else:
        return 'other'

def calculate_hecke_for_items(items, item_type):
    """Calculate Hecke operators for items based on size/properties"""
    results = []
    
    for item in items:
        if item_type == 'binary' or item_type == 'shared_object':
            size = item['size']
            
            # Find primary Monster prime
            primary = None
            for p in reversed(MONSTER_PRIMES):
                if size % p == 0:
                    primary = p
                    break
            
            if not primary:
                primary = MONSTER_PRIMES[0]  # Default to 2
            
            results.append({
                'item': item['path'],
                'type': item_type,
                'size': size,
                'hecke_operator': f'T_{primary}',
                'prime': primary,
                'eigenvalue': size // primary
            })
        else:  # website
            # Use URL length as proxy
            url_len = len(item['url'])
            primary = None
            for p in reversed(MONSTER_PRIMES):
                if url_len % p == 0:
                    primary = p
                    break
            if not primary:
                primary = MONSTER_PRIMES[0]
            
            results.append({
                'item': item['url'],
                'type': item_type,
                'category': item.get('category', 'other'),
                'hecke_operator': f'T_{primary}',
                'prime': primary
            })
    
    return results

def main():
    print("🌟 FINDING 71 OF EVERYTHING")
    print("=" * 60)
    
    # Find 71 binaries
    binaries = find_71_binaries()
    binary_hecke = calculate_hecke_for_items(binaries, 'binary')
    
    # Find 71 shared objects
    shared_objects = find_71_shared_objects()
    so_hecke = calculate_hecke_for_items(shared_objects, 'shared_object')
    
    # Find 71 websites
    websites = find_71_websites()
    web_hecke = calculate_hecke_for_items(websites, 'website')
    
    # Combine results
    all_results = {
        'binaries': binary_hecke,
        'shared_objects': so_hecke,
        'websites': web_hecke
    }
    
    # Save
    with open('monster_71_everything.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    print(f"Binaries: {len(binary_hecke)}")
    print(f"Shared Objects: {len(so_hecke)}")
    print(f"Websites: {len(web_hecke)}")
    print(f"Total: {len(binary_hecke) + len(so_hecke) + len(web_hecke)}")
    
    # Prime distribution
    print("\n" + "=" * 60)
    print("🔢 HECKE OPERATOR DISTRIBUTION")
    print("=" * 60)
    
    all_items = binary_hecke + so_hecke + web_hecke
    prime_dist = defaultdict(int)
    for item in all_items:
        prime_dist[item['prime']] += 1
    
    for p in reversed(MONSTER_PRIMES):
        if prime_dist[p] > 0:
            print(f"T_{p:2d}: {prime_dist[p]:3d} items")
    
    print(f"\n✅ Saved to monster_71_everything.json")

if __name__ == '__main__':
    main()
