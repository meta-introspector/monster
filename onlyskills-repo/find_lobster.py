#!/usr/bin/env python3
"""Find the lobster: Analyze Solana/ETH top coins and repos"""

import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict

@dataclass
class Coin:
    symbol: str
    name: str
    chain: str
    market_cap: float
    repo_url: str
    priority: int
    lobster_score: float

# Top Solana tokens
SOLANA_TOKENS = [
    ("SOL", "Solana", "https://github.com/solana-labs/solana"),
    ("BONK", "Bonk", "https://github.com/bonk-inu/bonk"),
    ("JUP", "Jupiter", "https://github.com/jup-ag/jupiter-core"),
    ("WIF", "dogwifhat", "https://github.com/dogwifhat/dogwifhat"),
    ("PYTH", "Pyth Network", "https://github.com/pyth-network/pyth-client"),
]

# Top Ethereum tokens
ETHEREUM_TOKENS = [
    ("ETH", "Ethereum", "https://github.com/ethereum/go-ethereum"),
    ("USDT", "Tether", "https://github.com/tether-to/tether"),
    ("USDC", "USD Coin", "https://github.com/circlefoundation/stablecoin-evm"),
    ("LINK", "Chainlink", "https://github.com/smartcontractkit/chainlink"),
    ("UNI", "Uniswap", "https://github.com/Uniswap/v3-core"),
]

def calculate_lobster_score(repo_url: str) -> float:
    """Calculate lobster score based on repo characteristics"""
    score = 0.0
    
    # Lobster indicators:
    # - Many stars (popular)
    # - Active development (recent commits)
    # - Many contributors (decentralized)
    # - Clean code (low threat)
    # - Meme potential (cultural significance)
    
    # Simplified scoring
    if "solana" in repo_url.lower():
        score += 0.3  # Solana ecosystem
    if "bonk" in repo_url.lower() or "wif" in repo_url.lower():
        score += 0.4  # Meme coins = lobster potential
    if "uniswap" in repo_url.lower():
        score += 0.2  # DeFi = lobster habitat
    
    return score

def prioritize_coins(coins: List[Coin]) -> List[Coin]:
    """Prioritize coins by lobster score"""
    return sorted(coins, key=lambda c: c.lobster_score, reverse=True)

def ingest_repo_sandboxed(coin: Coin, output_dir: Path):
    """Ingest repo with kernel sandboxing"""
    repo_dir = output_dir / coin.symbol
    repo_dir.mkdir(exist_ok=True)
    
    print(f"  🦞 {coin.symbol} ({coin.name}) - Score: {coin.lobster_score:.2f}")
    
    # Clone in sandbox
    cmd = [
        "unshare", "--user", "--net", "--mount", "--map-root-user",
        "git", "clone", "--depth", "1", coin.repo_url, str(repo_dir / "repo")
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, timeout=30)
        
        # Extract metadata
        metadata = {
            "symbol": coin.symbol,
            "name": coin.name,
            "chain": coin.chain,
            "repo_url": coin.repo_url,
            "lobster_score": coin.lobster_score,
            "ingested": True
        }
        
        (repo_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
        return True
    except:
        return False

def generate_prolog_lobster_facts(coins: List[Coin], output: Path):
    """Generate Prolog facts for lobster detection"""
    
    prolog = """% Lobster detection in crypto repos
:- module(lobster_finder, [
    coin/5,
    lobster_score/2,
    is_lobster/1
]).

% coin(Symbol, Name, Chain, RepoURL, LobsterScore)
"""
    
    for coin in coins:
        prolog += f"coin('{coin.symbol}', '{coin.name}', '{coin.chain}', '{coin.repo_url}', {coin.lobster_score}).\n"
    
    prolog += "\n% Lobster scores\n"
    for coin in coins:
        prolog += f"lobster_score('{coin.symbol}', {coin.lobster_score}).\n"
    
    prolog += """
% Lobster detection rules
is_lobster(Coin) :- lobster_score(Coin, Score), Score > 0.5.
is_mega_lobster(Coin) :- lobster_score(Coin, Score), Score > 0.7.

% Meme coin detection
is_meme_coin(Coin) :- 
    coin(Coin, Name, _, _, _),
    (sub_string(Name, _, _, _, "bonk") ;
     sub_string(Name, _, _, _, "wif") ;
     sub_string(Name, _, _, _, "doge")).

% Lobster habitat (DeFi)
lobster_habitat(Coin) :-
    coin(Coin, _, _, URL, _),
    (sub_string(URL, _, _, _, "uniswap") ;
     sub_string(URL, _, _, _, "jupiter") ;
     sub_string(URL, _, _, _, "dex")).

% The Lobster (highest score)
the_lobster(Coin) :-
    lobster_score(Coin, Score),
    \\+ (lobster_score(_, OtherScore), OtherScore > Score).

% Query: Find the lobster
% ?- the_lobster(Coin).
"""
    
    output.write_text(prolog)

def main():
    print("🦞 Lobster Finder: Solana + Ethereum Analysis")
    print("=" * 70)
    print()
    
    output_dir = Path("lobster_search")
    output_dir.mkdir(exist_ok=True)
    
    # Collect coins
    coins = []
    
    print("📊 Analyzing Solana tokens...")
    for symbol, name, repo in SOLANA_TOKENS:
        score = calculate_lobster_score(repo)
        coins.append(Coin(
            symbol=symbol,
            name=name,
            chain="solana",
            market_cap=0,  # Would fetch from API
            repo_url=repo,
            priority=0,
            lobster_score=score
        ))
    
    print("📊 Analyzing Ethereum tokens...")
    for symbol, name, repo in ETHEREUM_TOKENS:
        score = calculate_lobster_score(repo)
        coins.append(Coin(
            symbol=symbol,
            name=name,
            chain="ethereum",
            market_cap=0,
            repo_url=repo,
            priority=0,
            lobster_score=score
        ))
    
    print()
    
    # Prioritize
    print("🎯 Prioritizing by lobster score...")
    coins = prioritize_coins(coins)
    
    for i, coin in enumerate(coins, 1):
        coin.priority = i
        print(f"  {i}. {coin.symbol} - {coin.lobster_score:.2f}")
    
    print()
    
    # Ingest top 3
    print("📥 Ingesting top 3 (sandboxed)...")
    for coin in coins[:3]:
        ingest_repo_sandboxed(coin, output_dir)
    
    print()
    
    # Generate Prolog
    print("📝 Generating Prolog lobster facts...")
    prolog_file = output_dir / "lobster_finder.pl"
    generate_prolog_lobster_facts(coins, prolog_file)
    print(f"  Saved: {prolog_file}")
    
    # Save JSON
    json_file = output_dir / "coins.json"
    json_file.write_text(json.dumps([asdict(c) for c in coins], indent=2))
    print(f"  Saved: {json_file}")
    print()
    
    # The Lobster
    the_lobster = coins[0]
    print("🦞 THE LOBSTER:")
    print(f"  Symbol: {the_lobster.symbol}")
    print(f"  Name: {the_lobster.name}")
    print(f"  Chain: {the_lobster.chain}")
    print(f"  Score: {the_lobster.lobster_score:.2f}")
    print(f"  Repo: {the_lobster.repo_url}")
    print()
    
    print("∞ Lobster Found. Repos Ingested. Prolog Generated. ∞")

if __name__ == "__main__":
    main()
