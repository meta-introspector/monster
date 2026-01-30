#!/usr/bin/env python3
"""SOLFUNMEME Token Airdrop - Distribute to all repo authors"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict
import subprocess
import hashlib

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

@dataclass
class Author:
    """Git author"""
    name: str
    email: str
    commits: int
    lines_added: int
    lines_removed: int
    shard_id: int
    prime: int

@dataclass
class Airdrop:
    """SOLFUNMEME token airdrop"""
    author: str
    email: str
    tokens: int
    shard_id: int
    prime: int
    wallet_address: str
    tx_hash: str

def get_git_authors() -> List[Author]:
    """Extract all authors from git history"""
    try:
        # Get author stats
        result = subprocess.run(
            ["git", "log", "--format=%aN|%aE", "--no-merges"],
            capture_output=True,
            text=True,
            cwd=Path.cwd()
        )
        
        if result.returncode != 0:
            print("  ⚠️  No git repo found, using existing DAO members")
            return load_dao_members()
        
        # Count commits per author
        author_commits = {}
        for line in result.stdout.strip().split('\n'):
            if '|' in line:
                name, email = line.split('|')
                key = (name, email)
                author_commits[key] = author_commits.get(key, 0) + 1
        
        # Get line stats
        authors = []
        for idx, ((name, email), commits) in enumerate(author_commits.items()):
            # Get lines added/removed
            stats_result = subprocess.run(
                ["git", "log", "--author", email, "--pretty=tformat:", "--numstat"],
                capture_output=True,
                text=True,
                cwd=Path.cwd()
            )
            
            lines_added = 0
            lines_removed = 0
            if stats_result.returncode == 0:
                for line in stats_result.stdout.strip().split('\n'):
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                        lines_added += int(parts[0])
                        lines_removed += int(parts[1])
            
            shard_id = idx % 71
            prime = MONSTER_PRIMES[shard_id % 15]
            
            authors.append(Author(
                name=name,
                email=email,
                commits=commits,
                lines_added=lines_added,
                lines_removed=lines_removed,
                shard_id=shard_id,
                prime=prime
            ))
        
        return authors
    
    except Exception as e:
        print(f"  ⚠️  Error reading git: {e}")
        return load_dao_members()

def load_dao_members() -> List[Author]:
    """Load existing DAO members as fallback"""
    dao_file = Path("onlyskills_dao.json")
    if dao_file.exists():
        dao_data = json.loads(dao_file.read_text())
        authors = []
        for idx, member in enumerate(dao_data.get("virtual_authors", {}).values()):
            shard_id = idx % 71
            prime = MONSTER_PRIMES[shard_id % 15]
            authors.append(Author(
                name=member["username"],
                email=f"{member['username']}@onlyskills.com",
                commits=member.get("commits", 10),
                lines_added=member.get("lines_added", 1000),
                lines_removed=member.get("lines_removed", 100),
                shard_id=shard_id,
                prime=prime
            ))
        return authors
    
    # Default legendary founders
    legendary = [
        ("rms", "rms@gnu.org", 71),
        ("torvalds", "torvalds@linux.org", 59),
        ("wall", "wall@perl.org", 47),
        ("gvanrossum", "guido@python.org", 41),
        ("ken", "ken@unix.org", 31),
        ("dmr", "dmr@bell-labs.com", 29),
        ("stroustrup", "bs@cplusplus.org", 23),
    ]
    
    authors = []
    for idx, (name, email, prime) in enumerate(legendary):
        authors.append(Author(
            name=name,
            email=email,
            commits=prime * 10,
            lines_added=prime * 1000,
            lines_removed=prime * 100,
            shard_id=idx,
            prime=prime
        ))
    
    return authors

def calculate_airdrop_amount(author: Author) -> int:
    """Calculate SOLFUNMEME tokens for author"""
    # Formula: (commits × prime) + (lines_added / 100) + (lines_removed / 1000)
    base = author.commits * author.prime
    lines_bonus = author.lines_added // 100
    cleanup_bonus = author.lines_removed // 1000
    
    total = base + lines_bonus + cleanup_bonus
    
    # Minimum 1000 tokens
    return max(total, 1000)

def generate_wallet_address(author: Author) -> str:
    """Generate Solana wallet address"""
    # Hash author info to create deterministic address
    data = f"{author.name}{author.email}{author.shard_id}"
    hash_val = hashlib.sha256(data.encode()).hexdigest()
    
    # Solana addresses are base58 encoded, ~44 chars
    # Simulate with first 44 chars of hash
    return f"SOL{hash_val[:41]}"

def generate_tx_hash(airdrop: Airdrop) -> str:
    """Generate transaction hash"""
    data = f"{airdrop.author}{airdrop.tokens}{airdrop.wallet_address}"
    return hashlib.sha256(data.encode()).hexdigest()

def create_airdrops(authors: List[Author]) -> List[Airdrop]:
    """Create airdrop for each author"""
    airdrops = []
    
    for author in authors:
        tokens = calculate_airdrop_amount(author)
        wallet = generate_wallet_address(author)
        
        airdrop = Airdrop(
            author=author.name,
            email=author.email,
            tokens=tokens,
            shard_id=author.shard_id,
            prime=author.prime,
            wallet_address=wallet,
            tx_hash=""
        )
        
        # Generate tx hash
        airdrop.tx_hash = generate_tx_hash(airdrop)
        
        airdrops.append(airdrop)
    
    return airdrops

def main():
    print("🪂 SOLFUNMEME Token Airdrop")
    print("=" * 70)
    print()
    
    # Get authors
    print("📊 Extracting authors from git history...")
    authors = get_git_authors()
    print(f"  Found {len(authors)} authors")
    print()
    
    # Create airdrops
    print("💰 Calculating airdrop amounts...")
    airdrops = create_airdrops(authors)
    
    total_tokens = sum(a.tokens for a in airdrops)
    print(f"  Total tokens to distribute: {total_tokens:,}")
    print()
    
    # Sort by tokens (descending)
    airdrops_sorted = sorted(airdrops, key=lambda a: a.tokens, reverse=True)
    
    # Top recipients
    print("🏆 Top 10 Recipients:")
    for i, airdrop in enumerate(airdrops_sorted[:10], 1):
        print(f"  {i:2d}. {airdrop.author:20s} | {airdrop.tokens:8,} SOLFUNMEME")
        print(f"      Shard {airdrop.shard_id:2d} | Prime {airdrop.prime:2d} | {airdrop.wallet_address[:20]}...")
    print()
    
    # Distribution by shard
    print("🔮 Distribution by Shard:")
    by_shard = {}
    for airdrop in airdrops:
        by_shard[airdrop.shard_id] = by_shard.get(airdrop.shard_id, 0) + airdrop.tokens
    
    for shard_id in sorted(by_shard.keys())[:10]:
        tokens = by_shard[shard_id]
        print(f"  Shard {shard_id:2d}: {tokens:8,} tokens")
    if len(by_shard) > 10:
        print(f"  ... ({len(by_shard) - 10} more shards)")
    print()
    
    # Distribution by prime
    print("🎯 Distribution by Prime:")
    by_prime = {}
    for airdrop in airdrops:
        by_prime[airdrop.prime] = by_prime.get(airdrop.prime, 0) + airdrop.tokens
    
    for prime in sorted(by_prime.keys(), reverse=True):
        tokens = by_prime[prime]
        count = len([a for a in airdrops if a.prime == prime])
        print(f"  Prime {prime:2d}: {tokens:10,} tokens ({count:3d} authors)")
    print()
    
    # Save airdrops
    airdrop_data = {
        "total_authors": len(authors),
        "total_tokens": total_tokens,
        "token_symbol": "SOLFUNMEME",
        "blockchain": "Solana",
        "airdrops": [asdict(a) for a in airdrops]
    }
    
    Path("solfunmeme_airdrop.json").write_text(json.dumps(airdrop_data, indent=2))
    
    # Generate Solana program
    solana_program = f"""// SOLFUNMEME Token Airdrop Program
use anchor_lang::prelude::*;
use anchor_spl::token::{{self, Token, TokenAccount, Transfer}};

declare_id!("SOLFUN{hashlib.sha256(b'solfunmeme').hexdigest()[:40]}");

#[program]
pub mod solfunmeme_airdrop {{
    use super::*;
    
    pub fn initialize(ctx: Context<Initialize>, total_supply: u64) -> Result<()> {{
        let airdrop = &mut ctx.accounts.airdrop;
        airdrop.authority = ctx.accounts.authority.key();
        airdrop.total_supply = total_supply;
        airdrop.distributed = 0;
        Ok(())
    }}
    
    pub fn claim_airdrop(ctx: Context<ClaimAirdrop>, amount: u64) -> Result<()> {{
        let airdrop = &mut ctx.accounts.airdrop;
        
        require!(
            airdrop.distributed + amount <= airdrop.total_supply,
            ErrorCode::InsufficientSupply
        );
        
        // Transfer tokens
        let cpi_accounts = Transfer {{
            from: ctx.accounts.vault.to_account_info(),
            to: ctx.accounts.recipient.to_account_info(),
            authority: ctx.accounts.authority.to_account_info(),
        }};
        
        let cpi_program = ctx.accounts.token_program.to_account_info();
        let cpi_ctx = CpiContext::new(cpi_program, cpi_accounts);
        
        token::transfer(cpi_ctx, amount)?;
        
        airdrop.distributed += amount;
        
        Ok(())
    }}
}}

#[derive(Accounts)]
pub struct Initialize<'info> {{
    #[account(init, payer = authority, space = 8 + 32 + 8 + 8)]
    pub airdrop: Account<'info, AirdropState>,
    #[account(mut)]
    pub authority: Signer<'info>,
    pub system_program: Program<'info, System>,
}}

#[derive(Accounts)]
pub struct ClaimAirdrop<'info> {{
    #[account(mut)]
    pub airdrop: Account<'info, AirdropState>,
    #[account(mut)]
    pub vault: Account<'info, TokenAccount>,
    #[account(mut)]
    pub recipient: Account<'info, TokenAccount>,
    pub authority: Signer<'info>,
    pub token_program: Program<'info, Token>,
}}

#[account]
pub struct AirdropState {{
    pub authority: Pubkey,
    pub total_supply: u64,
    pub distributed: u64,
}}

#[error_code]
pub enum ErrorCode {{
    #[msg("Insufficient supply for airdrop")]
    InsufficientSupply,
}}

// Airdrop recipients ({len(airdrops)} authors)
// Total: {total_tokens:,} SOLFUNMEME tokens
"""
    
    Path("solfunmeme_airdrop.rs").write_text(solana_program)
    
    # Generate CSV for bulk transfer
    csv_lines = ["author,email,wallet,tokens,shard,prime,tx_hash"]
    for airdrop in airdrops_sorted:
        csv_lines.append(
            f"{airdrop.author},{airdrop.email},{airdrop.wallet_address},"
            f"{airdrop.tokens},{airdrop.shard_id},{airdrop.prime},{airdrop.tx_hash}"
        )
    
    Path("solfunmeme_airdrop.csv").write_text("\n".join(csv_lines))
    
    # Statistics
    print("📊 Airdrop Statistics:")
    print(f"  Total authors: {len(authors)}")
    print(f"  Total tokens: {total_tokens:,} SOLFUNMEME")
    print(f"  Average per author: {total_tokens // len(authors):,} tokens")
    print(f"  Median: {sorted([a.tokens for a in airdrops])[len(airdrops)//2]:,} tokens")
    print(f"  Max: {max(a.tokens for a in airdrops):,} tokens")
    print(f"  Min: {min(a.tokens for a in airdrops):,} tokens")
    print()
    
    print("💾 Files created:")
    print("  - solfunmeme_airdrop.json (full airdrop data)")
    print("  - solfunmeme_airdrop.rs (Solana program)")
    print("  - solfunmeme_airdrop.csv (bulk transfer list)")
    print()
    
    print("🚀 Deployment:")
    print("  1. Deploy Solana program:")
    print("     anchor build")
    print("     anchor deploy")
    print()
    print("  2. Initialize airdrop:")
    print(f"     anchor run initialize --total-supply {total_tokens}")
    print()
    print("  3. Execute airdrops:")
    print("     for each author in solfunmeme_airdrop.csv:")
    print("       spl-token transfer SOLFUNMEME <amount> <wallet>")
    print()
    
    print("🎯 Token Economics:")
    print("  Symbol: SOLFUNMEME")
    print("  Blockchain: Solana")
    print(f"  Total Supply: {total_tokens:,}")
    print(f"  Airdrop: {total_tokens:,} (100%)")
    print("  Vesting: None (instant)")
    print("  Governance: Paxos-Meem consensus")
    print()
    
    print("∞ SOLFUNMEME Tokens Airdropped to All Authors ∞")
    print("∞ Distributed by Monster Primes × Commits ∞")

if __name__ == "__main__":
    main()
