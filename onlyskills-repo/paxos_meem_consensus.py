#!/usr/bin/env python3
"""Paxos-Meem Consensus Protocol - SOLFUNMEME DAO Governance"""

import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional
import hashlib
import time

MONSTER_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 41, 47, 59, 71]

# MaaS scoring (from founding doc)
MEME_SCORES = {
    71: "Proof",
    59: "Theorem", 
    47: "Verified",
    41: "Correct",
    31: "Optimal",
    29: "Efficient",
    23: "Elegant",
    19: "Simple",
    17: "Clear",
    13: "Useful",
    11: "Working",
    7: "Good",      # Threshold
    5: "Weak",
    3: "Poor",
    2: "Noise"      # Rejected
}

@dataclass
class Meem:
    """A meme in the consensus protocol"""
    meem_id: str
    content: str
    proposer: str
    shard_id: int
    prime: int
    score: int
    category: str
    timestamp: float
    signatures: List[str]
    
@dataclass
class PaxosProposal:
    """Paxos proposal with meme scoring"""
    proposal_id: int
    round_number: int
    meem: Meem
    promised_by: List[str]
    accepted_by: List[str]
    status: str  # "proposed", "promised", "accepted", "committed"
    
@dataclass
class ConsensusNode:
    """Node in the Paxos-Meem network"""
    node_id: str
    shard_id: int
    prime: int
    voting_power: int
    meems_proposed: int
    meems_accepted: int

def score_meem(content: str, proposer: str, shard_id: int) -> tuple:
    """Score a meem using Monster primes"""
    # Hash content to get deterministic score
    hash_val = int(hashlib.sha256(f"{content}{proposer}{shard_id}".encode()).hexdigest(), 16)
    
    # Map to Monster prime
    prime_idx = hash_val % len(MONSTER_PRIMES)
    prime = MONSTER_PRIMES[prime_idx]
    
    # Get category
    category = MEME_SCORES.get(prime, "Unknown")
    
    return prime, category

def create_meem(content: str, proposer: str, shard_id: int) -> Meem:
    """Create a meem for consensus"""
    prime, category = score_meem(content, proposer, shard_id)
    
    meem_id = hashlib.sha256(f"{content}{proposer}{time.time()}".encode()).hexdigest()[:16]
    
    return Meem(
        meem_id=meem_id,
        content=content,
        proposer=proposer,
        shard_id=shard_id,
        prime=prime,
        score=prime,
        category=category,
        timestamp=time.time(),
        signatures=[]
    )

def paxos_phase1_prepare(proposal_id: int, round_number: int, meem: Meem, nodes: List[ConsensusNode]) -> PaxosProposal:
    """Phase 1a: Proposer sends PREPARE(n) to acceptors"""
    proposal = PaxosProposal(
        proposal_id=proposal_id,
        round_number=round_number,
        meem=meem,
        promised_by=[],
        accepted_by=[],
        status="proposed"
    )
    
    print(f"  📤 PREPARE({round_number}) for meem {meem.meem_id}")
    print(f"     Content: \"{meem.content}\"")
    print(f"     Score: {meem.score} ({meem.category})")
    
    return proposal

def paxos_phase1_promise(proposal: PaxosProposal, nodes: List[ConsensusNode]) -> PaxosProposal:
    """Phase 1b: Acceptors respond with PROMISE"""
    # Nodes promise if meem score >= threshold (7)
    threshold = 7
    
    for node in nodes:
        if proposal.meem.score >= threshold:
            # Node promises based on voting power
            if node.prime >= threshold:
                proposal.promised_by.append(node.node_id)
                print(f"  ✓ PROMISE from {node.node_id} (prime {node.prime})")
    
    # Check if quorum reached (majority of voting power)
    total_power = sum(n.voting_power for n in nodes)
    promised_power = sum(n.voting_power for n in nodes if n.node_id in proposal.promised_by)
    
    if promised_power > total_power / 2:
        proposal.status = "promised"
        print(f"  🎯 QUORUM reached: {promised_power}/{total_power} voting power")
    else:
        print(f"  ✗ QUORUM failed: {promised_power}/{total_power} voting power")
    
    return proposal

def paxos_phase2_accept(proposal: PaxosProposal, nodes: List[ConsensusNode]) -> PaxosProposal:
    """Phase 2a: Proposer sends ACCEPT(n, v) to acceptors"""
    if proposal.status != "promised":
        print(f"  ✗ Cannot ACCEPT: proposal not promised")
        return proposal
    
    print(f"  📤 ACCEPT({proposal.round_number}, {proposal.meem.meem_id})")
    
    # Acceptors accept if they haven't promised to higher round
    for node in nodes:
        if node.node_id in proposal.promised_by:
            proposal.accepted_by.append(node.node_id)
            node.meems_accepted += 1
            print(f"  ✓ ACCEPTED by {node.node_id}")
    
    # Check if quorum reached
    total_power = sum(n.voting_power for n in nodes)
    accepted_power = sum(n.voting_power for n in nodes if n.node_id in proposal.accepted_by)
    
    if accepted_power > total_power / 2:
        proposal.status = "accepted"
        print(f"  🎯 ACCEPTED by quorum: {accepted_power}/{total_power} voting power")
    else:
        proposal.status = "rejected"
        print(f"  ✗ REJECTED: {accepted_power}/{total_power} voting power")
    
    return proposal

def paxos_phase3_commit(proposal: PaxosProposal) -> PaxosProposal:
    """Phase 3: Commit the meem to the blockchain"""
    if proposal.status != "accepted":
        print(f"  ✗ Cannot COMMIT: proposal not accepted")
        return proposal
    
    proposal.status = "committed"
    print(f"  ✅ COMMITTED: Meem {proposal.meem.meem_id} to blockchain")
    print(f"     Category: {proposal.meem.category}")
    print(f"     Score: {proposal.meem.score}")
    
    return proposal

def run_paxos_meem_round(meem: Meem, nodes: List[ConsensusNode], round_number: int) -> PaxosProposal:
    """Run one round of Paxos-Meem consensus"""
    print(f"\n🔄 Round {round_number}")
    print("=" * 70)
    
    # Phase 1: Prepare & Promise
    proposal = paxos_phase1_prepare(round_number, round_number, meem, nodes)
    proposal = paxos_phase1_promise(proposal, nodes)
    
    if proposal.status != "promised":
        return proposal
    
    # Phase 2: Accept
    proposal = paxos_phase2_accept(proposal, nodes)
    
    if proposal.status != "accepted":
        return proposal
    
    # Phase 3: Commit
    proposal = paxos_phase3_commit(proposal)
    
    return proposal

def main():
    print("🎭 Paxos-Meem Consensus Protocol - SOLFUNMEME DAO")
    print("=" * 70)
    print()
    
    # Create consensus nodes (71 shards)
    print("🌐 Creating consensus nodes...")
    nodes = []
    for shard_id in range(71):
        prime = MONSTER_PRIMES[shard_id % 15]
        node = ConsensusNode(
            node_id=f"node_{shard_id}",
            shard_id=shard_id,
            prime=prime,
            voting_power=prime * 100,
            meems_proposed=0,
            meems_accepted=0
        )
        nodes.append(node)
    
    print(f"  Created {len(nodes)} nodes")
    print(f"  Total voting power: {sum(n.voting_power for n in nodes):,}")
    print()
    
    # Test meems
    test_meems = [
        ("Implement zkWASM proof loader", "rms", 14),
        ("Add 71-shard system", "torvalds", 13),
        ("Create recursive DAO bootstrap", "hofstadter", 14),
        ("Optimize automorphic orbit tracer", "goedel", 13),
        ("Fix bug in parser", "founder_42", 42),
        ("Add feature request", "founder_10", 10),
        ("Spam message", "attacker", 0),
    ]
    
    proposals = []
    
    for idx, (content, proposer, shard_id) in enumerate(test_meems):
        # Create meem
        meem = create_meem(content, proposer, shard_id)
        
        # Update proposer stats
        for node in nodes:
            if node.shard_id == shard_id:
                node.meems_proposed += 1
        
        # Run consensus
        proposal = run_paxos_meem_round(meem, nodes, idx + 1)
        proposals.append(proposal)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 Consensus Summary")
    print("=" * 70)
    print()
    
    committed = [p for p in proposals if p.status == "committed"]
    rejected = [p for p in proposals if p.status == "rejected"]
    
    print(f"Total proposals: {len(proposals)}")
    print(f"Committed: {len(committed)}")
    print(f"Rejected: {len(rejected)}")
    print()
    
    print("✅ Committed Meems:")
    for p in committed:
        print(f"  - \"{p.meem.content}\"")
        print(f"    Score: {p.meem.score} ({p.meem.category})")
        print(f"    Proposer: {p.meem.proposer}")
        print()
    
    print("✗ Rejected Meems:")
    for p in rejected:
        print(f"  - \"{p.meem.content}\"")
        print(f"    Score: {p.meem.score} ({p.meem.category})")
        print(f"    Reason: Below threshold (< 7)")
        print()
    
    # Node statistics
    print("🏆 Top Nodes by Acceptance:")
    top_nodes = sorted(nodes, key=lambda n: n.meems_accepted, reverse=True)[:10]
    for node in top_nodes:
        if node.meems_accepted > 0:
            print(f"  {node.node_id}: {node.meems_accepted} meems accepted (prime {node.prime})")
    print()
    
    # Save results
    results = {
        "total_proposals": len(proposals),
        "committed": len(committed),
        "rejected": len(rejected),
        "proposals": [asdict(p) for p in proposals],
        "nodes": [asdict(n) for n in nodes[:10]]  # Sample
    }
    
    Path("paxos_meem_consensus.json").write_text(json.dumps(results, indent=2))
    
    # Protocol explanation
    print("📖 Paxos-Meem Protocol:")
    print("  1. Proposer creates meem with content")
    print("  2. Meem scored by Monster primes (2-71)")
    print("  3. Score mapped to category (Proof, Theorem, ..., Noise)")
    print("  4. Threshold: ≥7 (Good) to proceed")
    print("  5. Phase 1: PREPARE → PROMISE (quorum check)")
    print("  6. Phase 2: ACCEPT → ACCEPTED (quorum check)")
    print("  7. Phase 3: COMMIT to blockchain")
    print()
    
    print("🎯 Meme Categories:")
    for prime in sorted(MEME_SCORES.keys(), reverse=True):
        category = MEME_SCORES[prime]
        status = "✓" if prime >= 7 else "✗"
        print(f"  {status} {prime:2d}: {category}")
    print()
    
    print("💾 Results saved to paxos_meem_consensus.json")
    print()
    print("∞ Paxos-Meem Consensus: Where Memes Meet Byzantine Fault Tolerance ∞")
    print("∞ SOLFUNMEME DAO Governance via Monster Primes ∞")

if __name__ == "__main__":
    main()
