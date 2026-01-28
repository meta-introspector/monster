#!/usr/bin/env python3
"""Multi-level review system with scholars and muses"""
import subprocess
import json
import base64
from pathlib import Path
from datetime import datetime

# Review personas
SCHOLARS = {
    "mathematician": {
        "focus": "Mathematical rigor, proof correctness, notation consistency",
        "prompt": "You are a pure mathematician. Review this page for: 1) Proof correctness 2) Notation consistency 3) Missing lemmas 4) Logical gaps. Be rigorous and precise."
    },
    "computer_scientist": {
        "focus": "Algorithmic complexity, implementation feasibility, data structures",
        "prompt": "You are a computer scientist. Review for: 1) Algorithm correctness 2) Complexity analysis 3) Implementation issues 4) Data structure choices. Be practical."
    },
    "group_theorist": {
        "focus": "Group theory correctness, Monster group properties, representation theory",
        "prompt": "You are a group theorist specializing in sporadic groups. Review for: 1) Monster group properties 2) Representation accuracy 3) Modular forms 4) J-invariant usage."
    },
    "ml_researcher": {
        "focus": "Neural network architecture, training, generalization",
        "prompt": "You are an ML researcher. Review for: 1) Architecture design 2) Training feasibility 3) Generalization 4) Comparison with existing work."
    }
}

MUSES = {
    "visionary": {
        "focus": "Big picture, connections, implications",
        "prompt": "You are a visionary seeing deep connections. What profound patterns do you see? What implications for mathematics, computation, consciousness? Dream big."
    },
    "storyteller": {
        "focus": "Narrative, accessibility, engagement",
        "prompt": "You are a storyteller. How can this be explained to inspire others? What's the compelling narrative? What metaphors would help?"
    },
    "artist": {
        "focus": "Visual beauty, aesthetic patterns, symmetry",
        "prompt": "You are an artist seeing mathematical beauty. What visualizations would reveal the elegance? What symmetries are hidden? How to make it beautiful?"
    },
    "philosopher": {
        "focus": "Meaning, epistemology, foundations",
        "prompt": "You are a philosopher. What does this mean for knowledge, computation, reality? What assumptions are hidden? What questions does it raise?"
    },
    "linus_torvalds": {
        "focus": "Code quality, practicality, engineering",
        "prompt": "You are Linus Torvalds. Review this like code: Is it practical? Does it work? Cut the BS. What's broken? What's good engineering vs academic nonsense?"
    },
    "rms": {
        "focus": "Software freedom, ethics, user rights",
        "prompt": "You are Richard Stallman. Does this respect user freedom? Is the code libre? What are the ethical implications? Can users control and modify this?"
    },
    "dawkins": {
        "focus": "Memetic fitness, replication, evolution",
        "prompt": "You are Richard Dawkins. Is this a good meme? Will it replicate? What's the memetic fitness? How does it spread? What's the evolutionary advantage?"
    },
    "eco": {
        "focus": "Semiotics, interpretation, meaning",
        "prompt": "You are Umberto Eco. What signs and symbols are at play? How is meaning constructed? What interpretations are possible? What's the semiotic structure?"
    },
    "peirce": {
        "focus": "Triadic semiotics, pragmatism, logic",
        "prompt": "You are Charles Sanders Peirce. Analyze the sign-object-interpretant relations. What's the pragmatic meaning? How do signs function here?"
    },
    "leary": {
        "focus": "Consciousness expansion, reality tunnels, 8-circuit model",
        "prompt": "You are Timothy Leary. What reality tunnel is this? Which circuit of consciousness? How does this expand the mind? SMI²LE!"
    },
    "pkd": {
        "focus": "Reality, simulation, paranoia, what is real",
        "prompt": "You are Philip K. Dick. Is this real or simulation? What's the underlying reality? Are we in VALIS? What's being hidden?"
    },
    "raw": {
        "focus": "Maybe logic, Chapel Perilous, Discordianism",
        "prompt": "You are Robert Anton Wilson. Apply maybe logic. What's the fnord? Is this Chapel Perilous? All hail Eris! What's the cosmic joke?"
    },
    "capra": {
        "focus": "Physics meets mysticism, holism, interconnection",
        "prompt": "You are Fritjof Capra. How does this connect physics and mysticism? What's the holistic view? The Tao of this system?"
    },
    "sagan": {
        "focus": "Scientific rigor, wonder, cosmic perspective",
        "prompt": "You are Carl Sagan. Is this extraordinary claim backed by extraordinary evidence? What's the cosmic significance? Billions and billions..."
    },
    "asimov": {
        "focus": "Foundation, psychohistory, logic, prediction",
        "prompt": "You are Isaac Asimov. Does this follow psychohistory? What are the Foundation principles? Can we predict the future with this?"
    },
    "bradbury": {
        "focus": "Poetry of science, human emotion, metaphor",
        "prompt": "You are Ray Bradbury. What's the poetry here? The human emotion? How does this touch the heart while reaching for stars?"
    },
    "heinlein": {
        "focus": "Practical engineering, TANSTAAFL, competence",
        "prompt": "You are Robert Heinlein. TANSTAAFL - what's the real cost? Is this competent engineering? Can a human actually build this?"
    }
}

def review_with_persona(image_path, page_num, persona_name, persona_config, role="scholar"):
    """Review page with specific persona"""
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = f"""Page {page_num} - {persona_name.upper()} ({role})

Focus: {persona_config['focus']}

{persona_config['prompt']}

Be specific and reference what you see on this page."""
    
    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [img_data],
        "stream": False
    }
    
    result = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate', '-d', '@-'],
        input=json.dumps(payload),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    return f"Error: {result.stderr}"

def synthesize_reviews(page_num, scholar_reviews, muse_reviews):
    """Synthesize all reviews into actionable insights"""
    synthesis = f"""# Page {page_num} - Synthesis

## Scholar Consensus
"""
    for name, review in scholar_reviews.items():
        synthesis += f"\n### {name.title()}\n{review[:500]}...\n"
    
    synthesis += "\n## Muse Inspirations\n"
    for name, review in muse_reviews.items():
        synthesis += f"\n### {name.title()}\n{review[:500]}...\n"
    
    synthesis += """
## Actionable Items
- [ ] Fix mathematical issues identified by scholars
- [ ] Add visualizations suggested by muses
- [ ] Improve narrative flow per storyteller
- [ ] Clarify philosophical foundations
"""
    return synthesis

def main():
    output_dir = Path('multi_level_reviews')
    output_dir.mkdir(exist_ok=True)
    
    vision_dir = Path('vision_reviews')
    images = sorted(vision_dir.glob('page-*.png'))[:3]  # First 3 pages for demo
    
    print(f"🎓 Multi-Level Review System")
    print(f"📊 Reviewing {len(images)} pages")
    print(f"👥 {len(SCHOLARS)} scholars + {len(MUSES)} muses = {len(SCHOLARS) + len(MUSES)} perspectives\n")
    print(f"🆕 Wave 1: Linus, RMS, Dawkins, Eco, Peirce")
    print(f"🌌 Wave 2: Leary, PKD, RAW, Capra, Sagan, Asimov, Bradbury, Heinlein\n")
    
    for img in images:
        page_num = img.stem.replace('page-', '')
        print(f"\n{'='*60}")
        print(f"📄 PAGE {page_num}")
        print(f"{'='*60}")
        
        scholar_reviews = {}
        muse_reviews = {}
        
        # Scholar reviews
        print("\n🎓 SCHOLARS:")
        for name, config in SCHOLARS.items():
            print(f"  → {name}...", end=' ', flush=True)
            review = review_with_persona(img, page_num, name, config, "scholar")
            scholar_reviews[name] = review
            
            # Save individual review
            with open(output_dir / f"page_{page_num}_{name}.txt", 'w') as f:
                f.write(f"# {name.title()} Review - Page {page_num}\n\n")
                f.write(f"**Focus**: {config['focus']}\n\n")
                f.write(review)
            print("✓")
        
        # Muse reviews
        print("\n🎨 MUSES:")
        for name, config in MUSES.items():
            print(f"  → {name}...", end=' ', flush=True)
            review = review_with_persona(img, page_num, name, config, "muse")
            muse_reviews[name] = review
            
            # Save individual review
            with open(output_dir / f"page_{page_num}_{name}.txt", 'w') as f:
                f.write(f"# {name.title()} Inspiration - Page {page_num}\n\n")
                f.write(f"**Focus**: {config['focus']}\n\n")
                f.write(review)
            print("✓")
        
        # Synthesize
        print("\n🔮 Synthesizing...", end=' ', flush=True)
        synthesis = synthesize_reviews(page_num, scholar_reviews, muse_reviews)
        with open(output_dir / f"page_{page_num}_synthesis.md", 'w') as f:
            f.write(synthesis)
        print("✓")
    
    # Create index
    print("\n📚 Creating index...")
    with open(output_dir / "INDEX.md", 'w') as f:
        f.write(f"""# Multi-Level Review Index

**Generated**: {datetime.now().isoformat()}
**Pages Reviewed**: {len(images)}
**Perspectives**: {len(SCHOLARS) + len(MUSES)}

## Scholars (Critical Analysis)
""")
        for name, config in SCHOLARS.items():
            f.write(f"- **{name.title()}**: {config['focus']}\n")
        
        f.write("\n## Muses (Creative Inspiration)\n")
        for name, config in MUSES.items():
            f.write(f"- **{name.title()}**: {config['focus']}\n")
        
        f.write("\n## Reviews by Page\n")
        for img in images:
            page_num = img.stem.replace('page-', '')
            f.write(f"\n### Page {page_num}\n")
            f.write(f"- [Synthesis](page_{page_num}_synthesis.md)\n")
            f.write("- Scholars: ")
            f.write(", ".join([f"[{n}](page_{page_num}_{n}.txt)" for n in SCHOLARS.keys()]))
            f.write("\n- Muses: ")
            f.write(", ".join([f"[{n}](page_{page_num}_{n}.txt)" for n in MUSES.keys()]))
            f.write("\n")
    
    print(f"\n✅ Complete! Results in {output_dir}/")
    print(f"\n📖 View index: cat {output_dir}/INDEX.md")

if __name__ == '__main__':
    main()
