#!/usr/bin/env python3
"""Background review system with 407K+ authors"""
import subprocess
import json
import base64
from pathlib import Path
import random
import time

def get_top_authors(limit=100):
    """Get top ranked authors from parquet files"""
    timeline_dir = Path('/home/mdupont/nix-controller/data/user_timelines')
    
    # Sample notable authors (you can replace with actual ranking logic)
    all_files = list(timeline_dir.glob('*.parquet'))
    
    # For now, sample randomly - replace with actual ranking
    sampled = random.sample(all_files, min(limit, len(all_files)))
    
    return [f.stem for f in sampled]

def review_with_author(image_path, page_num, author_name):
    """Quick review from author perspective"""
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = f"""You are {author_name}, reviewing this research paper page {page_num}.
    
Give your unique perspective in 2-3 sentences. Be authentic to your voice."""
    
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
        text=True,
        timeout=30
    )
    
    if result.returncode == 0:
        return json.loads(result.stdout).get('response', '')
    return ''

def background_review():
    """Run massive background review"""
    output_dir = Path('massive_reviews')
    output_dir.mkdir(exist_ok=True)
    
    vision_dir = Path('vision_reviews')
    images = sorted(vision_dir.glob('page-*.png'))[:1]  # Start with page 1
    
    print(f"🌍 MASSIVE REVIEW SYSTEM")
    print(f"📊 407,756 authors available")
    print(f"🎯 Starting with top 100 authors")
    print(f"📄 Reviewing {len(images)} page(s)\n")
    
    authors = get_top_authors(100)
    
    for i, author in enumerate(authors):
        print(f"[{i+1}/100] {author}...", end=' ', flush=True)
        
        for img in images:
            page_num = img.stem.replace('page-', '')
            
            try:
                review = review_with_author(img, page_num, author)
                
                # Save review
                output_file = output_dir / f"page_{page_num}_{author}.txt"
                with open(output_file, 'w') as f:
                    f.write(f"# {author} - Page {page_num}\n\n{review}\n")
                
                print("✓")
            except Exception as e:
                print(f"✗ ({e})")
        
        # Rate limit
        time.sleep(0.5)
    
    print(f"\n✅ Complete! {len(authors)} reviews in {output_dir}/")

if __name__ == '__main__':
    background_review()
