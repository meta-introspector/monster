#!/usr/bin/env python3
"""Review paper pages with llava vision model"""
import subprocess
import json
import base64
from pathlib import Path

def review_page(image_path, page_num):
    """Review a single page with llava"""
    
    # Read image as base64
    with open(image_path, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    prompt = f"""Analyze this research paper page {page_num}. Focus on:

1. **Mathematical Correctness**: Check equations, proofs, and calculations
2. **Clarity**: Are concepts explained clearly?
3. **Missing Diagrams**: What visualizations would help?
4. **Notation Issues**: Any inconsistencies or unclear notation?
5. **Specific Improvements**: Concrete suggestions

Be critical and specific. Point out actual issues you see."""
    
    # Call ollama API with base64 image via stdin
    payload = {
        "model": "llava",
        "prompt": prompt,
        "images": [img_data],
        "stream": False
    }
    
    result = subprocess.run(
        ['curl', '-s', 'http://localhost:11434/api/generate',
         '-d', '@-'],
        input=json.dumps(payload),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        response = json.loads(result.stdout)
        return response.get('response', 'No response')
    else:
        return f"Error: {result.stderr}"

def main():
    vision_dir = Path('vision_reviews')
    images = sorted(vision_dir.glob('page-*.png'))
    
    print(f"📊 Found {len(images)} pages to review\n")
    
    for img in images:
        page_num = img.stem.replace('page-', '')
        print(f"=== Reviewing page {page_num} ===")
        
        review = review_page(img, page_num)
        
        output_file = vision_dir / f"review_{page_num}.txt"
        with open(output_file, 'w') as f:
            f.write(f"# Page {page_num} Review\n\n")
            f.write(review)
        
        print(f"✓ Saved to {output_file}\n")
    
    print("✅ All reviews complete!")

if __name__ == '__main__':
    main()
