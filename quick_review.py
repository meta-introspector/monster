#!/usr/bin/env python3
"""Quick re-review after improvements"""
import subprocess, json, base64
from pathlib import Path

def review_page(img, persona, prompt):
    with open(img, 'rb') as f:
        img_data = base64.b64encode(f.read()).decode('utf-8')
    
    payload = {"model": "llava", "prompt": prompt, "images": [img_data], "stream": False}
    result = subprocess.run(['curl', '-s', 'http://localhost:11434/api/generate', '-d', '@-'],
                          input=json.dumps(payload), capture_output=True, text=True)
    return json.loads(result.stdout).get('response', '') if result.returncode == 0 else ''

# Review page 1 with mathematician (notation check)
img = Path('vision_reviews_v2/page-01.png')
prompt = """Review this page as a mathematician. 

SPECIFIC QUESTION: Is there now a notation glossary? 
Does it define symbols clearly? 
Rate improvement: Better/Same/Worse than before."""

print("🎓 Mathematician reviewing notation glossary...")
review = review_page(img, 'mathematician', prompt)

with open('review_v2_mathematician.txt', 'w') as f:
    f.write(f"# V2 Review - Mathematician\n\n{review}")

print(f"✓ Saved to review_v2_mathematician.txt")
print(f"\n{review[:500]}...")
