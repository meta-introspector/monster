#!/usr/bin/env python3
"""Convert text and audio to images for vision model probing"""

from PIL import Image, ImageDraw, ImageFont
import json
import os

MONSTER_PRIMES = [
    (2, "🌙"), (3, "🌊"), (5, "⭐"), (7, "🎭"), (11, "🎪"),
    (13, "🎨"), (17, "🎯"), (19, "🎪"), (23, "🎵"), (29, "🎸"),
    (31, "🎹"), (41, "🎺"), (47, "🎻"), (59, "🎼"), (71, "🎤")
]

def text_to_image(text, output_path, size=(512, 512)):
    """Convert text to image"""
    img = Image.new('RGB', size, color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 32)
    except:
        font = ImageFont.load_default()
    
    # Center text
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    x = (size[0] - text_width) // 2
    y = (size[1] - text_height) // 2
    
    draw.text((x, y), text, fill='black', font=font)
    img.save(output_path)
    print(f"✓ Generated: {output_path}")

def generate_prime_text_images():
    """Generate text images for all Monster primes"""
    os.makedirs('generated_images/text', exist_ok=True)
    
    for prime, emoji in MONSTER_PRIMES:
        # Simple text
        text = f"Prime {prime}"
        text_to_image(text, f'generated_images/text/prime_{prime}_simple.png')
        
        # With emoji
        text = f"Prime {prime}: {emoji}"
        text_to_image(text, f'generated_images/text/prime_{prime}_emoji.png')
        
        # With context
        text = f"Prime {prime}: {emoji}\nMonster group factor"
        text_to_image(text, f'generated_images/text/prime_{prime}_context.png')

def generate_results_image():
    """Convert RESULTS.md key findings to image"""
    text = """Monster Prime Resonance

Prime 2: 80.0%
Prime 3: 49.3%
Prime 5: 43.1%
Prime 7: 34.6%
Prime 11: 32.2%

Conway activates:
Prime 17: 78.6%
Prime 47: 28.6%"""
    
    text_to_image(text, 'generated_images/text/results_summary.png', size=(512, 768))

def generate_equation_image():
    """Generate Monster group order equation as image"""
    text = """Monster Group Order:

2^46 × 3^20 × 5^9 × 7^6 × 11^2
× 13^3 × 17 × 19 × 23 × 29
× 31 × 41 × 47 × 59 × 71

≈ 8.080 × 10^53"""
    
    text_to_image(text, 'generated_images/text/monster_order.png', size=(512, 768))

if __name__ == '__main__':
    print("🎨 Converting text to images for vision models")
    print("=" * 50)
    
    generate_prime_text_images()
    generate_results_image()
    generate_equation_image()
    
    print("\n✓ All text images generated!")
    print("  Location: generated_images/text/")
