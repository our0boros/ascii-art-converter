#!/usr/bin/env python3
"""
Test script to verify char_aspect_ratio is working correctly.
"""

from PIL import Image
from ascii_art_converter.generator import AsciiArtGenerator, AsciiArtConfig

def test_char_ratio():
    """
    Test different char_ratio values and show their effects.
    """
    # Create a simple test image
    image = Image.new('RGB', (100, 150), color='red')  # 3:2 aspect ratio
    print(f"Original image size: {image.size}")
    print(f"Original aspect ratio: {image.height / image.width:.3f}")
    print("\nTesting different char_ratio values:")
    print("-" * 50)
    
    # Test with different char ratios
    ratios = [0.3, 0.5, 0.7]
    for ratio in ratios:
        # Create config
        config = AsciiArtConfig(
            width=80,
            char_aspect_ratio=ratio,
            mode="density"
        )
        
        # Generate ASCII art
        generator = AsciiArtGenerator()
        result = generator.convert(image, config)
        
        # Calculate effective aspect ratio
        effective_aspect = result.height / result.width
        print(f"char_ratio: {ratio:.1f}")
        print(f"  Output size: {result.width}x{result.height} chars")
        print(f"  Effective aspect ratio: {effective_aspect:.3f}")
        print(f"  Total characters: {result.width * result.height}")
        print("-" * 50)


