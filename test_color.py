#!/usr/bin/env python3
"""
Test script to verify color mode functionality.
"""
import sys
from PIL import Image
from ascii_art_converter.generator import AsciiArtGenerator
from ascii_art_converter.config import AsciiArtConfig, RenderMode, DitherMethod
from ascii_art_converter.formatters import AnsiColorFormatter

# Load test image
test_image_path = "./data/test.png"
print(f"Loading test image from {test_image_path}...")

try:
    image = Image.open(test_image_path)
except FileNotFoundError:
    print(f"Error: Test image not found at {test_image_path}")
    sys.exit(1)

# Create generator
generator = AsciiArtGenerator()

# Create config with color enabled
config = AsciiArtConfig(
    width=50,
    mode=RenderMode.DENSITY,
    colorize=True,
    color_sample_mode="pixel"
)

print("Generating ASCII art with color data...")
result = generator.convert(image, config)

print(f"Generated ASCII art with {result.height} lines and {result.width} characters per line")
print(f"Color data available: {result.colors is not None}")

if result.colors:
    print(f"Color data dimensions: {len(result.colors)}x{len(result.colors[0])}")
    
    # Test color formatter
    print("\nTesting ANSI color formatter...")
    formatted = AnsiColorFormatter.format_result(result, color_mode="24bit")
    
    print("First few lines with ANSI color codes:")
    # Print just the first few lines to verify
    lines = formatted.split('\n')[:5]
    for line in lines:
        print(repr(line))  # Show the raw string with ANSI codes
        print(line)        # Show the rendered version
else:
    print("Error: No color data generated!")
    sys.exit(1)

print("\nColor mode test completed successfully!")