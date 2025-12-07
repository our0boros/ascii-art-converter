from PIL import Image, ImageDraw
from typing import Optional, Union, Dict, Literal, List
import numpy as np

from .generator import AsciiArtGenerator
from .formatters import AnsiColorFormatter, HtmlFormatter
from .config import AsciiArtConfig, AsciiArtResult
from .constants import RenderMode, DitherMethod
from .character_sets import CharacterSet
from .complexity import ComplexityAnalyzer


def image_to_ascii(image: Image.Image,
                   width: Optional[int] = None,
                   mode: Union[str, RenderMode] = 'density',
                   charset: str = 'standard',
                   colorize: bool = False,
                   invert: bool = False,
                   **kwargs) -> AsciiArtResult:
    """Convenience function to generate ASCII art from image with minimal parameters."""
    # Parse mode
    if isinstance(mode, str):
        mode = mode.upper()
        if mode not in RenderMode.__members__:
            raise ValueError(f"Invalid mode: {mode}. Must be one of: {', '.join(RenderMode.__members__.keys())}")
        mode = RenderMode[mode]
    
    # Parse charset
    if isinstance(charset, str) and hasattr(CharacterSet, charset.upper()):
        charset = getattr(CharacterSet, charset.upper())
    
    # Build config
    config = AsciiArtConfig(
        width=width,
        mode=mode,
        charset=charset,
        colorize=colorize,
        invert=invert,
        **kwargs
    )
    
    # Generate ASCII art
    generator = AsciiArtGenerator()
    return generator.convert(image, config)


def print_ascii(image: Image.Image,
                width: Optional[int] = None,
                mode: str = 'density',
                colorize: bool = True,
                color_mode: str = '24bit',
                **kwargs) -> None:
    """Generate ASCII art and print it to terminal."""
    result = image_to_ascii(image, width=width, mode=mode, colorize=colorize, **kwargs)
    
    if colorize:
        print(AnsiColorFormatter.format_result(result, color_mode=color_mode))
    else:
        print(result.text)


def save_html(image: Image.Image,
              output_path: str,
              width: Optional[int] = None,
              mode: str = 'density',
              colorize: bool = True,
              **kwargs) -> None:
    """Generate ASCII art and save it as HTML file."""
    result = image_to_ascii(image, width=width, mode=mode, colorize=colorize, **kwargs)
    html_content = HtmlFormatter.format_result(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)


def demo():
    """Demonstrate the ASCII art generator capabilities."""
    
    print("=" * 60)
    print("ASCII Art Generator Demo")
    print("=" * 60)
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Draw some shapes
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([10, 10, 90, 90], fill='red', outline='black')
    draw.rectangle([30, 30, 70, 70], fill='blue')
    
    print("\n1. Density Mode (Standard charset):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='density', charset='standard')
    print(result.text)
    print(f"Complexity: {result.complexity_score:.2f}")
    
    print("\n2. Density Mode (Blocks charset):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='density', charset='blocks')
    print(result.text)
    
    print("\n3. Braille Mode:")
    print("-" * 40)
    result = image_to_ascii(test_image, width=20, mode='braille')
    print(result.text)
    
    print("\n4. Edge Detection Mode:")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='edge', 
                           edge_threshold=0.15, edge_charset='edge_detailed')
    print(result.text)
    
    print("\n5. With Colors (showing hex values):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='density', colorize=True)
    # Show a small part with color codes
    lines = result.text.split('\n')[:5]
    for line in lines[:5]:
        if len(line) > 20:
            print(line[:20] + "...")
        else:
            print(line)


def resize_to_fit(image: Image.Image, 
                  max_width: int = 800, 
                  max_height: int = 800) -> Image.Image:
    """Resize image to fit within max dimensions while maintaining aspect ratio."""
    width, height = image.size
    
    if width <= max_width and height <= max_height:
        return image
    
    ratio = min(max_width / width, max_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)


def create_comparison(image: Image.Image, 
                      width: int = 60) -> str:
    """Create a comparison of different rendering modes."""
    output = []
    
    output.append("=" * (width * 3 + 4))
    output.append("DENSITY MODE".center(width) + " | " + 
                  "BRAILLE MODE".center(width) + " | " + 
                  "EDGE MODE".center(width))
    output.append("=" * (width * 3 + 4))
    
    # Generate all three modes
    result_density = image_to_ascii(image, width=width, mode='density')
    result_braille = image_to_ascii(image, width=width, mode='braille')
    result_edge = image_to_ascii(image, width=width, mode='edge')
    
    # Combine them line by line
    density_lines = result_density.lines
    braille_lines = result_braille.lines
    edge_lines = result_edge.lines
    
    max_lines = max(len(density_lines), len(braille_lines), len(edge_lines))
    
    for i in range(max_lines):
        line = ""
        if i < len(density_lines):
            line += density_lines[i].ljust(width)
        else:
            line += " ".ljust(width)
            
        line += " | "
        
        if i < len(braille_lines):
            line += braille_lines[i].ljust(width)
        else:
            line += " ".ljust(width)
            
        line += " | "
        
        if i < len(edge_lines):
            line += edge_lines[i].ljust(width)
        else:
            line += " ".ljust(width)
            
        output.append(line)
    
    output.append("=" * (width * 3 + 4))
    return "\n".join(output)


def get_optimal_settings(image: Image.Image) -> AsciiArtConfig:
    """Analyze image and return optimal settings."""
    complexity = ComplexityAnalyzer.analyze(image)
    width, height = image.size
    aspect = height / width
    
    # Start with default config
    config = AsciiArtConfig()
    
    # Adjust based on complexity
    if complexity > 0.6:
        # High complexity - use detailed charset
        config.charset = CharacterSet.DETAILED
        config.max_width = 120
        config.sharpness = 1.2
    elif complexity < 0.3:
        # Low complexity - use simple charset
        config.charset = CharacterSet.SIMPLE
        config.max_width = 80
        config.contrast = 1.3
    else:
        config.charset = CharacterSet.STANDARD
        config.max_width = 100
    
    # Adjust for aspect ratio
    if aspect > 1.5:
        # Tall image
        config.max_width = 80
    elif aspect < 0.67:
        # Wide image
        config.max_width = 140
    
    return config
