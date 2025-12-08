#!/usr/bin/env python3
"""Command line interface for ASCII art converter."""

import argparse
from PIL import Image

from .generator import AsciiArtGenerator, AsciiArtConfig
from .formatters import HtmlFormatter, AnsiColorFormatter
from .analyzer import AsciiAnalyzer
from .utils import demo
from .constants import RenderMode, EdgeDetector, DitherMethod
from .character_sets import CharacterSet

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description='Convert images to ASCII/Braille art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                          # Basic conversion
  %(prog)s image.png -w 80                    # Set width to 80 chars
  %(prog)s image.png -m braille              # Use braille mode
  %(prog)s image.png -m edge -t 0.2          # Edge detection
  %(prog)s image.png -c -o output.html       # Colored HTML output
  %(prog)s image.png --color-mode 256        # 256-color terminal output
        """
    )
    
    # Input/Output
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file (txt, html, or ansi)')
    
    # Size options
    parser.add_argument('-w', '--width', type=int, help='Output width in characters')
    parser.add_argument('-H', '--height', type=int, help='Output height in characters')
    parser.add_argument('--max-width', type=int, default=120, help='Maximum auto width')
    parser.add_argument('--min-width', type=int, default=40, help='Minimum auto width')
    parser.add_argument('--char-ratio', type=float, default=0.45, 
                        help='Character aspect ratio (width/height)')
    
    # Mode options
    parser.add_argument('-m', '--mode', choices=['density', 'braille', 'edge'],
                        default='density', help='Rendering mode')
    
    # Character set options
    parser.add_argument('--charset', default='standard',
                        help='Character set: standard, detailed, blocks, simple, binary, dots, geometric')
    parser.add_argument('--custom-charset', help='Custom character string (dark to light)')
    parser.add_argument('-i', '--invert', action='store_true', help='Invert brightness')
    
    # Edge detection options
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Edge detection threshold (0-1)')
    parser.add_argument('--edge-detector', choices=['sobel', 'prewitt', 'laplacian', 'canny', 'scharr'],
                        default='sobel', help='Edge detection algorithm')
    parser.add_argument('--edge-sigma', type=float, default=1.0,
                        help='Gaussian blur sigma for edge detection')
    parser.add_argument('--edge-charset', default='edge_basic', 
                        choices=['edge_basic', 'edge_detailed', 'edge_round', 'edge_simple', 
                                 'edge_charset_basic', 'edge_charset_contrast', 'edge_charset_extended',
                                 'edge_charset_gray', 'edge_charset_gray_dir', 'edge_charset_dir_detail',
                                 'edge_charset_strength'],
                        help='Character set for edge mode')
    
    # Braille options
    parser.add_argument('--braille-threshold', type=float, default=0.5,
                        help='Threshold for braille dots (0-1)')
    parser.add_argument('--dither', choices=['none', 'floyd_steinberg', 'ordered', 'atkinson'],
                        default='none', help='Dithering method for braille')
    
    # Color options
    parser.add_argument('-c', '--colorize', action='store_true', help='Enable color output')
    parser.add_argument('--color-mode', choices=['24bit', '256', '16'],
                        default='24bit', help='Terminal color mode')
    parser.add_argument('--color-sample', choices=['center', 'average', 'dominant'],
                        default='average', help='Color sampling method')
    
    # Enhancement options
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='Contrast adjustment (0.5-2.0)')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness adjustment (0.5-2.0)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction')
    parser.add_argument('--sharpness', type=float, default=1.0,
                        help='Sharpness enhancement')
    
    # Other options
    parser.add_argument('--complexity-factor', type=float, default=1.0,
                        help='Multiplier for auto-size complexity calculation')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('-a', '--analyze', action='store_true', 
                        help='Show analysis of the generated ASCII art')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    return parser


def main():
    """Main entry point for command line usage."""
    parser = create_argument_parser()
    args = parser.parse_args()

    # Run demo if requested
    if args.demo:
        demo()
        return
    
    # Check for input
    if not args.input:
        parser.print_help()
        return
    
    # Load image
    try:
        image = Image.open(args.input)
        if args.verbose:
            print(f"Loaded image: {args.input}")
            print(f"Size: {image.size}, Mode: {image.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Parse mode
    mode_map = {
        'density': RenderMode.DENSITY,
        'braille': RenderMode.BRAILLE,
        'edge': RenderMode.EDGE
    }
    mode = mode_map[args.mode]
    
    # Parse edge detector
    edge_detector_map = {
        'sobel': EdgeDetector.SOBEL,
        'prewitt': EdgeDetector.PREWITT,
        'laplacian': EdgeDetector.LAPLACIAN,
        'canny': EdgeDetector.CANNY,
        'scharr': EdgeDetector.SCHARR
    }
    edge_detector = edge_detector_map[args.edge_detector]
    
    # Parse dither method
    dither_map = {
        'none': DitherMethod.NONE,
        'floyd_steinberg': DitherMethod.FLOYD_STEINBERG,
        'ordered': DitherMethod.ORDERED,
        'atkinson': DitherMethod.ATKINSON
    }
    dither_method = dither_map[args.dither]
    
    # Get charset
    charset = args.custom_charset if args.custom_charset else CharacterSet.get_preset(args.charset)
    edge_charset = CharacterSet.get_preset(args.edge_charset)

    # Build config
    config = AsciiArtConfig(
        width=args.width,
        height=args.height,
        max_width=args.max_width,
        min_width=args.min_width,
        char_aspect_ratio=args.char_ratio,
        mode=mode,
        charset=charset,
        invert=args.invert,
        edge_detector=edge_detector,
        edge_threshold=args.threshold,
        edge_charset=edge_charset,
        edge_sigma=args.edge_sigma,
        braille_threshold=args.braille_threshold,
        dither_method=dither_method,
        colorize=args.colorize,
        color_sample_mode=args.color_sample,
        contrast=args.contrast,
        brightness=args.brightness,
        gamma=args.gamma,
        sharpness=args.sharpness,
        complexity_factor=args.complexity_factor
    )
    
    # Generate ASCII art
    generator = AsciiArtGenerator()
    result = generator.convert(image, config)
    
    if args.verbose:
        print(f"Output size: {result.width}x{result.height}")
        print(f"Complexity score: {result.complexity_score:.3f}")
        print()
    
    # Handle output
    if args.output:
        ext = args.output.lower().split('.')[-1]
        
        if ext == 'html':
            html = HtmlFormatter.format_result(result)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Saved to {args.output}")
            
        elif ext == 'ansi':
            if result.colors:
                ansi = AnsiColorFormatter.format_result(result, color_mode=args.color_mode)
            else:
                ansi = result.text
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(ansi)
            print(f"Saved to {args.output}")
            
        else:  # txt or other
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Saved to {args.output}")
    else:
        # Print to terminal
        if args.colorize and result.colors:
            output = AnsiColorFormatter.format_result(result, color_mode=args.color_mode)
            print(output)
        else:
            print(result.text)
    
    # Show analysis if requested
    if args.analyze:
        print("\n" + "=" * 50)
        print("Analysis:")
        print("=" * 50)
        
        # Get comprehensive analysis
        analysis = AsciiAnalyzer.analyze(result)
        
        # Basic statistics
        print(f"\nBasic Statistics:")
        print(f"  Lines: {analysis['line_count']}")
        print(f"  Characters: {analysis['char_count']}")
        print(f"  Average line length: {analysis['avg_line_length']:.1f}")
        
        # Character distribution
        print(f"\nCharacter Distribution:")
        print(f"  Character diversity: {analysis['char_diversity']}")
        top_char = analysis['top_char']
        char_display = repr(top_char) if top_char in ' \t\n' else top_char
        print(f"  Most used character: {char_display}")
        
        # Complexity measures
        print(f"\nComplexity Measures:")
        print(f"  Density: {analysis['density']:.3f}")
        print(f"  Contrast: {analysis['contrast']:.3f}")
        
        # Braille-specific analysis
        if args.mode.lower() == 'braille':
            print(f"\nBraille Analysis:")
            # Only print Braille-specific metrics if they exist in the analysis
            if 'avg_dots_per_char' in analysis:
                print(f"  Average dots per character: {analysis['avg_dots_per_char']:.2f}")
                print(f"  Dot density: {analysis['dot_density']:.3f}")
                print(f"  Max dots per character: {analysis['max_dots_per_char']}")
                print(f"  Min dots per character: {analysis['min_dots_per_char']}")


if __name__ == '__main__':
    main()
