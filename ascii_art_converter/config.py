#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Configuration
====================================================
This module contains the configuration classes for ASCII art generation.
"""

from typing import Optional, Tuple, List, Literal
from dataclasses import dataclass, field

from .constants import RenderMode, EdgeDetector, DitherMethod
from .character_sets import CharacterSet


@dataclass
class AsciiArtConfig:
    """Configuration for ASCII art generation."""
    
    # Size parameters
    width: Optional[int] = None              # Output width (auto if None)
    height: Optional[int] = None             # Output height (auto if None)
    max_width: int = 120                     # Maximum auto width
    min_width: int = 40                      # Minimum auto width
    
    # Character aspect ratio (width/height of monospace char)
    char_aspect_ratio: float = 0.45           # Typical terminal char ratio
    
    # Rendering mode
    mode: RenderMode = RenderMode.DENSITY
    
    # Character set
    charset: str = CharacterSet.STANDARD
    invert: bool = False                     # Invert brightness mapping
    
    # Edge detection settings
    edge_detector: EdgeDetector = EdgeDetector.SOBEL
    edge_threshold: float = 0.1              # Edge detection threshold (0-1)
    edge_charset: str = CharacterSet.EDGE_BASIC
    edge_sigma: float = 1.0                  # Gaussian blur sigma for edge
    
    # Braille settings
    braille_threshold: float = 0.5           # Threshold for braille dots
    dither_method: DitherMethod = DitherMethod.NONE
    
    # Color settings
    colorize: bool = False                   # Enable color output
    color_depth: int = 8                     # Color quantization depth
    color_sample_mode: Literal['center', 'average', 'dominant'] = 'average'
    
    # Enhancement settings
    contrast: float = 1.0                    # Contrast adjustment (0.5-2.0)
    brightness: float = 1.0                  # Brightness adjustment (0.5-2.0)
    gamma: float = 1.0                       # Gamma correction
    sharpness: float = 1.0                   # Sharpness enhancement
    
    # Complexity-based auto-sizing
    complexity_factor: float = 1.0           # Multiplier for auto-size
    
    # Background handling
    background_char: str = ' '               # Character for background
    transparent_threshold: int = 128         # Alpha threshold for transparency


@dataclass
class AsciiArtResult:
    """Result of ASCII art generation."""
    text: str                                          # The ASCII art text
    lines: List[str]                                   # Lines of ASCII art
    colors: Optional[List[List[str]]] = None           # Hex color array if colorized
    width: int = 0                                     # Output width
    height: int = 0                                    # Output height
    original_size: Tuple[int, int] = (0, 0)           # Original image size
    complexity_score: float = 0.0                      # Calculated complexity
