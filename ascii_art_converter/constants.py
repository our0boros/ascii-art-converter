#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Constants and Enums
========================================================
This module contains all the constants and enumerations used by the converter.
"""

from enum import Enum, auto


# =============================================================================
# ENUMS
# =============================================================================

class RenderMode(Enum):
    """Rendering mode for ASCII art generation."""
    DENSITY = auto()      # Character density based
    BRAILLE = auto()      # Unicode braille patterns
    EDGE = auto()         # Edge detection based


class EdgeDetector(Enum):
    """Edge detection algorithm."""
    SOBEL = auto()
    PREWITT = auto()
    LAPLACIAN = auto()
    CANNY = auto()
    SCHARR = auto()


class DitherMethod(Enum):
    """Dithering method for braille patterns."""
    NONE = auto()
    FLOYD_STEINBERG = auto()
    ORDERED = auto()
    ATKINSON = auto()


# =============================================================================
# CONSTANTS
# =============================================================================

# Braille pattern base and dot positions
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = [
    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40),
    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80)
]
