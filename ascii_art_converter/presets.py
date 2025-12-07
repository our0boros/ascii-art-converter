#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Presets
==============================================
This module contains predefined configuration presets for ASCII art generation.
"""

from .config import AsciiArtConfig
from .character_sets import CharacterSet
from .constants import RenderMode, EdgeDetector, DitherMethod


class Presets:
    """Predefined configuration presets for ASCII art generation."""
    
    @staticmethod
    def photo_realistic() -> AsciiArtConfig:
        """High-detail configuration for photographs."""
        return AsciiArtConfig(
            max_width=150,
            charset=CharacterSet.DETAILED,
            contrast=1.2,
            sharpness=1.3,
            gamma=0.9
        )
    
    @staticmethod
    def simple_icons() -> AsciiArtConfig:
        """Simple configuration for icons and logos."""
        return AsciiArtConfig(
            max_width=60,
            min_width=30,
            charset=CharacterSet.SIMPLE,
            contrast=1.5
        )
    
    @staticmethod
    def retro_terminal() -> AsciiArtConfig:
        """Retro terminal look with blocks."""
        return AsciiArtConfig(
            max_width=80,
            charset=CharacterSet.BLOCKS,
            colorize=True
        )
    
    @staticmethod
    def edge_art() -> AsciiArtConfig:
        """Edge detection based art configuration."""
        return AsciiArtConfig(
            max_width=100,
            mode=RenderMode.EDGE,
            edge_threshold=0.15,
            edge_detector=EdgeDetector.CANNY
        )
    
    @staticmethod
    def braille_art() -> AsciiArtConfig:
        """Braille pattern configuration."""
        return AsciiArtConfig(
            max_width=80,
            mode=RenderMode.BRAILLE,
            braille_threshold=0.5,
            dither_method=DitherMethod.FLOYD_STEINBERG
        )
    
    @staticmethod
    def low_res() -> AsciiArtConfig:
        """Low resolution configuration for small displays."""
        return AsciiArtConfig(
            max_width=40,
            min_width=20,
            charset=CharacterSet.SIMPLE
        )
