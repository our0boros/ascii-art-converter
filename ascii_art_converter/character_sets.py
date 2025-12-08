#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Character Sets
====================================================
This module contains predefined character sets for ASCII and Braille art generation.
"""

from dataclasses import dataclass


@dataclass
class CharacterSet:
    """Predefined character sets for density mapping."""
    
    # Standard ASCII ramps (dark to light)
    STANDARD: str = " .:-=+*#%@"
    DETAILED: str = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
    BLOCKS: str = " ░▒▓█"
    SIMPLE: str = " .oO@"
    BINARY: str = " █"
    DOTS: str = " ⠁⠂⠃⠄⠅⠆⠇⡀⡁⡂⡃⡄⡅⡆⡇"
    GEOMETRIC: str = " ·∘○◎●◉"
    ARROWS: str = " ←↑→↓↔↕↖↗↘↙"
    MATH: str = " ∙∴∷⊕⊗⊙⊚⊛"
    
    # Edge detection characters - ordered according to main.py's DIRECTION_CHARS
    # TODO: generator.py Line 302
    # Order: none, horizontal, vertical, diagonal_up(/), diagonal_down(\), cross, corners, etc.
    EDGE_BASIC: str = " -|/\\+LT7VXY"         # Basic edge characters: [none, horizontal, vertical, diagonal_up, diagonal_down, ...]
    EDGE_DETAILED: str = " ─│╱╲┼┌┐└┘├┤┬┴╭╮╯╰"  # Detailed edge characters: [none, horizontal, vertical, diagonal_up, diagonal_down, cross, corners, ...]
    EDGE_ROUND: str = " ·─│╱╲┼╭╮╯╰"           # Round edge characters: [none, horizontal, vertical, diagonal_up, diagonal_down, cross, rounded corners, ...]
    
    @classmethod
    def get_preset(cls, name: str) -> str:
        """Get character set by name."""
        presets = {
            'standard': cls.STANDARD,
            'detailed': cls.DETAILED,
            'blocks': cls.BLOCKS,
            'simple': cls.SIMPLE,
            'binary': cls.BINARY,
            'dots': cls.DOTS,
            'geometric': cls.GEOMETRIC,
            'edge_basic': cls.EDGE_BASIC,
            'edge_detailed': cls.EDGE_DETAILED,
            'edge_round': cls.EDGE_ROUND,
        }
        return presets.get(name.lower(), cls.STANDARD)
