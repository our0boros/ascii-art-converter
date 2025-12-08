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
    EDGE_ROUND: str = " ─│╱╲┼╭╮╯╰"            # Round edge characters: [none, horizontal, vertical, diagonal_up, diagonal_down, cross, rounded corners, ...]
    
    # User-defined character sets
    # 4字符（仅强度，无方向区分）：无边缘→弱边缘→中边缘→强边缘
    EDGE_SIMPLE: str = " .-=#"
    
    # 5字符（完整方向+强度）：无→水平→垂直→/斜→\斜
    EDGE_CHARSET_BASIC: str = " -|/\\"
    
    # 5字符增强版
    EDGE_CHARSET_CONTRAST: str = " ─│╱╲"
    
    # 扩展8字符（增加强度分级）
    EDGE_CHARSET_EXTENDED: str = " ─│╱╲═┃■"
    
    # 5字符灰度版（无边缘→水平→垂直→/斜→\斜，同时体现强度）
    EDGE_CHARSET_GRAY: str = " ░▒▓█"
    
    # 8字符灰度+方向版
    EDGE_CHARSET_GRAY_DIR: str = " ─│/\\▒▓█"
    
    # 索引：0(无)、1(细水平)、2(细垂直)、3(细/)、4(细\)、5(粗水平)、6(粗垂直)、7(交叉)
    EDGE_CHARSET_DIR_DETAIL: str = " ─│╱╲═┃╳"
    
    # 索引：0(无)、1(极弱)、2(弱)、3(中)、4(强)、5(极强)、6(极值)
    EDGE_CHARSET_STRENGTH: str = " ·░▒▓█▉"
    
    @classmethod
    def get_preset(cls, name: str) -> str:
        """Get character set by name."""
        presets = {
            # Standard density character sets
            'standard': cls.STANDARD,
            'detailed': cls.DETAILED,
            'blocks': cls.BLOCKS,
            'simple': cls.SIMPLE,
            'binary': cls.BINARY,
            'dots': cls.DOTS,
            'geometric': cls.GEOMETRIC,
            'arrows': cls.ARROWS,
            'math': cls.MATH,
            
            # Edge detection character sets
            'edge_basic': cls.EDGE_BASIC,
            'edge_detailed': cls.EDGE_DETAILED,
            'edge_round': cls.EDGE_ROUND,
            'edge_simple': cls.EDGE_SIMPLE,
            'edge_charset_basic': cls.EDGE_CHARSET_BASIC,
            'edge_charset_contrast': cls.EDGE_CHARSET_CONTRAST,
            'edge_charset_extended': cls.EDGE_CHARSET_EXTENDED,
            'edge_charset_gray': cls.EDGE_CHARSET_GRAY,
            'edge_charset_gray_dir': cls.EDGE_CHARSET_GRAY_DIR,
            'edge_charset_dir_detail': cls.EDGE_CHARSET_DIR_DETAIL,
            'edge_charset_strength': cls.EDGE_CHARSET_STRENGTH,
        }
        return presets.get(name.lower(), cls.STANDARD)
