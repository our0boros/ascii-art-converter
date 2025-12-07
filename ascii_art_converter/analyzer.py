#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - ASCII Analysis
====================================================
This module contains the AsciiAnalyzer class for analyzing ASCII/Braille art.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from ascii_art_converter.config import AsciiArtResult
from ascii_art_converter.character_sets import CharacterSet


class AsciiAnalyzer:
    """Analyze ASCII/Braille art properties."""
    
    @staticmethod
    def analyze(result: AsciiArtResult) -> Dict[str, float]:
        """
        Analyze various properties of ASCII/Braille art.
        
        Args:
            result: AsciiArtResult containing the ASCII art
            
        Returns:
            Dictionary containing analysis results
        """
        analysis = {}
        
        # Basic statistics
        analysis['line_count'] = AsciiAnalyzer._count_lines(result.text)
        analysis['char_count'] = AsciiAnalyzer._count_characters(result.text)
        analysis['avg_line_length'] = AsciiAnalyzer._average_line_length(result.text)
        
        # Character distribution
        char_dist = AsciiAnalyzer._character_distribution(result.text)
        analysis['char_diversity'] = len(char_dist)
        analysis['top_char'] = max(char_dist.items(), key=lambda x: x[1])[0] if char_dist else ''
        
        # Complexity measures
        # Note: result.character_set might not be available in all cases
        # We'll use a default character set if needed
        charset_name = getattr(result, 'character_set', CharacterSet.STANDARD)
        # Convert string charset name to CharacterSet object
        try:
            charset = CharacterSet.from_name(charset_name)
        except (AttributeError, ValueError):
            # Fallback to default if conversion fails
            charset = CharacterSet.STANDARD
        analysis['density'] = AsciiAnalyzer._calculate_density(result.text, charset)
        analysis['contrast'] = AsciiAnalyzer._calculate_contrast(result.text, charset)
        
        # Braille-specific analysis
        if hasattr(result, 'render_mode') and result.render_mode.name.lower() == 'braille':
            braille_analysis = AsciiAnalyzer._analyze_braille(result.text)
            analysis.update(braille_analysis)
        
        return analysis
    
    @staticmethod
    def _count_lines(art: str) -> int:
        """Count number of lines in ASCII art."""
        if not art:
            return 0
        return len(art.split('\n'))
    
    @staticmethod
    def _count_characters(art: str) -> int:
        """Count total number of characters in ASCII art."""
        if not art:
            return 0
        # Exclude newlines
        return len(art.replace('\n', ''))
    
    @staticmethod
    def _average_line_length(art: str) -> float:
        """Calculate average line length in ASCII art."""
        if not art:
            return 0.0
        
        lines = art.split('\n')
        line_lengths = [len(line) for line in lines]
        return sum(line_lengths) / len(line_lengths) if line_lengths else 0.0
    
    @staticmethod
    def _character_distribution(art: str) -> Dict[str, int]:
        """Calculate character distribution in ASCII art."""
        distribution = {}
        
        if not art:
            return distribution
        
        for char in art:
            if char != '\n':  # Skip newlines
                distribution[char] = distribution.get(char, 0) + 1
        
        return distribution
    
    @staticmethod
    def _calculate_density(art: str, char_set: str) -> float:
        """
        Calculate density of ASCII art (how much of the space is filled).
        
        Args:
            art: ASCII art string
            char_set: CharacterSet used for generation
            
        Returns:
            Density value between 0 and 1
        """
        if not art or not char_set:
            return 0.0
        
        # Create a simple brightness map based on character positions
        # Characters are mapped from 0.0 (darkest) to 1.0 (lightest)
        brightness_map = {char: i / (len(char_set) - 1) if len(char_set) > 1 else 0.5 
                         for i, char in enumerate(char_set)}
        
        # Calculate average brightness
        total_brightness = 0.0
        total_chars = 0
        
        for char in art:
            if char != '\n':
                brightness = brightness_map.get(char, 0.5)  # Default to mid brightness
                total_brightness += brightness
                total_chars += 1
        
        return total_brightness / total_chars if total_chars > 0 else 0.0
    
    @staticmethod
    def _calculate_contrast(art: str, char_set: str) -> float:
        """
        Calculate contrast of ASCII art.
        
        Args:
            art: ASCII art string
            char_set: CharacterSet used for generation
            
        Returns:
            Contrast value between 0 and 1
        """
        if not art or not char_set:
            return 0.0
        
        # Create a simple brightness map based on character positions
        # Characters are mapped from 0.0 (darkest) to 1.0 (lightest)
        brightness_map = {char: i / (len(char_set) - 1) if len(char_set) > 1 else 0.5 
                         for i, char in enumerate(char_set)}
        
        # Collect brightness values
        brightness_values = []
        for char in art:
            if char != '\n':
                brightness = brightness_map.get(char, 0.5)  # Default to mid brightness
                brightness_values.append(brightness)
        
        if not brightness_values:
            return 0.0
        
        # Calculate contrast as standard deviation
        contrast = np.std(brightness_values)
        
        # Normalize to 0-1 range (max possible std for uniform distribution is 0.5)
        return min(contrast * 2, 1.0)
    
    @staticmethod
    def _analyze_braille(art: str) -> Dict[str, float]:
        """
        Analyze Braille-specific properties.
        
        Args:
            art: Braille art string
            
        Returns:
            Dictionary containing Braille-specific analysis
        """
        analysis = {}
        
        if not art:
            return analysis
        
        # Count dot usage
        dot_counts = AsciiAnalyzer._count_braille_dots(art)
        analysis['avg_dots_per_char'] = sum(dot_counts.values()) / len(dot_counts) if dot_counts else 0.0
        analysis['max_dots_per_char'] = max(dot_counts.values()) if dot_counts else 0
        analysis['min_dots_per_char'] = min(dot_counts.values()) if dot_counts else 0
        
        # Calculate dot density
        total_dots = sum(dot_counts.values())
        total_chars = AsciiAnalyzer._count_characters(art)
        analysis['dot_density'] = total_dots / (total_chars * 8) if total_chars > 0 else 0.0
        
        return analysis
    
    @staticmethod
    def _count_braille_dots(art: str) -> Dict[str, int]:
        """
        Count the number of dots in each Braille character.
        
        Args:
            art: Braille art string
            
        Returns:
            Dictionary mapping Braille characters to dot counts
        """
        # Braille base code point
        BRAILLE_BASE = 0x2800
        
        dot_counts = {}
        
        for char in art:
            if char != '\n':
                # Check if character is a Braille character
                if 0x2800 <= ord(char) <= 0x28FF:
                    # Calculate dot pattern
                    dot_pattern = ord(char) - BRAILLE_BASE
                    # Count number of set bits (dots)
                    dot_count = bin(dot_pattern).count('1')
                    dot_counts[char] = dot_count
        
        return dot_counts
