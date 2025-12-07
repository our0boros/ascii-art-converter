#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Formatters
================================================
This module contains formatters for ASCII/Braille art.
"""

from PIL import Image
import numpy as np
from typing import Tuple, Optional
from ascii_art_converter.config import AsciiArtResult


class AnsiColorFormatter:
    """Format ASCII/Braille art with ANSI color codes."""
    
    @staticmethod
    def format_result(result: AsciiArtResult, color_mode: str = '24bit') -> str:
        """
        Format ASCII art with ANSI color codes.
        
        Args:
            result: AsciiArtResult containing the ASCII art with colors
            color_mode: Color mode (24bit, 256, 16)
            
        Returns:
            ANSI-colored ASCII art string
        """
        if not result.colors:
            return result.text
        
        # Format each character with color
        colored_lines = []
        for i, line in enumerate(result.lines):
            colored_chars = []
            for j, char in enumerate(line):
                if i < len(result.colors) and j < len(result.colors[i]):
                    color = result.colors[i][j]
                    # Parse hex color to RGB
                    color = color.lstrip('#')
                    r = int(color[0:2], 16)
                    g = int(color[2:4], 16)
                    b = int(color[4:6], 16)
                    
                    # Foreground color
                    colored_chars.append(f"\033[38;2;{r};{g};{b}m{char}\033[0m")
                else:
                    colored_chars.append(char)
            
            colored_lines.append(''.join(colored_chars))
        
        return '\n'.join(colored_lines)
    
    @staticmethod
    def get_ansi_color_code(r: int, g: int, b: int, bg: bool = False) -> str:
        """
        Get ANSI color code for RGB values.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            bg: Use background color (default: foreground)
            
        Returns:
            ANSI color code string
        """
        color_type = 48 if bg else 38
        return f"\033[{color_type};2;{r};{g};{b}m"


class HtmlFormatter:
    """Format ASCII/Braille art as HTML."""
    
    @staticmethod
    def format(result: AsciiArtResult, original_image: Optional[Image.Image] = None, 
              colorize: bool = True, title: str = "ASCII Art") -> str:
        """
        Format ASCII art as HTML.
        
        Args:
            result: AsciiArtResult containing the ASCII art
            original_image: Original image for color reference
            colorize: Apply color to ASCII characters
            title: HTML page title
            
        Returns:
            HTML string
        """
        # Start HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{title}</title>
    <style>
        body {{
            font-family: monospace;
            line-height: 1;
            white-space: pre;
            background-color: #000;
            color: #fff;
            margin: 0;
            padding: 20px;
        }}
        .ascii-art {{
            font-size: 1px;
            letter-spacing: 0px;
            line-height: 1;
        }}
        .ascii-art-colorized {{
            image-rendering: pixelated;
        }}
    </style>
</head>
<body>
    <div class="ascii-art">
"""
        
        if colorize and original_image:
            # Colorized HTML with span elements
            ascii_lines = result.art.split('\n')
            ascii_height = len(ascii_lines)
            ascii_width = len(ascii_lines[0]) if ascii_height > 0 else 0
            
            # Resize original image to match ASCII dimensions
            resized = original_image.resize((ascii_width, ascii_height), Image.LANCZOS)
            rgb_array = np.array(resized)
            
            for y, line in enumerate(ascii_lines):
                html += f"        <div>\n"
                for x, char in enumerate(line):
                    if x < rgb_array.shape[1] and y < rgb_array.shape[0]:
                        r, g, b = rgb_array[y, x]
                        html += f"            <span style='color: rgb({r}, {g}, {b})'>{HtmlFormatter._escape_html(char)}</span>\n"
                    else:
                        html += f"            <span>{HtmlFormatter._escape_html(char)}</span>\n"
                html += f"        </div>\n"
        else:
            # Plain text HTML
            html += f"        <pre>{HtmlFormatter._escape_html(result.art)}</pre>\n"
        
        # End HTML
        html += f"""    </div>
</body>
</html>"""
        
        return html
    
    @staticmethod
    def _escape_html(text: str) -> str:
        """
        Escape HTML special characters.
        
        Args:
            text: Text to escape
            
        Returns:
            HTML-escaped text
        """
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
