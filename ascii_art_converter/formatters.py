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
                    
                    # Get appropriate ANSI color code based on color mode
                    color_code = AnsiColorFormatter.get_ansi_color_code(r, g, b, color_mode=color_mode)
                    colored_chars.append(f"{color_code}{char}\033[0m")
                else:
                    colored_chars.append(char)
            
            colored_lines.append(''.join(colored_chars))
        
        return '\n'.join(colored_lines)
    
    @staticmethod
    def get_ansi_color_code(r: int, g: int, b: int, bg: bool = False, color_mode: str = '24bit') -> str:
        """
        Get ANSI color code for RGB values.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            bg: Use background color (default: foreground)
            color_mode: Color mode (24bit, 256, 16)
            
        Returns:
            ANSI color code string
        """
        color_type = 48 if bg else 38
        
        if color_mode == '24bit':
            # 24-bit color mode
            return f"\033[{color_type};2;{r};{g};{b}m"
        elif color_mode == '256':
            # 256 color mode
            # Convert RGB to ANSI 256 color
            ansi_color = AnsiColorFormatter._rgb_to_ansi256(r, g, b)
            return f"\033[{color_type};5;{ansi_color}m"
        elif color_mode == '16':
            # 16 color mode
            # Convert RGB to ANSI 16 color
            # For 16-color mode, we don't use 38/48 prefix, just the color code directly
            return f"\033[{AnsiColorFormatter._rgb_to_ansi16(r, g, b, bg)}m"
        else:
            # Default to 24-bit if unknown mode
            return f"\033[{color_type};2;{r};{g};{b}m"
    
    @staticmethod
    def _rgb_to_ansi256(r: int, g: int, b: int) -> int:
        """
        Convert RGB values to ANSI 256 color code.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            
        Returns:
            ANSI 256 color code (0-255)
        """
        # Convert RGB to 6x6x6 color cube (216 colors)
        r6 = max(0, min(5, r // 43))  # 0-5
        g6 = max(0, min(5, g // 43))  # 0-5
        b6 = max(0, min(5, b // 43))  # 0-5
        cube_color = 16 + 36 * r6 + 6 * g6 + b6
        
        # Find the closest gray scale color (24 colors)
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
        gray4 = max(0, min(23, (gray - 8) // 10))  # 0-23
        gray_color = 232 + gray4
        
        # Calculate distances to find the closest color
        cube_rgb = ((r6 * 43 + 21), (g6 * 43 + 21), (b6 * 43 + 21))
        gray_rgb = (gray4 * 10 + 8, gray4 * 10 + 8, gray4 * 10 + 8)
        
        def distance(color1, color2):
            return sum((c1 - c2) ** 2 for c1, c2 in zip(color1, color2))
        
        cube_dist = distance((r, g, b), cube_rgb)
        gray_dist = distance((r, g, b), gray_rgb)
        
        return cube_color if cube_dist < gray_dist else gray_color
    
    @staticmethod
    def _rgb_to_ansi16(r: int, g: int, b: int, bg: bool = False) -> int:
        """
        Convert RGB values to ANSI 16 color code.
        
        Args:
            r: Red component (0-255)
            g: Green component (0-255)
            b: Blue component (0-255)
            bg: Use background color (default: foreground)
            
        Returns:
            ANSI 16 color code (30-37, 90-97 for foreground; 40-47, 100-107 for background)
        """
        # Calculate brightness
        brightness = (r * 299 + g * 587 + b * 114) // 1000
        
        # Basic 16 colors
        colors = {
            (0, 0, 0): 0,    # Black
            (128, 0, 0): 1,  # Red
            (0, 128, 0): 2,  # Green
            (128, 128, 0): 3,# Yellow
            (0, 0, 128): 4,  # Blue
            (128, 0, 128): 5,# Magenta
            (0, 128, 128): 6,# Cyan
            (192, 192, 192):7,# Light gray
        }
        
        # Find the closest basic color
        closest_color = min(colors.items(), key=lambda x: AnsiColorFormatter._distance((r, g, b), x[0]))[1]
        
        # Determine if we should use the bright version and calculate the final color code
        if brightness > 127:
            # Bright colors: 90-97 for foreground, 100-107 for background
            return (closest_color + 100) if bg else (closest_color + 90)
        else:
            # Basic colors: 30-37 for foreground, 40-47 for background
            return (closest_color + 40) if bg else (closest_color + 30)
    
    @staticmethod
    def _distance(color1, color2):
        """
        Calculate Euclidean distance between two RGB colors.
        """
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        return (r1 - r2) ** 2 + (g1 - g2) ** 2 + (b1 - b2) ** 2


class HtmlFormatter:
    """Format ASCII/Braille art as HTML."""
    
    @staticmethod
    def format(result: AsciiArtResult, original_image: Optional[Image.Image] = None, 
              colorize: bool = True, title: str = "ASCII Art", stretch_factor: float = 4.5) -> str:
        """
        Format ASCII art as HTML.
        
        Args:
            result: AsciiArtResult containing the ASCII art
            original_image: Original image for color reference
            colorize: Apply color to ASCII characters
            title: HTML page title
            stretch_factor: Stretch factor for horizontal scaling (default: 2.0)
            
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
            transform: scaleX({stretch_factor});
            transform-origin: left;
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
            ascii_lines = result.text.split('\n')
            ascii_height = len(ascii_lines)
            ascii_width = len(ascii_lines[0]) if ascii_height > 0 else 0
            
            # Resize original image to match ASCII dimensions
            resized = original_image.resize((ascii_width, ascii_height), Image.LANCZOS)
            # Convert to RGB if the image has alpha channel (RGBA)
            resized = resized.convert('RGB')
            rgb_array = np.array(resized)
            
            for y, line in enumerate(ascii_lines):
                html += f"        <div>\n"
                html += "            "
                for x, char in enumerate(line):
                    if x < rgb_array.shape[1] and y < rgb_array.shape[0]:
                        r, g, b = rgb_array[y, x]
                        html += f"<span style='color: rgb({r}, {g}, {b})'>{HtmlFormatter._escape_html(char)}</span>"
                    else:
                        html += f"<span>{HtmlFormatter._escape_html(char)}</span>"
                html += "\n"
                html += f"        </div>\n"
        else:
            # Plain text HTML
            html += f"        <pre>{HtmlFormatter._escape_html(result.text)}</pre>\n"
        
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
