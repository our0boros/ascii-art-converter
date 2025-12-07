#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - GIF Processing
====================================================
This module contains the GifToAscii class for converting GIF animations to ASCII/Braille animations.
"""

from PIL import Image, ImageSequence
import os
import numpy as np
from typing import List, Dict, Optional
from ascii_art_converter.config import AsciiArtConfig
from ascii_art_converter.generator import AsciiArtGenerator
from ascii_art_converter.formatters import HtmlFormatter


class GifToAscii:
    """Convert GIF animations to ASCII/Braille animations."""
    
    def __init__(self):
        self.generator = AsciiArtGenerator()
    
    def convert(self, gif_path: str, config: AsciiArtConfig, output_path: str = None, 
               format: str = 'html', colorize: bool = False) -> Optional[str]:
        """
        Convert a GIF animation to ASCII/Braille animation.
        
        Args:
            gif_path: Path to input GIF file
            config: Configuration for ASCII art generation
            output_path: Output path for animation
            format: Output format ('html', 'txt_sequence')
            colorize: Apply color to ASCII art
            
        Returns:
            Output path if successful, None otherwise
        """
        # Open GIF file
        with Image.open(gif_path) as gif:
            # Extract frames and durations
            frames = []
            durations = []
            
            for i, frame in enumerate(ImageSequence.Iterator(gif)):
                # Convert to RGB
                frame_rgb = frame.convert('RGB')
                frames.append(frame_rgb)
                
                # Get frame duration
                try:
                    duration = gif.info['duration']
                    durations.append(duration)
                except KeyError:
                    # Default to 100ms if duration not specified
                    durations.append(100)
            
            # Convert each frame to ASCII art
            ascii_frames = []
            for frame in frames:
                result = self.generator.convert(frame, config)
                ascii_frames.append((result.text, frame))
            
            # Generate output based on format
            if format == 'html':
                return self._generate_html_animation(ascii_frames, durations, output_path, colorize)
            elif format == 'txt_sequence':
                return self._generate_txt_sequence(ascii_frames, output_path)
            else:
                raise ValueError(f"Unknown format: {format}")
    
    def _generate_html_animation(self, frames: List[tuple], durations: List[int], 
                                output_path: str, colorize: bool) -> str:
        """
        Generate HTML animation from ASCII frames.
        
        Args:
            frames: List of tuples containing ASCII art and original frame
            durations: List of frame durations in milliseconds
            output_path: Output HTML file path
            colorize: Apply color to ASCII art
            
        Returns:
            Output file path
        """
        if output_path is None:
            output_path = 'animation.html'
        
        # Get the first frame's dimensions
        first_ascii, first_frame = frames[0]
        ascii_lines = first_ascii.split('\n')
        ascii_height = len(ascii_lines)
        ascii_width = len(ascii_lines[0]) if ascii_height > 0 else 0
        
        # Start HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>ASCII Animation</title>
    <style>
        body {{
            font-family: monospace;
            background-color: #000;
            color: #fff;
            margin: 0;
            padding: 20px;
            overflow: hidden;
        }}
        .animation-container {{
            font-size: 1px;
            line-height: 1;
            letter-spacing: 0;
        }}
        .frame {{
            display: none;
        }}
        .frame.active {{
            display: block;
        }}
    </style>
</head>
<body>
    <div class="animation-container">
"""
        
        # Add frames
        for i, (ascii_art, original_frame) in enumerate(frames):
            html += f"        <div class='frame' id='frame-{i}'>\n"
            
            if colorize:
                # Colorized frame
                ascii_lines = ascii_art.split('\n')
                
                # Resize original frame to match ASCII dimensions
                resized = original_frame.resize((ascii_width, ascii_height), Image.LANCZOS)
                rgb_array = np.array(resized)
                
                for y, line in enumerate(ascii_lines):
                    html += f"            <div>\n"
                    for x, char in enumerate(line):
                        if x < rgb_array.shape[1] and y < rgb_array.shape[0]:
                            r, g, b = rgb_array[y, x]
                            html += f"                <span style='color: rgb({r}, {g}, {b})'>{self._escape_html(char)}</span>\n"
                        else:
                            html += f"                <span>{self._escape_html(char)}</span>\n"
                    html += f"            </div>\n"
            else:
                # Plain text frame
                html += f"            <pre>{self._escape_html(ascii_art)}</pre>\n"
            
            html += f"        </div>\n"
        
        # Add JavaScript for animation
        html += f"""    </div>
    <script>
        const frames = document.querySelectorAll('.frame');
        const durations = {durations};
        let currentFrame = 0;
        
        function nextFrame() {{
            // Hide current frame
            frames[currentFrame].classList.remove('active');
            
            // Move to next frame
            currentFrame = (currentFrame + 1) % frames.length;
            
            // Show next frame
            frames[currentFrame].classList.add('active');
            
            // Set timeout for next frame
            setTimeout(nextFrame, durations[currentFrame]);
        }}
        
        // Start animation
        frames[0].classList.add('active');
        setTimeout(nextFrame, durations[0]);
    </script>
</body>
</html>"""
        
        # Write HTML file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path
    
    def _generate_txt_sequence(self, frames: List[tuple], output_dir: str) -> str:
        """
        Generate a sequence of text files for each frame.
        
        Args:
            frames: List of tuples containing ASCII art and original frame
            output_dir: Output directory for text files
            
        Returns:
            Output directory path
        """
        if output_dir is None:
            output_dir = 'animation_frames'
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Write each frame to a text file
        for i, (ascii_art, _) in enumerate(frames):
            frame_path = os.path.join(output_dir, f'frame_{i:04d}.txt')
            with open(frame_path, 'w', encoding='utf-8') as f:
                f.write(ascii_art)
        
        return output_dir
    
    def _escape_html(self, text: str) -> str:
        """
        Escape HTML special characters.
        
        Args:
            text: Text to escape
            
        Returns:
            HTML-escaped text
        """
        return text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
