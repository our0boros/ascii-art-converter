#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Batch Processing
======================================================
This module contains the BatchProcessor class for batch converting images to ASCII art.
"""

import os
from PIL import Image
from typing import List, Dict, Optional
import glob
from ascii_art_converter.config import AsciiArtConfig
from ascii_art_converter.generator import AsciiArtGenerator
from ascii_art_converter.formatters import AnsiColorFormatter, HtmlFormatter


class BatchProcessor:
    """Batch process images to ASCII/Braille art."""
    
    def __init__(self):
        self.generator = AsciiArtGenerator()
    
    def process_directory(self, input_dir: str, output_dir: str, config: AsciiArtConfig, 
                         extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif'],
                         format: str = 'txt', colorize: bool = False) -> Dict[str, bool]:
        """
        Process all images in a directory.
        
        Args:
            input_dir: Input directory containing images
            output_dir: Output directory for ASCII art files
            config: Configuration for ASCII art generation
            extensions: List of image extensions to process
            format: Output format ('txt', 'html', 'ansi')
            colorize: Apply color to ASCII art
            
        Returns:
            Dictionary mapping filenames to success status
        """
        results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Find all image files in input directory
        image_files = []
        for ext in extensions:
            pattern = os.path.join(input_dir, f'*{ext.lower()}')
            image_files.extend(glob.glob(pattern))
            pattern = os.path.join(input_dir, f'*{ext.upper()}')
            image_files.extend(glob.glob(pattern))
        
        # Process each image file
        for file_path in image_files:
            try:
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Open image
                with Image.open(file_path) as img:
                    # Convert to ASCII art
                    result = self.generator.convert(img, config)
                    
                    # Format output
                    if format == 'html':
                        output = HtmlFormatter.format(result, img, colorize)
                        output_file = os.path.join(output_dir, f'{name_without_ext}.html')
                    elif format == 'ansi':
                        output = AnsiColorFormatter.format_result(result, color_mode="256")
                        output_file = os.path.join(output_dir, f'{name_without_ext}.ansi')
                    else:  # txt
                        output = result.text
                        output_file = os.path.join(output_dir, f'{name_without_ext}.txt')
                    
                    # Save output
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(output)
                    
                    results[filename] = True
                    print(f"Processed: {filename} -> {os.path.basename(output_file)}")
                    
            except Exception as e:
                filename = os.path.basename(file_path)
                results[filename] = False
                print(f"Error processing {filename}: {str(e)}")
        
        return results
    
    def process_files(self, file_paths: List[str], output_dir: str, config: AsciiArtConfig, 
                     format: str = 'txt', colorize: bool = False) -> Dict[str, bool]:
        """
        Process a list of image files.
        
        Args:
            file_paths: List of image file paths
            output_dir: Output directory for ASCII art files
            config: Configuration for ASCII art generation
            format: Output format ('txt', 'html', 'ansi')
            colorize: Apply color to ASCII art
            
        Returns:
            Dictionary mapping filenames to success status
        """
        results = {}
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Process each image file
        for file_path in file_paths:
            try:
                filename = os.path.basename(file_path)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Open image
                with Image.open(file_path) as img:
                    # Convert to ASCII art
                    result = self.generator.convert(img, config)
                    
                    # Format output
                    if format == 'html':
                        output = HtmlFormatter.format(result, img, colorize)
                        output_file = os.path.join(output_dir, f'{name_without_ext}.html')
                    elif format == 'ansi':
                        output = AnsiColorFormatter.format_result(result, color_mode="256")
                        output_file = os.path.join(output_dir, f'{name_without_ext}.ansi')
                    else:  # txt
                        output = result.text
                        output_file = os.path.join(output_dir, f'{name_without_ext}.txt')
                    
                    # Save output
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(output)
                    
                    results[filename] = True
                    print(f"Processed: {filename} -> {os.path.basename(output_file)}")
                    
            except Exception as e:
                filename = os.path.basename(file_path)
                results[filename] = False
                print(f"Error processing {filename}: {str(e)}")
        
        return results
