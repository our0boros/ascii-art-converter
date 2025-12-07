#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Braille Generation
=======================================================
This module contains the BrailleGenerator class for converting images to Braille art.
"""

from PIL import Image
import numpy as np
from ascii_art_converter.constants import BRAILLE_BASE, BRAILLE_DOTS


class BrailleGenerator:
    """Generate Braille art from images."""
    
    @staticmethod
    def image_to_braille(image: Image.Image, width: int = None, height: int = None, 
                       dither: bool = False, invert: bool = False, threshold: float = 0.5) -> str:
        """
        Convert an image to Braille art.
        
        Args:
            image: Input image
            width: Output width in Braille characters
            height: Output height in Braille characters
            dither: Apply dithering
            invert: Invert colors
            threshold: Threshold for dot activation (0-1)
            
        Returns:
            Braille art string
        """
        # Resize image
        resized = BrailleGenerator._resize_image(image, width, height)
        
        # Convert to grayscale
        gray = resized.convert('L')
        
        # Apply dithering if requested
        if dither:
            gray = BrailleGenerator._apply_dithering(gray)
        
        # Convert to numpy array
        arr = np.array(gray)
        
        # Invert if requested
        if invert:
            arr = 255 - arr
        
        # Generate Braille
        return BrailleGenerator._generate_braille_from_array(arr, threshold=threshold)
    
    @staticmethod
    def _resize_image(image: Image.Image, width: int, height: int) -> Image.Image:
        """
        Resize image to fit Braille dimensions.
        Braille characters represent 2x4 pixel blocks.
        """
        if width is None and height is None:
            # Use original dimensions with Braille aspect ratio
            width = image.width // 2
            height = image.height // 4
        elif width is None:
            # Calculate width based on height
            width = int(image.width * (height * 4) / image.height / 2)
        elif height is None:
            # Calculate height based on width
            height = int(image.height * (width * 2) / image.width / 4)
        
        # Convert to Braille pixel dimensions (2x4 per character)
        braille_width = width * 2
        braille_height = height * 4
        
        return image.resize((braille_width, braille_height), Image.LANCZOS)
    
    @staticmethod
    def _apply_dithering(image: Image.Image) -> Image.Image:
        """Apply Floyd-Steinberg dithering to an image."""
        arr = np.array(image, dtype=np.float64)
        h, w = arr.shape
        
        for y in range(h):
            for x in range(w):
                old_pixel = arr[y, x]
                new_pixel = 255.0 if old_pixel > 128 else 0.0
                arr[y, x] = new_pixel
                error = old_pixel - new_pixel
                
                # Distribute error to neighboring pixels
                if x + 1 < w:
                    arr[y, x + 1] += error * 7 / 16
                if x - 1 >= 0 and y + 1 < h:
                    arr[y + 1, x - 1] += error * 3 / 16
                if y + 1 < h:
                    arr[y + 1, x] += error * 5 / 16
                if x + 1 < w and y + 1 < h:
                    arr[y + 1, x + 1] += error * 1 / 16
        
        return Image.fromarray(arr.astype(np.uint8))
    
    @staticmethod
    def _generate_braille_from_array(arr: np.ndarray, threshold: float = 0.5) -> str:
        """
        Generate Braille characters from a numpy array.
        
        Each Braille character represents a 2x4 pixel block:
        [0] [1]
        [2] [3]
        [4] [5]
        [6] [7]
        """
        h, w = arr.shape
        
        # Ensure dimensions are multiples of 2 and 4
        new_h = (h // 4) * 4
        new_w = (w // 2) * 2
        arr = arr[:new_h, :new_w]
        
        braille_lines = []
        
        # Process each 4x2 pixel block
        for y in range(0, new_h, 4):
            line = []
            for x in range(0, new_w, 2):
                # Get the 4x2 pixel block
                block = arr[y:y+4, x:x+2]
                
                # Determine which dots to activate
                braille_char = BrailleGenerator._block_to_braille(block, threshold=threshold)
                line.append(braille_char)
            
            braille_lines.append(''.join(line))
        
        return '\n'.join(braille_lines)
    
    @staticmethod
    def _block_to_braille(block: np.ndarray, threshold: float = 0.5) -> str:
        """
        Convert a 2x4 pixel block to a Braille character.
        
        Args:
            block: 4x2 numpy array representing the pixel block
            threshold: Threshold for dot activation (0-1)
            
        Returns:
            Braille character
        """
        if block.shape != (4, 2):
            raise ValueError(f"Block must be 4x2, got {block.shape}")
        
        # Convert threshold from 0-1 range to 0-255
        pixel_threshold = threshold * 255
        
        # Calculate dot pattern
        dot_pattern = 0
        for y, x, value in BRAILLE_DOTS:
            if block[x, y] >= pixel_threshold:  # Swap x and y indices to match block dimensions
                dot_pattern |= value
        
        # Convert to Braille character
        return chr(BRAILLE_BASE + dot_pattern)
