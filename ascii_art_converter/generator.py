#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - ASCII Generation
======================================================
This module contains the AsciiArtGenerator class for converting images to ASCII art.
"""

from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, Optional, List
from ascii_art_converter.constants import RenderMode, EdgeDetector, DitherMethod
from ascii_art_converter.character_sets import CharacterSet
from ascii_art_converter.config import AsciiArtConfig, AsciiArtResult
from ascii_art_converter.complexity import ComplexityAnalyzer
from ascii_art_converter.edge_detection import EdgeProcessor
from ascii_art_converter.braille import BrailleGenerator


class AsciiArtGenerator:
    """Generate ASCII art from images."""
    
    def __init__(self):
        self.complexity_analyzer = ComplexityAnalyzer()
        self.edge_processor = EdgeProcessor()
        self.braille_generator = BrailleGenerator()
    
    def convert(self, image: Image.Image, config: AsciiArtConfig) -> AsciiArtResult:
        """
        Convert an image to ASCII/Braille art using the given configuration.
        
        Args:
            image: Input image
            config: Configuration for ASCII art generation
            
        Returns:
            AsciiArtResult containing the generated art and metadata
        """
        # Analyze complexity for automatic sizing
        if config.width is None:
            complexity = self.complexity_analyzer.analyze(image)
            config.width = self._calculate_auto_width(complexity, image.width)
        else:
            complexity = 0.0
        
        # Preprocess image
        processed_image = self._preprocess_image(image, config)
        
        # Get character set based on render mode
        from .character_sets import CharacterSet
        character_set = config.charset
        if isinstance(character_set, str):
            character_set = CharacterSet.get_preset(character_set.upper())
        
        # For edge mode, use edge_charset instead
        edge_character_set = config.edge_charset
        if isinstance(edge_character_set, str):
            # Check if it's a preset name (contains no spaces) or already a character set
            if ' ' not in edge_character_set:
                edge_character_set = CharacterSet.get_preset(edge_character_set)

        # Generate art based on render mode
        if config.mode == RenderMode.BRAILLE:
            art = self.braille_generator.image_to_braille(
                processed_image,
                width=config.width,
                dither=config.dither_method != DitherMethod.NONE,
                invert=config.invert,
                threshold=config.braille_threshold
            )
        elif config.mode == RenderMode.EDGE:
            # Edge mode uses specialized edge detection algorithm
            art = self._image_to_edge(
                processed_image,
                config.width,
                edge_character_set,
                config.edge_detector,
                config.edge_threshold,
                config.edge_sigma,
                char_aspect_ratio=config.char_aspect_ratio
            )
        else:
            # ASCII generation (density mode)
            art = self._image_to_ascii(
                processed_image,
                config.width,
                character_set,
                config.dither_method != DitherMethod.NONE,
                config.invert,
                char_aspect_ratio=config.char_aspect_ratio
            )
        
        # Create result
        result = AsciiArtResult(
            text=art,
            lines=art.split('\n') if art else [],
            width=config.width,
            height=len(art.split('\n')) if art else 0,
            original_size=image.size,
            complexity_score=complexity
        )
        
        # Generate color data if colorize is enabled
        if config.colorize:
            colors = self._generate_color_data(image, config, result)
            result.colors = colors
        
        return result
    
    def _calculate_auto_width(self, complexity: float, original_width: int) -> int:
        """Calculate optimal width based on image complexity."""
        # More complex images get wider ASCII art for better detail
        base_width = 80
        max_width = 200
        
        # Map complexity (0-1) to width (base_width - max_width)
        auto_width = int(base_width + (max_width - base_width) * complexity)
        
        # Ensure minimum width
        return max(base_width, auto_width)
    
    def _preprocess_image(self, image: Image.Image, config: AsciiArtConfig) -> Image.Image:
        """
        Preprocess image before conversion.
        
        Args:
            image: Input image
            config: Configuration
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Adjust brightness and contrast
        if config.brightness != 1.0:
            enhancer = ImageEnhance.Brightness(processed)
            processed = enhancer.enhance(config.brightness)
        
        if config.contrast != 1.0:
            enhancer = ImageEnhance.Contrast(processed)
            processed = enhancer.enhance(config.contrast)
        
        # Apply edge detection if render mode is edge
        if config.mode == RenderMode.EDGE:
            processed = self.edge_processor.detect_edges(processed, config.edge_detector)

        # Apply smoothing if edge sigma is greater than 0
        if config.edge_sigma > 0:
            processed = processed.filter(ImageFilter.GaussianBlur(radius=config.edge_sigma))
        
        return processed
    
    def _image_to_ascii(self, image: Image.Image, width: int, character_set: CharacterSet,
                       dither: bool = False, invert: bool = False, char_aspect_ratio: float = 0.5) -> str:
        """
        Convert an image to ASCII art.
        
        Args:
            image: Input image
            width: Output width in characters
            character_set: Character set to use
            dither: Apply dithering
            invert: Invert colors
            char_aspect_ratio: Character aspect ratio (width/height)
            
        Returns:
            ASCII art string
        """
        # Resize image with aspect ratio
        resized = self._resize_image(image, width, char_aspect_ratio)
        
        # Convert to grayscale
        gray = resized.convert('L')
        
        # Apply dithering if requested
        if dither:
            gray = self._apply_dithering(gray)
        
        # Convert to numpy array
        arr = np.array(gray)
        
        # Invert if requested
        if invert:
            arr = 255 - arr
        
        # Generate ASCII
        return self._generate_ascii_from_array(arr, character_set)
    
    def _resize_image(self, image: Image.Image, width: int, char_aspect_ratio: float = 0.5) -> Image.Image:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            width: Output width in characters
            char_aspect_ratio: Character aspect ratio (width/height)
            
        Returns:
            Resized image
        """
        # Calculate height maintaining aspect ratio
        image_aspect_ratio = image.height / image.width
        # Adjust for character aspect ratio (width/height of monospace char)
        height = int(width * image_aspect_ratio * char_aspect_ratio)
        
        return image.resize((width, height), Image.LANCZOS)
    
    def _apply_dithering(self, image: Image.Image) -> Image.Image:
        """
        Apply Floyd-Steinberg dithering to an image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Dithered image
        """
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
    
    def _generate_ascii_from_array(self, arr: np.ndarray, character_set: str) -> str:
        """
        Generate ASCII art from a numpy array.
        
        Args:
            arr: Grayscale image array
            character_set: Character set string to use
            
        Returns:
            ASCII art string
        """
        chars = character_set
        char_count = len(chars)
        
        # Normalize array to 0-1 range
        normalized = arr / 255.0
        
        # Map pixel values to characters
        ascii_array = np.vectorize(lambda x: chars[min(char_count - 1, int(x * char_count))])(normalized)
        
        # Convert to string
        lines = [''.join(row) for row in ascii_array]
        return '\n'.join(lines)
    
    def _image_to_edge(self, image: Image.Image, width: int, edge_charset: str, detector: EdgeDetector,
                      threshold: float, sigma: float, char_aspect_ratio: float = 0.5) -> str:
        """
        Convert an image to edge-based ASCII art.
        
        Args:
            image: Input image
            width: Output width in characters
            edge_charset: Edge character set to use
            detector: Edge detection algorithm
            threshold: Edge detection threshold (0-1)
            sigma: Gaussian blur sigma for edge detection
            char_aspect_ratio: Character aspect ratio (width/height)
            
        Returns:
            Edge-based ASCII art string
        """
        # Resize image with aspect ratio
        resized = self._resize_image(image, width, char_aspect_ratio)
        
        # Get edge magnitude and direction using EdgeProcessor
        magnitude, direction = self.edge_processor.detect(resized, detector, sigma)
        
        # Normalize magnitude to 0-1 range
        if magnitude.max() > 0:
            normalized_magnitude = magnitude / magnitude.max()
        else:
            normalized_magnitude = magnitude.astype(np.float64)
        
        # Generate edge ASCII using direction-based character mapping
        h, w = normalized_magnitude.shape
        ascii_art = []
        
        for y in range(h):
            line = []
            for x in range(w):
                mag = normalized_magnitude[y, x]
                dir_rad = direction[y, x]
                
                # Get appropriate edge character based on magnitude and direction
                char = self.edge_processor.get_edge_char(mag, dir_rad, threshold, edge_charset)
                line.append(char)
            ascii_art.append(''.join(line))
        
        return '\n'.join(ascii_art)
    
    def _generate_color_data(self, original_image: Image.Image, config: AsciiArtConfig, result: AsciiArtResult) -> List[List[str]]:
        """
        Generate color data for ASCII art.
        
        Args:
            original_image: Original input image
            config: Configuration
            result: AsciiArtResult with generated ASCII art
            
        Returns:
            List of lists containing hex color codes for each character
        """
        if not result.lines:
            return []
        
        # Calculate target dimensions
        target_width = result.width
        target_height = result.height
        
        # Resize original image to match ASCII dimensions
        resized = original_image.resize((target_width, target_height), Image.LANCZOS)
        
        # Convert to RGB mode if not already
        if resized.mode != 'RGB':
            resized = resized.convert('RGB')
        
        # Convert to numpy array for easy access
        rgb_array = np.array(resized)
        
        # Generate color data based on sample mode
        colors = []
        for y, line in enumerate(result.lines):
            line_colors = []
            for x, char in enumerate(line):
                if x < rgb_array.shape[1] and y < rgb_array.shape[0]:
                    r, g, b = rgb_array[y, x]
                    hex_color = f"#{r:02x}{g:02x}{b:02x}"
                    line_colors.append(hex_color)
                else:
                    # Default to white if outside bounds
                    line_colors.append("#ffffff")
            colors.append(line_colors)
        
        return colors
