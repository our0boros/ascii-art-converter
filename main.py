#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter
=====================================
A comprehensive library for converting PIL images to various text-art formats.

Features:
- Density-based ASCII art
- Braille pattern art
- Edge detection art
- Automatic size detection based on image complexity
- Optional colorization with hex RGB output
- ANSI terminal color output support
"""

from PIL import Image, ImageFilter, ImageOps
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Literal
from enum import Enum, auto
from dataclasses import dataclass, field
import math


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RenderMode(Enum):
    """Rendering mode for ASCII art generation."""
    DENSITY = auto()      # Character density based
    BRAILLE = auto()      # Unicode braille patterns
    EDGE = auto()         # Edge detection based


class EdgeDetector(Enum):
    """Edge detection algorithm."""
    SOBEL = auto()
    PREWITT = auto()
    LAPLACIAN = auto()
    CANNY = auto()
    SCHARR = auto()


class DitherMethod(Enum):
    """Dithering method for braille patterns."""
    NONE = auto()
    FLOYD_STEINBERG = auto()
    ORDERED = auto()
    ATKINSON = auto()


# =============================================================================
# CHARACTER SETS
# =============================================================================

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
    
    # Edge detection characters
    EDGE_BASIC: str = " -|/\\+LT7VXY"
    EDGE_DETAILED: str = " ─│╱╲┼┌┐└┘├┤┬┴╭╮╯╰"
    EDGE_ROUND: str = " ·─│╱╲┼╭╮╯╰"
    
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


# Braille pattern base and dot positions
BRAILLE_BASE = 0x2800
BRAILLE_DOTS = [
    (0, 0, 0x01), (0, 1, 0x02), (0, 2, 0x04), (0, 3, 0x40),
    (1, 0, 0x08), (1, 1, 0x10), (1, 2, 0x20), (1, 3, 0x80)
]


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class AsciiArtConfig:
    """Configuration for ASCII art generation."""
    
    # Size parameters
    width: Optional[int] = None              # Output width (auto if None)
    height: Optional[int] = None             # Output height (auto if None)
    max_width: int = 120                     # Maximum auto width
    min_width: int = 40                      # Minimum auto width
    
    # Character aspect ratio (width/height of monospace char)
    char_aspect_ratio: float = 0.5           # Typical terminal char ratio
    
    # Rendering mode
    mode: RenderMode = RenderMode.DENSITY
    
    # Character set
    charset: str = CharacterSet.STANDARD
    invert: bool = False                     # Invert brightness mapping
    
    # Edge detection settings
    edge_detector: EdgeDetector = EdgeDetector.SOBEL
    edge_threshold: float = 0.1              # Edge detection threshold (0-1)
    edge_charset: str = CharacterSet.EDGE_BASIC
    edge_sigma: float = 1.0                  # Gaussian blur sigma for edge
    
    # Braille settings
    braille_threshold: float = 0.5           # Threshold for braille dots
    dither_method: DitherMethod = DitherMethod.NONE
    
    # Color settings
    colorize: bool = False                   # Enable color output
    color_depth: int = 8                     # Color quantization depth
    color_sample_mode: Literal['center', 'average', 'dominant'] = 'average'
    
    # Enhancement settings
    contrast: float = 1.0                    # Contrast adjustment (0.5-2.0)
    brightness: float = 1.0                  # Brightness adjustment (0.5-2.0)
    gamma: float = 1.0                       # Gamma correction
    sharpness: float = 1.0                   # Sharpness enhancement
    
    # Complexity-based auto-sizing
    complexity_factor: float = 1.0           # Multiplier for auto-size
    
    # Background handling
    background_char: str = ' '               # Character for background
    transparent_threshold: int = 128         # Alpha threshold for transparency


@dataclass
class AsciiArtResult:
    """Result of ASCII art generation."""
    text: str                                          # The ASCII art text
    lines: List[str]                                   # Lines of ASCII art
    colors: Optional[List[List[str]]] = None           # Hex color array if colorized
    width: int = 0                                     # Output width
    height: int = 0                                    # Output height
    original_size: Tuple[int, int] = (0, 0)           # Original image size
    complexity_score: float = 0.0                      # Calculated complexity


# =============================================================================
# IMAGE COMPLEXITY ANALYSIS
# =============================================================================

class ComplexityAnalyzer:
    """Analyze image complexity for automatic sizing."""
    
    @staticmethod
    def calculate_edge_density(image: Image.Image) -> float:
        """Calculate edge density using Sobel operator."""
        gray = image.convert('L')
        arr = np.array(gray, dtype=np.float64) / 255.0
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        # Convolve
        from scipy import ndimage
        try:
            gx = ndimage.convolve(arr, sobel_x)
            gy = ndimage.convolve(arr, sobel_y)
            magnitude = np.sqrt(gx**2 + gy**2)
            return float(np.mean(magnitude))
        except ImportError:
            # Fallback without scipy
            edges = gray.filter(ImageFilter.FIND_EDGES)
            return float(np.mean(np.array(edges))) / 255.0
    
    @staticmethod
    def calculate_histogram_complexity(image: Image.Image) -> float:
        """Calculate complexity based on histogram distribution."""
        gray = image.convert('L')
        histogram = gray.histogram()
        total = sum(histogram)
        if total == 0:
            return 0.0
        
        # Calculate entropy
        entropy = 0.0
        for count in histogram:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
        
        # Normalize to 0-1 range (max entropy for 256 bins is 8)
        return entropy / 8.0
    
    @staticmethod
    def calculate_texture_complexity(image: Image.Image) -> float:
        """Calculate texture complexity using variance."""
        gray = image.convert('L')
        arr = np.array(gray, dtype=np.float64)
        
        # Local variance using sliding window
        window_size = min(16, min(image.size) // 4)
        if window_size < 2:
            return float(np.std(arr) / 128.0)
        
        # Simple block-based variance
        h, w = arr.shape
        block_h = h // window_size
        block_w = w // window_size
        
        if block_h == 0 or block_w == 0:
            return float(np.std(arr) / 128.0)
        
        variances = []
        for i in range(block_h):
            for j in range(block_w):
                block = arr[i*window_size:(i+1)*window_size, 
                           j*window_size:(j+1)*window_size]
                variances.append(np.var(block))
        
        return float(np.mean(variances) / (128.0 ** 2))
    
    @classmethod
    def analyze(cls, image: Image.Image) -> float:
        """
        Calculate overall image complexity score (0-1).
        Higher values indicate more complex images.
        """
        edge_score = cls.calculate_edge_density(image)
        histogram_score = cls.calculate_histogram_complexity(image)
        texture_score = cls.calculate_texture_complexity(image)
        
        # Weighted combination
        complexity = (
            edge_score * 0.4 +
            histogram_score * 0.3 +
            texture_score * 0.3
        )
        
        return min(1.0, max(0.0, complexity))


# =============================================================================
# EDGE DETECTION
# =============================================================================

class EdgeProcessor:
    """Edge detection and character mapping."""
    
    DIRECTION_CHARS = {
        'horizontal': '─',
        'vertical': '│',
        'diagonal_up': '╱',
        'diagonal_down': '╲',
        'corner_tl': '╭',
        'corner_tr': '╮',
        'corner_bl': '╰',
        'corner_br': '╯',
        'cross': '┼',
        'none': ' '
    }
    
    @staticmethod
    def sobel(image: Image.Image, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Sobel edge detection."""
        gray = image.convert('L')
        if sigma > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        arr = np.array(gray, dtype=np.float64)
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)
        
        try:
            from scipy import ndimage
            gx = ndimage.convolve(arr, sobel_x)
            gy = ndimage.convolve(arr, sobel_y)
        except ImportError:
            gx = EdgeProcessor._convolve2d(arr, sobel_x)
            gy = EdgeProcessor._convolve2d(arr, sobel_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return magnitude, direction
    
    @staticmethod
    def prewitt(image: Image.Image, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Prewitt edge detection."""
        gray = image.convert('L')
        if sigma > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        arr = np.array(gray, dtype=np.float64)
        
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
        
        gx = EdgeProcessor._convolve2d(arr, prewitt_x)
        gy = EdgeProcessor._convolve2d(arr, prewitt_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return magnitude, direction
    
    @staticmethod
    def scharr(image: Image.Image, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Scharr edge detection."""
        gray = image.convert('L')
        if sigma > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        arr = np.array(gray, dtype=np.float64)
        
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=np.float64)
        scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=np.float64)
        
        gx = EdgeProcessor._convolve2d(arr, scharr_x)
        gy = EdgeProcessor._convolve2d(arr, scharr_y)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx)
        
        return magnitude, direction
    
    @staticmethod
    def laplacian(image: Image.Image, sigma: float = 1.0) -> np.ndarray:
        """Apply Laplacian edge detection."""
        gray = image.convert('L')
        if sigma > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        arr = np.array(gray, dtype=np.float64)
        
        laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float64)
        
        return np.abs(EdgeProcessor._convolve2d(arr, laplacian_kernel))
    
    @staticmethod
    def canny(image: Image.Image, sigma: float = 1.0, 
              low_threshold: float = 0.1, high_threshold: float = 0.3) -> np.ndarray:
        """Apply Canny edge detection."""
        try:
            from scipy import ndimage
            
            gray = image.convert('L')
            if sigma > 0:
                gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
            
            arr = np.array(gray, dtype=np.float64)
            
            # Gradient
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            gx = ndimage.convolve(arr, sobel_x)
            gy = ndimage.convolve(arr, sobel_y)
            
            magnitude = np.sqrt(gx**2 + gy**2)
            direction = np.arctan2(gy, gx)
            
            # Non-maximum suppression
            magnitude_max = EdgeProcessor._non_max_suppression(magnitude, direction)
            
            # Double threshold
            max_val = magnitude_max.max()
            if max_val > 0:
                magnitude_max /= max_val
            
            strong = magnitude_max >= high_threshold
            weak = (magnitude_max >= low_threshold) & (magnitude_max < high_threshold)
            
            # Edge tracking by hysteresis
            result = np.zeros_like(magnitude_max)
            result[strong] = 1.0
            
            # Simple hysteresis
            from scipy.ndimage import binary_dilation
            dilated = binary_dilation(strong, iterations=2)
            result[weak & dilated] = 1.0
            
            return result * 255
            
        except ImportError:
            # Fallback to simple edge detection
            magnitude, _ = EdgeProcessor.sobel(image, sigma)
            return magnitude
    
    @staticmethod
    def _convolve2d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        """Simple 2D convolution without scipy."""
        kh, kw = kernel.shape
        pad_h, pad_w = kh // 2, kw // 2
        
        padded = np.pad(arr, ((pad_h, pad_h), (pad_w, pad_w)), mode='edge')
        output = np.zeros_like(arr)
        
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                output[i, j] = np.sum(
                    padded[i:i+kh, j:j+kw] * kernel
                )
        
        return output
    
    @staticmethod
    def _non_max_suppression(magnitude: np.ndarray, direction: np.ndarray) -> np.ndarray:
        """Non-maximum suppression for edge thinning."""
        rows, cols = magnitude.shape
        result = np.zeros_like(magnitude)
        
        # Convert direction to degrees
        angle = np.rad2deg(direction) % 180
        
        for i in range(1, rows - 1):
            for j in range(1, cols - 1):
                q, r = 255, 255
                
                # Angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = magnitude[i, j + 1]
                    r = magnitude[i, j - 1]
                # Angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = magnitude[i + 1, j - 1]
                    r = magnitude[i - 1, j + 1]
                # Angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = magnitude[i + 1, j]
                    r = magnitude[i - 1, j]
                # Angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = magnitude[i - 1, j - 1]
                    r = magnitude[i + 1, j + 1]
                
                if magnitude[i, j] >= q and magnitude[i, j] >= r:
                    result[i, j] = magnitude[i, j]
        
        return result
    
    @classmethod
    def get_edge_char(cls, magnitude: float, direction: float, 
                      threshold: float, charset: str) -> str:
        """Get appropriate character for edge based on magnitude and direction."""
        if magnitude < threshold:
            return ' '
        
        # Normalize direction to 0-180
        deg = (np.degrees(direction) + 180) % 180
        
        # Map direction to character
        if len(charset) >= 12:
            # Use detailed charset
            if deg < 22.5 or deg >= 157.5:
                return charset[1]  # horizontal
            elif 22.5 <= deg < 67.5:
                return charset[3]  # diagonal /
            elif 67.5 <= deg < 112.5:
                return charset[2]  # vertical
            else:
                return charset[4]  # diagonal \
        else:
            # Simple charset
            idx = min(len(charset) - 1, int((1 - threshold + magnitude) * len(charset)))
            return charset[max(0, idx)]
    
    @classmethod
    def detect(cls, image: Image.Image, config: AsciiArtConfig) -> Tuple[np.ndarray, np.ndarray]:
        """Apply edge detection based on config."""
        detector = config.edge_detector
        sigma = config.edge_sigma
        
        if detector == EdgeDetector.SOBEL:
            return cls.sobel(image, sigma)
        elif detector == EdgeDetector.PREWITT:
            return cls.prewitt(image, sigma)
        elif detector == EdgeDetector.SCHARR:
            return cls.scharr(image, sigma)
        elif detector == EdgeDetector.LAPLACIAN:
            mag = cls.laplacian(image, sigma)
            return mag, np.zeros_like(mag)
        elif detector == EdgeDetector.CANNY:
            mag = cls.canny(image, sigma, config.edge_threshold * 0.5, config.edge_threshold)
            return mag, np.zeros_like(mag)
        else:
            return cls.sobel(image, sigma)


# =============================================================================
# BRAILLE PATTERN GENERATOR
# =============================================================================

class BrailleGenerator:
    """Generate braille patterns from images."""
    
    @staticmethod
    def apply_dithering(arr: np.ndarray, method: DitherMethod) -> np.ndarray:
        """Apply dithering to the image array."""
        if method == DitherMethod.NONE:
            return arr
        
        result = arr.astype(np.float64).copy()
        h, w = result.shape
        
        if method == DitherMethod.FLOYD_STEINBERG:
            for y in range(h):
                for x in range(w):
                    old_val = result[y, x]
                    new_val = 255.0 if old_val > 127.5 else 0.0
                    result[y, x] = new_val
                    error = old_val - new_val
                    
                    if x + 1 < w:
                        result[y, x + 1] += error * 7 / 16
                    if y + 1 < h:
                        if x > 0:
                            result[y + 1, x - 1] += error * 3 / 16
                        result[y + 1, x] += error * 5 / 16
                        if x + 1 < w:
                            result[y + 1, x + 1] += error * 1 / 16
        
        elif method == DitherMethod.ORDERED:
            bayer_matrix = np.array([
                [0, 8, 2, 10],
                [12, 4, 14, 6],
                [3, 11, 1, 9],
                [15, 7, 13, 5]
            ]) / 16.0 * 255
            
            for y in range(h):
                for x in range(w):
                    threshold = bayer_matrix[y % 4, x % 4]
                    result[y, x] = 255.0 if result[y, x] > threshold else 0.0
        
        elif method == DitherMethod.ATKINSON:
            for y in range(h):
                for x in range(w):
                    old_val = result[y, x]
                    new_val = 255.0 if old_val > 127.5 else 0.0
                    result[y, x] = new_val
                    error = (old_val - new_val) / 8
                    
                    offsets = [(1, 0), (2, 0), (-1, 1), (0, 1), (1, 1), (0, 2)]
                    for dx, dy in offsets:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h:
                            result[ny, nx] += error
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    @staticmethod
    def pixels_to_braille(block: np.ndarray, threshold: float = 127.5) -> str:
        """Convert a 2x4 pixel block to a braille character."""
        if block.shape != (4, 2):
            # Resize block if needed
            block = np.resize(block, (4, 2))
        
        code = BRAILLE_BASE
        for dx, dy, bit in BRAILLE_DOTS:
            if dy < block.shape[0] and dx < block.shape[1]:
                if block[dy, dx] > threshold:
                    code |= bit
        
        return chr(code)
    
    @classmethod
    def generate(cls, image: Image.Image, config: AsciiArtConfig, 
                 target_width: int, target_height: int) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Generate braille art from image."""
        # Braille characters are 2x4 dots
        pixel_width = target_width * 2
        pixel_height = target_height * 4
        
        # Resize image
        resized = image.resize((pixel_width, pixel_height), Image.Resampling.LANCZOS)
        gray = resized.convert('L')
        
        # Apply enhancements
        arr = np.array(gray, dtype=np.uint8)
        
        # Apply dithering
        arr = cls.apply_dithering(arr, config.dither_method)
        
        # Invert if needed
        if config.invert:
            arr = 255 - arr
        
        # Calculate threshold
        threshold = config.braille_threshold * 255
        
        # Generate braille characters
        lines = []
        colors = [] if config.colorize else None
        
        # Get color image for colorization
        if config.colorize:
            color_img = resized.convert('RGB')
            color_arr = np.array(color_img)
        
        for y in range(0, pixel_height, 4):
            line = ""
            color_line = [] if config.colorize else None
            
            for x in range(0, pixel_width, 2):
                # Extract 2x4 block
                block = arr[y:y+4, x:x+2]
                if block.size == 0:
                    continue
                
                # Pad block if needed
                if block.shape != (4, 2):
                    padded = np.zeros((4, 2), dtype=np.uint8)
                    padded[:block.shape[0], :block.shape[1]] = block
                    block = padded
                
                char = cls.pixels_to_braille(block, threshold)
                line += char
                
                if config.colorize:
                    # Get color for this block
                    color_block = color_arr[y:y+4, x:x+2]
                    if config.color_sample_mode == 'center':
                        cy, cx = min(1, color_block.shape[0]-1), min(0, color_block.shape[1]-1)
                        r, g, b = color_block[cy, cx]
                    elif config.color_sample_mode == 'dominant':
                        r, g, b = cls._get_dominant_color(color_block)
                    else:  # average
                        r = int(np.mean(color_block[:, :, 0]))
                        g = int(np.mean(color_block[:, :, 1]))
                        b = int(np.mean(color_block[:, :, 2]))
                    
                    color_line.append(f"#{r:02X}{g:02X}{b:02X}")
            
            lines.append(line)
            if colors is not None:
                colors.append(color_line)
        
        return lines, colors
    
    @staticmethod
    def _get_dominant_color(block: np.ndarray) -> Tuple[int, int, int]:
        """Get the dominant color in a block."""
        if block.size == 0:
            return (0, 0, 0)
        
        # Simple approach: use the color furthest from gray
        flat = block.reshape(-1, 3)
        max_saturation = 0
        dominant = flat[0]
        
        for pixel in flat:
            r, g, b = pixel
            max_c = max(r, g, b)
            min_c = min(r, g, b)
            saturation = max_c - min_c
            if saturation > max_saturation:
                max_saturation = saturation
                dominant = pixel
        
        return tuple(dominant)


# =============================================================================
# MAIN ASCII ART GENERATOR
# =============================================================================

class AsciiArtGenerator:
    """Main class for generating ASCII art from images."""
    
    def __init__(self, config: Optional[AsciiArtConfig] = None):
        """Initialize with optional configuration."""
        self.config = config or AsciiArtConfig()
    
    def _calculate_auto_size(self, image: Image.Image) -> Tuple[int, int]:
        """Calculate automatic output size based on image complexity."""
        img_width, img_height = image.size
        
        # Analyze complexity
        complexity = ComplexityAnalyzer.analyze(image)
        
        # Base width calculation
        # More complex images get larger output
        complexity_multiplier = 0.5 + complexity * 1.0  # Range: 0.5 - 1.5
        complexity_multiplier *= self.config.complexity_factor
        
        # Calculate target width
        base_width = self.config.min_width + (
            (self.config.max_width - self.config.min_width) * complexity_multiplier
        )
        target_width = int(min(self.config.max_width, max(self.config.min_width, base_width)))
        
        # Calculate height maintaining aspect ratio
        aspect_ratio = img_height / img_width
        target_height = int(target_width * aspect_ratio * self.config.char_aspect_ratio)
        
        # Ensure minimum height
        target_height = max(10, target_height)
        
        return target_width, target_height
    
    def _preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing enhancements to the image."""
        img = image.copy()
        
        # Handle transparency
        if img.mode == 'RGBA':
            # Create background
            background = Image.new('RGB', img.size, (255, 255, 255))
            # Get alpha channel
            alpha = img.split()[3]
            background.paste(img, mask=alpha)
            img = background
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Apply contrast
        if self.config.contrast != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(self.config.contrast)
        
        # Apply brightness
        if self.config.brightness != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(self.config.brightness)
        
        # Apply sharpness
        if self.config.sharpness != 1.0:
            from PIL import ImageEnhance
            enhancer = ImageEnhance.Sharpness(img)
            img = enhancer.enhance(self.config.sharpness)
        
        # Apply gamma correction
        if self.config.gamma != 1.0:
            arr = np.array(img, dtype=np.float64) / 255.0
            arr = np.power(arr, 1.0 / self.config.gamma)
            arr = (arr * 255).astype(np.uint8)
            img = Image.fromarray(arr)
        
        return img
    
    def _resize_image(self, image: Image.Image, 
                      target_width: int, target_height: int) -> Image.Image:
        """Resize image to target dimensions."""
        return image.resize((target_width, target_height), Image.Resampling.LANCZOS)
    
    def _generate_density(self, image: Image.Image, 
                          target_width: int, target_height: int) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Generate density-based ASCII art."""
        # Resize image
        resized = self._resize_image(image, target_width, target_height)
        
        # Convert to grayscale
        gray = resized.convert('L')
        gray_arr = np.array(gray)
        
        # Get charset
        charset = self.config.charset
        if self.config.invert:
            charset = charset[::-1]
        
        num_chars = len(charset)
        
        # Generate ASCII
        lines = []
        colors = [] if self.config.colorize else None
        
        # Get color data if needed
        if self.config.colorize:
            color_img = resized.convert('RGB')
            color_arr = np.array(color_img)
        
        for y in range(target_height):
            line = ""
            color_line = [] if self.config.colorize else None
            
            for x in range(target_width):
                # Get brightness value
                brightness = gray_arr[y, x]
                
                # Map to character
                char_idx = int((brightness / 255.0) * (num_chars - 1))
                char_idx = max(0, min(num_chars - 1, char_idx))
                char = charset[char_idx]
                
                line += char
                
                if self.config.colorize:
                    r, g, b = color_arr[y, x]
                    color_line.append(f"#{r:02X}{g:02X}{b:02X}")
            
            lines.append(line)
            if colors is not None:
                colors.append(color_line)
        
        return lines, colors
    
    def _generate_edge(self, image: Image.Image, 
                       target_width: int, target_height: int) -> Tuple[List[str], Optional[List[List[str]]]]:
        """Generate edge-detection based ASCII art."""
        # Resize image
        resized = self._resize_image(image, target_width, target_height)
        
        # Detect edges
        magnitude, direction = EdgeProcessor.detect(resized, self.config)
        
        # Normalize magnitude
        max_mag = np.max(magnitude)
        if max_mag > 0:
            magnitude = magnitude / max_mag
        
        # Get edge charset
        edge_charset = self.config.edge_charset
        threshold = self.config.edge_threshold
        
        # Generate ASCII
        lines = []
        colors = [] if self.config.colorize else None
        
        # Get color data if needed
        if self.config.colorize:
            color_img = resized.convert('RGB')
            color_arr = np.array(color_img)
        
        for y in range(target_height):
            line = ""
            color_line = [] if self.config.colorize else None
            
            for x in range(target_width):
                mag = magnitude[y, x]
                dir_val = direction[y, x] if direction.size > 0 else 0
                
                char = EdgeProcessor.get_edge_char(mag, dir_val, threshold, edge_charset)
                line += char
                
                if self.config.colorize:
                    r, g, b = color_arr[y, x]
                    color_line.append(f"#{r:02X}{g:02X}{b:02X}")
            
            lines.append(line)
            if colors is not None:
                colors.append(color_line)
        
        return lines, colors
    
    def generate(self, image: Image.Image) -> AsciiArtResult:
        """
        Generate ASCII art from a PIL Image.
        
        Args:
            image: PIL Image to convert
            
        Returns:
            AsciiArtResult containing the ASCII art and optional color data
        """
        original_size = image.size
        
        # Preprocess image
        processed = self._preprocess_image(image)
        
        # Calculate output dimensions
        if self.config.width is not None:
            target_width = self.config.width
            if self.config.height is not None:
                target_height = self.config.height
            else:
                aspect_ratio = image.height / image.width
                target_height = int(target_width * aspect_ratio * self.config.char_aspect_ratio)
        elif self.config.height is not None:
            target_height = self.config.height
            aspect_ratio = image.width / image.height
            target_width = int(target_height * aspect_ratio / self.config.char_aspect_ratio)
        else:
            target_width, target_height = self._calculate_auto_size(processed)
        
        # Calculate complexity score
        complexity_score = ComplexityAnalyzer.analyze(processed)
        
        # Generate based on mode
        if self.config.mode == RenderMode.DENSITY:
            lines, colors = self._generate_density(processed, target_width, target_height)
        elif self.config.mode == RenderMode.BRAILLE:
            lines, colors = BrailleGenerator.generate(processed, self.config, 
                                                       target_width, target_height)
        elif self.config.mode == RenderMode.EDGE:
            lines, colors = self._generate_edge(processed, target_width, target_height)
        else:
            lines, colors = self._generate_density(processed, target_width, target_height)
        
        # Build result
        text = '\n'.join(lines)
        
        return AsciiArtResult(
            text=text,
            lines=lines,
            colors=colors,
            width=target_width if self.config.mode != RenderMode.BRAILLE else len(lines[0]) if lines else 0,
            height=len(lines),
            original_size=original_size,
            complexity_score=complexity_score
        )


# =============================================================================
# ANSI COLOR OUTPUT
# =============================================================================

class AnsiColorFormatter:
    """Format ASCII art with ANSI color codes for terminal output."""
    
    # ANSI escape codes
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    
    @staticmethod
    def hex_to_rgb(hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    @staticmethod
    def rgb_to_ansi_24bit(r: int, g: int, b: int, foreground: bool = True) -> str:
        """Convert RGB to 24-bit ANSI color code (true color)."""
        code = 38 if foreground else 48
        return f"\033[{code};2;{r};{g};{b}m"
    
    @staticmethod
    def rgb_to_ansi_256(r: int, g: int, b: int, foreground: bool = True) -> str:
        """Convert RGB to 256-color ANSI code."""
        # Convert to 256-color palette
        if r == g == b:
            # Grayscale
            if r < 8:
                color = 16
            elif r > 248:
                color = 231
            else:
                color = round((r - 8) / 247 * 24) + 232
        else:
            # Color cube (6x6x6)
            color = 16 + (36 * round(r / 255 * 5)) + (6 * round(g / 255 * 5)) + round(b / 255 * 5)
        
        code = 38 if foreground else 48
        return f"\033[{code};5;{color}m"
    
    @staticmethod
    def rgb_to_ansi_16(r: int, g: int, b: int, foreground: bool = True) -> str:
        """Convert RGB to 16-color ANSI code."""
        # Calculate intensity
        intensity = (r + g + b) / 3
        bright = intensity > 127
        
        # Determine base color
        r_bit = 1 if r > 127 else 0
        g_bit = 1 if g > 127 else 0
        b_bit = 1 if b > 127 else 0
        
        color = r_bit + (g_bit << 1) + (b_bit << 2)
        
        if foreground:
            code = 90 + color if bright else 30 + color
        else:
            code = 100 + color if bright else 40 + color
        
        return f"\033[{code}m"
    
    @classmethod
    def format_result(cls, result: AsciiArtResult, 
                      color_mode: Literal['24bit', '256', '16'] = '24bit',
                      background: bool = False,
                      bold: bool = False) -> str:
        """
        Format ASCII art result with ANSI colors.
        
        Args:
            result: AsciiArtResult with color data
            color_mode: Color mode ('24bit', '256', or '16')
            background: Apply color to background instead of foreground
            bold: Apply bold styling
            
        Returns:
            String with ANSI color codes
        """
        if result.colors is None:
            return result.text
        
        output_lines = []
        
        for y, (line, color_line) in enumerate(zip(result.lines, result.colors)):
            output = ""
            
            if bold:
                output += cls.BOLD
            
            prev_color = None
            for x, (char, hex_color) in enumerate(zip(line, color_line)):
                # Only add color code if color changed
                if hex_color != prev_color:
                    r, g, b = cls.hex_to_rgb(hex_color)
                    
                    if color_mode == '24bit':
                        output += cls.rgb_to_ansi_24bit(r, g, b, not background)
                    elif color_mode == '256':
                        output += cls.rgb_to_ansi_256(r, g, b, not background)
                    else:  # 16
                        output += cls.rgb_to_ansi_16(r, g, b, not background)
                    
                    prev_color = hex_color
                
                output += char
            
            output += cls.RESET
            output_lines.append(output)
        
        return '\n'.join(output_lines)
    
    @classmethod
    def format_with_gradient(cls, result: AsciiArtResult,
                             start_color: str = "#FF0000",
                             end_color: str = "#0000FF",
                             direction: Literal['horizontal', 'vertical', 'diagonal'] = 'horizontal') -> str:
        """
        Apply a gradient color overlay to the ASCII art.
        
        Args:
            result: AsciiArtResult
            start_color: Starting hex color
            end_color: Ending hex color
            direction: Gradient direction
            
        Returns:
            String with ANSI gradient colors
        """
        sr, sg, sb = cls.hex_to_rgb(start_color)
        er, eg, eb = cls.hex_to_rgb(end_color)
        
        output_lines = []
        height = len(result.lines)
        
        for y, line in enumerate(result.lines):
            output = ""
            width = len(line)
            
            for x, char in enumerate(line):
                # Calculate gradient position
                if direction == 'horizontal':
                    t = x / max(1, width - 1)
                elif direction == 'vertical':
                    t = y / max(1, height - 1)
                else:  # diagonal
                    t = (x + y) / max(1, width + height - 2)
                
                # Interpolate color
                r = int(sr + (er - sr) * t)
                g = int(sg + (eg - sg) * t)
                b = int(sb + (eb - sb) * t)
                
                output += cls.rgb_to_ansi_24bit(r, g, b) + char
            
            output += cls.RESET
            output_lines.append(output)
        
        return '\n'.join(output_lines)


# =============================================================================
# HTML OUTPUT
# =============================================================================

class HtmlFormatter:
    """Format ASCII art as HTML with styling."""
    
    @staticmethod
    def format_result(result: AsciiArtResult,
                      font_size: str = "10px",
                      font_family: str = "monospace",
                      background_color: str = "#000000",
                      line_height: float = 1.0) -> str:
        """
        Format ASCII art result as HTML.
        
        Args:
            result: AsciiArtResult with optional color data
            font_size: CSS font size
            font_family: CSS font family
            background_color: Background color
            line_height: Line height multiplier
            
        Returns:
            HTML string
        """
        html = f"""<!DOCTYPE html>
<html>
<head>
    <style>
        .ascii-art {{
            font-family: {font_family};
            font-size: {font_size};
            line-height: {line_height};
            background-color: {background_color};
            white-space: pre;
            display: inline-block;
            padding: 10px;
        }}
        .ascii-art span {{
            display: inline;
        }}
    </style>
</head>
<body>
<div class="ascii-art">
"""
        
        if result.colors is None:
            # No colors, just use plain text
            for line in result.lines:
                escaped = line.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html += escaped + '\n'
        else:
            # Add colored spans
            for line, color_line in zip(result.lines, result.colors):
                prev_color = None
                span_open = False
                
                for char, hex_color in zip(line, color_line):
                    escaped_char = char.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                    
                    if hex_color != prev_color:
                        if span_open:
                            html += "</span>"
                        html += f'<span style="color:{hex_color}">'
                        span_open = True
                        prev_color = hex_color
                    
                    html += escaped_char
                
                if span_open:
                    html += "</span>"
                html += '\n'
        
        html += """</div>
</body>
</html>"""
        
        return html


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def image_to_ascii(image: Image.Image,
                   width: Optional[int] = None,
                   mode: Union[str, RenderMode] = 'density',
                   charset: str = 'standard',
                   colorize: bool = False,
                   invert: bool = False,
                   **kwargs) -> AsciiArtResult:
    """
    Convenience function to convert image to ASCII art.
    
    Args:
        image: PIL Image
        width: Output width (auto if None)
        mode: 'density', 'braille', or 'edge'
        charset: Character set name or custom string
        colorize: Enable color output
        invert: Invert brightness
        **kwargs: Additional config options
        
    Returns:
        AsciiArtResult
    """
    # Parse mode
    if isinstance(mode, str):
        mode_map = {
            'density': RenderMode.DENSITY,
            'braille': RenderMode.BRAILLE,
            'edge': RenderMode.EDGE
        }
        mode = mode_map.get(mode.lower(), RenderMode.DENSITY)
    
    # Get charset
    if charset in ['standard', 'detailed', 'blocks', 'simple', 'binary', 
                   'dots', 'geometric', 'edge_basic', 'edge_detailed']:
        charset = CharacterSet.get_preset(charset)
    
    # Build config
    config = AsciiArtConfig(
        width=width,
        mode=mode,
        charset=charset,
        colorize=colorize,
        invert=invert,
        **kwargs
    )
    
    # Generate
    generator = AsciiArtGenerator(config)
    return generator.generate(image)


def print_ascii(image: Image.Image,
                width: Optional[int] = None,
                mode: str = 'density',
                colorize: bool = True,
                color_mode: str = '24bit',
                **kwargs) -> None:
    """
    Print ASCII art to terminal with optional colors.
    
    Args:
        image: PIL Image
        width: Output width (auto if None)
        mode: 'density', 'braille', or 'edge'
        colorize: Enable color output
        color_mode: '24bit', '256', or '16'
        **kwargs: Additional config options
    """
    result = image_to_ascii(image, width=width, mode=mode, colorize=colorize, **kwargs)
    
    if colorize and result.colors is not None:
        output = AnsiColorFormatter.format_result(result, color_mode=color_mode)
        print(output)
    else:
        print(result.text)


def save_html(image: Image.Image,
              output_path: str,
              width: Optional[int] = None,
              mode: str = 'density',
              colorize: bool = True,
              **kwargs) -> None:
    """
    Save ASCII art as HTML file.
    
    Args:
        image: PIL Image
        output_path: Output file path
        width: Output width (auto if None)
        mode: 'density', 'braille', or 'edge'
        colorize: Enable color output
        **kwargs: Additional config options
    """
    result = image_to_ascii(image, width=width, mode=mode, colorize=colorize, **kwargs)
    html = HtmlFormatter.format_result(result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# =============================================================================
# BATCH PROCESSING
# =============================================================================

class BatchProcessor:
    """Process multiple images with the same settings."""
    
    def __init__(self, config: Optional[AsciiArtConfig] = None):
        """Initialize with optional configuration."""
        self.generator = AsciiArtGenerator(config)
    
    def process_files(self, input_paths: List[str], 
                      output_dir: Optional[str] = None,
                      output_format: Literal['txt', 'html', 'ansi'] = 'txt') -> List[AsciiArtResult]:
        """
        Process multiple image files.
        
        Args:
            input_paths: List of input file paths
            output_dir: Output directory (None for no file output)
            output_format: Output format
            
        Returns:
            List of AsciiArtResult objects
        """
        import os
        
        results = []
        
        for path in input_paths:
            try:
                image = Image.open(path)
                result = self.generator.generate(image)
                results.append(result)
                
                if output_dir:
                    base_name = os.path.splitext(os.path.basename(path))[0]
                    
                    if output_format == 'txt':
                        output_path = os.path.join(output_dir, f"{base_name}.txt")
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(result.text)
                    elif output_format == 'html':
                        output_path = os.path.join(output_dir, f"{base_name}.html")
                        html = HtmlFormatter.format_result(result)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(html)
                    elif output_format == 'ansi':
                        output_path = os.path.join(output_dir, f"{base_name}.ansi")
                        ansi = AnsiColorFormatter.format_result(result)
                        with open(output_path, 'w', encoding='utf-8') as f:
                            f.write(ansi)
                            
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        return results


# =============================================================================
# ANIMATION SUPPORT
# =============================================================================

class GifToAscii:
    """Convert animated GIFs to ASCII art frames."""
    
    def __init__(self, config: Optional[AsciiArtConfig] = None):
        """Initialize with optional configuration."""
        self.generator = AsciiArtGenerator(config)
    
    def extract_frames(self, gif_path: str) -> List[AsciiArtResult]:
        """
        Extract all frames from a GIF and convert to ASCII.
        
        Args:
            gif_path: Path to GIF file
            
        Returns:
            List of AsciiArtResult for each frame
        """
        gif = Image.open(gif_path)
        frames = []
        
        try:
            while True:
                frame = gif.copy().convert('RGB')
                result = self.generator.generate(frame)
                frames.append(result)
                gif.seek(gif.tell() + 1)
        except EOFError:
            pass
        
        return frames
    
    def play_in_terminal(self, gif_path: str, 
                         delay: float = 0.1,
                         loops: int = -1,
                         color_mode: str = '24bit') -> None:
        """
        Play ASCII animation in terminal.
        
        Args:
            gif_path: Path to GIF file
            delay: Delay between frames in seconds
            loops: Number of loops (-1 for infinite)
            color_mode: ANSI color mode
        """
        import time
        import os
        
        frames = self.extract_frames(gif_path)
        if not frames:
            return
        
        loop_count = 0
        try:
            while loops == -1 or loop_count < loops:
                for frame in frames:
                    # Clear screen
                    os.system('cls' if os.name == 'nt' else 'clear')
                    
                    # Print frame
                    if frame.colors:
                        output = AnsiColorFormatter.format_result(frame, color_mode=color_mode)
                    else:
                        output = frame.text
                    print(output)
                    
                    time.sleep(delay)
                
                loop_count += 1
                
        except KeyboardInterrupt:
            print("\nAnimation stopped.")


# =============================================================================
# COMPARISON AND ANALYSIS TOOLS
# =============================================================================

class AsciiAnalyzer:
    """Tools for analyzing and comparing ASCII art output."""
    
    @staticmethod
    def calculate_accuracy(original: Image.Image, 
                          result: AsciiArtResult,
                          charset: str) -> float:
        """
        Calculate how accurately the ASCII represents the original image.
        
        Args:
            original: Original PIL Image
            result: Generated ASCII art result
            charset: Character set used
            
        Returns:
            Accuracy score (0-1)
        """
        # Resize original to match ASCII dimensions
        target = original.resize((result.width, result.height)).convert('L')
        target_arr = np.array(target) / 255.0
        
        # Reconstruct brightness from ASCII
        num_chars = len(charset)
        reconstructed = np.zeros((result.height, result.width))
        
        for y, line in enumerate(result.lines):
            for x, char in enumerate(line):
                if x < result.width:
                    idx = charset.find(char)
                    if idx >= 0:
                        reconstructed[y, x] = idx / (num_chars - 1)
        
        # Calculate mean squared error
        mse = np.mean((target_arr - reconstructed) ** 2)
        
        # Convert to accuracy (1 - normalized error)
        accuracy = 1.0 - np.sqrt(mse)
        
        return max(0.0, min(1.0, accuracy))
    
    @staticmethod
    def get_character_distribution(result: AsciiArtResult) -> Dict[str, int]:
        """
        Get distribution of characters in the ASCII art.
        
        Args:
            result: AsciiArtResult
            
        Returns:
            Dictionary mapping characters to counts
        """
        distribution = {}
        
        for line in result.lines:
            for char in line:
                distribution[char] = distribution.get(char, 0) + 1
        
        return dict(sorted(distribution.items(), key=lambda x: -x[1]))
    
    @staticmethod
    def get_color_palette(result: AsciiArtResult, 
                          num_colors: int = 16) -> List[str]:
        """
        Extract dominant colors from colorized ASCII art.
        
        Args:
            result: AsciiArtResult with color data
            num_colors: Number of colors to extract
            
        Returns:
            List of hex colors
        """
        if result.colors is None:
            return []
        
        # Collect all colors
        all_colors = []
        for row in result.colors:
            all_colors.extend(row)
        
        # Count occurrences
        color_counts = {}
        for color in all_colors:
            color_counts[color] = color_counts.get(color, 0) + 1
        
        # Sort by count and return top colors
        sorted_colors = sorted(color_counts.items(), key=lambda x: -x[1])
        return [color for color, _ in sorted_colors[:num_colors]]


# =============================================================================
# EXAMPLE USAGE AND DEMO
# =============================================================================

def demo():
    """Demonstrate the ASCII art generator capabilities."""
    
    print("=" * 60)
    print("ASCII Art Generator Demo")
    print("=" * 60)
    
    # Create a simple test image
    test_image = Image.new('RGB', (100, 100), color='white')
    
    # Draw some shapes
    from PIL import ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.ellipse([10, 10, 90, 90], fill='red', outline='black')
    draw.rectangle([30, 30, 70, 70], fill='blue')
    
    print("\n1. Density Mode (Standard charset):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='density', charset='standard')
    print(result.text)
    print(f"Complexity: {result.complexity_score:.2f}")
    
    print("\n2. Density Mode (Blocks charset):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='density', charset='blocks')
    print(result.text)
    
    print("\n3. Braille Mode:")
    print("-" * 40)
    result = image_to_ascii(test_image, width=20, mode='braille')
    print(result.text)
    
    print("\n4. Edge Detection Mode:")
    print("-" * 40)
    result = image_to_ascii(test_image, width=40, mode='edge', 
                           edge_threshold=0.15, edge_charset='edge_detailed')
    print(result.text)
    
    print("\n5. With Colors (showing hex values):")
    print("-" * 40)
    result = image_to_ascii(test_image, width=20, mode='density', colorize=True)
    if result.colors:
        print(f"Color array shape: {len(result.colors)}x{len(result.colors[0])}")
        print(f"Sample colors: {result.colors[0][:5]}")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def create_argument_parser():
    """Create command line argument parser."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Convert images to ASCII/Braille art',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s image.png                          # Basic conversion
  %(prog)s image.png -w 80                    # Set width to 80 chars
  %(prog)s image.png -m braille              # Use braille mode
  %(prog)s image.png -m edge -t 0.2          # Edge detection
  %(prog)s image.png -c -o output.html       # Colored HTML output
  %(prog)s image.png --color-mode 256        # 256-color terminal output
        """
    )
    
    # Input/Output
    parser.add_argument('input', nargs='?', help='Input image file')
    parser.add_argument('-o', '--output', help='Output file (txt, html, or ansi)')
    
    # Size options
    parser.add_argument('-w', '--width', type=int, help='Output width in characters')
    parser.add_argument('-H', '--height', type=int, help='Output height in characters')
    parser.add_argument('--max-width', type=int, default=120, help='Maximum auto width')
    parser.add_argument('--min-width', type=int, default=40, help='Minimum auto width')
    parser.add_argument('--char-ratio', type=float, default=0.5, 
                        help='Character aspect ratio (width/height)')
    
    # Mode options
    parser.add_argument('-m', '--mode', choices=['density', 'braille', 'edge'],
                        default='density', help='Rendering mode')
    
    # Character set options
    parser.add_argument('--charset', default='standard',
                        help='Character set: standard, detailed, blocks, simple, binary, dots, geometric')
    parser.add_argument('--custom-charset', help='Custom character string (dark to light)')
    parser.add_argument('-i', '--invert', action='store_true', help='Invert brightness')
    
    # Edge detection options
    parser.add_argument('-t', '--threshold', type=float, default=0.1,
                        help='Edge detection threshold (0-1)')
    parser.add_argument('--edge-detector', choices=['sobel', 'prewitt', 'laplacian', 'canny', 'scharr'],
                        default='sobel', help='Edge detection algorithm')
    parser.add_argument('--edge-sigma', type=float, default=1.0,
                        help='Gaussian blur sigma for edge detection')
    parser.add_argument('--edge-charset', default='edge_basic',
                        help='Character set for edge mode')
    
    # Braille options
    parser.add_argument('--braille-threshold', type=float, default=0.5,
                        help='Threshold for braille dots (0-1)')
    parser.add_argument('--dither', choices=['none', 'floyd_steinberg', 'ordered', 'atkinson'],
                        default='none', help='Dithering method for braille')
    
    # Color options
    parser.add_argument('-c', '--colorize', action='store_true', help='Enable color output')
    parser.add_argument('--color-mode', choices=['24bit', '256', '16'],
                        default='24bit', help='Terminal color mode')
    parser.add_argument('--color-sample', choices=['center', 'average', 'dominant'],
                        default='average', help='Color sampling method')
    
    # Enhancement options
    parser.add_argument('--contrast', type=float, default=1.0,
                        help='Contrast adjustment (0.5-2.0)')
    parser.add_argument('--brightness', type=float, default=1.0,
                        help='Brightness adjustment (0.5-2.0)')
    parser.add_argument('--gamma', type=float, default=1.0,
                        help='Gamma correction')
    parser.add_argument('--sharpness', type=float, default=1.0,
                        help='Sharpness enhancement')
    
    # Other options
    parser.add_argument('--complexity-factor', type=float, default=1.0,
                        help='Multiplier for auto-size complexity calculation')
    parser.add_argument('--demo', action='store_true', help='Run demo')
    parser.add_argument('--analyze', action='store_true', 
                        help='Show analysis of the generated ASCII art')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
    
    return parser


def main():
    """Main entry point for command line usage."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Run demo if requested
    if args.demo:
        demo()
        return
    
    # Check for input
    if not args.input:
        parser.print_help()
        return
    
    # Load image
    try:
        image = Image.open(args.input)
        if args.verbose:
            print(f"Loaded image: {args.input}")
            print(f"Size: {image.size}, Mode: {image.mode}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return
    
    # Parse mode
    mode_map = {
        'density': RenderMode.DENSITY,
        'braille': RenderMode.BRAILLE,
        'edge': RenderMode.EDGE
    }
    mode = mode_map[args.mode]
    
    # Parse edge detector
    edge_detector_map = {
        'sobel': EdgeDetector.SOBEL,
        'prewitt': EdgeDetector.PREWITT,
        'laplacian': EdgeDetector.LAPLACIAN,
        'canny': EdgeDetector.CANNY,
        'scharr': EdgeDetector.SCHARR
    }
    edge_detector = edge_detector_map[args.edge_detector]
    
    # Parse dither method
    dither_map = {
        'none': DitherMethod.NONE,
        'floyd_steinberg': DitherMethod.FLOYD_STEINBERG,
        'ordered': DitherMethod.ORDERED,
        'atkinson': DitherMethod.ATKINSON
    }
    dither_method = dither_map[args.dither]
    
    # Get charset
    charset = args.custom_charset if args.custom_charset else CharacterSet.get_preset(args.charset)
    edge_charset = CharacterSet.get_preset(args.edge_charset)
    
    # Build config
    config = AsciiArtConfig(
        width=args.width,
        height=args.height,
        max_width=args.max_width,
        min_width=args.min_width,
        char_aspect_ratio=args.char_ratio,
        mode=mode,
        charset=charset,
        invert=args.invert,
        edge_detector=edge_detector,
        edge_threshold=args.threshold,
        edge_charset=edge_charset,
        edge_sigma=args.edge_sigma,
        braille_threshold=args.braille_threshold,
        dither_method=dither_method,
        colorize=args.colorize,
        color_sample_mode=args.color_sample,
        contrast=args.contrast,
        brightness=args.brightness,
        gamma=args.gamma,
        sharpness=args.sharpness,
        complexity_factor=args.complexity_factor
    )
    
    # Generate ASCII art
    generator = AsciiArtGenerator(config)
    result = generator.generate(image)
    
    if args.verbose:
        print(f"Output size: {result.width}x{result.height}")
        print(f"Complexity score: {result.complexity_score:.3f}")
        print()
    
    # Handle output
    if args.output:
        ext = args.output.lower().split('.')[-1]
        
        if ext == 'html':
            html = HtmlFormatter.format_result(result)
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"Saved to {args.output}")
            
        elif ext == 'ansi':
            if result.colors:
                ansi = AnsiColorFormatter.format_result(result, color_mode=args.color_mode)
            else:
                ansi = result.text
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(ansi)
            print(f"Saved to {args.output}")
            
        else:  # txt or other
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write(result.text)
            print(f"Saved to {args.output}")
    else:
        # Print to terminal
        if args.colorize and result.colors:
            output = AnsiColorFormatter.format_result(result, color_mode=args.color_mode)
            print(output)
        else:
            print(result.text)
    
    # Show analysis if requested
    if args.analyze:
        print("\n" + "=" * 50)
        print("Analysis:")
        print("=" * 50)
        
        # Character distribution
        dist = AsciiAnalyzer.get_character_distribution(result)
        print("\nCharacter Distribution (top 10):")
        for i, (char, count) in enumerate(list(dist.items())[:10]):
            char_display = repr(char) if char in ' \t\n' else char
            print(f"  {char_display}: {count}")
        
        # Color palette
        if result.colors:
            palette = AsciiAnalyzer.get_color_palette(result, num_colors=8)
            print("\nDominant Colors:")
            for color in palette:
                print(f"  {color}")
        
        # Accuracy estimation
        accuracy = AsciiAnalyzer.calculate_accuracy(image, result, charset)
        print(f"\nEstimated Accuracy: {accuracy:.1%}")


# =============================================================================
# INTERACTIVE MODE
# =============================================================================

class InteractiveMode:
    """Interactive ASCII art preview with parameter adjustment."""
    
    def __init__(self, image: Image.Image):
        """Initialize with an image."""
        self.image = image
        self.config = AsciiArtConfig()
        self.generator = AsciiArtGenerator(self.config)
        self.result = None
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        self.generator = AsciiArtGenerator(self.config)
    
    def render(self) -> str:
        """Render current configuration."""
        self.result = self.generator.generate(self.image)
        return self.result.text
    
    def render_colored(self, color_mode: str = '24bit') -> str:
        """Render with colors."""
        self.config.colorize = True
        self.generator = AsciiArtGenerator(self.config)
        self.result = self.generator.generate(self.image)
        
        if self.result.colors:
            return AnsiColorFormatter.format_result(self.result, color_mode=color_mode)
        return self.result.text
    
    def run(self):
        """Run interactive mode."""
        import os
        
        print("Interactive ASCII Art Mode")
        print("=" * 50)
        print("Commands:")
        print("  w <num>     - Set width")
        print("  m <mode>    - Set mode (density/braille/edge)")
        print("  c <charset> - Set charset")
        print("  i           - Toggle invert")
        print("  color       - Toggle color")
        print("  t <num>     - Set threshold (edge mode)")
        print("  contrast <num> - Set contrast")
        print("  render      - Render current settings")
        print("  save <file> - Save to file")
        print("  q           - Quit")
        print()
        
        while True:
            try:
                cmd = input("> ").strip().split()
                if not cmd:
                    continue
                
                command = cmd[0].lower()
                
                if command == 'q' or command == 'quit':
                    break
                    
                elif command == 'w' and len(cmd) > 1:
                    self.update_config(width=int(cmd[1]))
                    print(f"Width set to {cmd[1]}")
                    
                elif command == 'm' and len(cmd) > 1:
                    mode_map = {
                        'density': RenderMode.DENSITY,
                        'braille': RenderMode.BRAILLE,
                        'edge': RenderMode.EDGE
                    }
                    if cmd[1] in mode_map:
                        self.update_config(mode=mode_map[cmd[1]])
                        print(f"Mode set to {cmd[1]}")
                    else:
                        print("Invalid mode. Use: density, braille, edge")
                        
                elif command == 'c' and len(cmd) > 1:
                    charset = CharacterSet.get_preset(cmd[1])
                    self.update_config(charset=charset)
                    print(f"Charset set to {cmd[1]}")
                    
                elif command == 'i':
                    self.config.invert = not self.config.invert
                    print(f"Invert: {self.config.invert}")
                    
                elif command == 'color':
                    self.config.colorize = not self.config.colorize
                    print(f"Colorize: {self.config.colorize}")
                    
                elif command == 't' and len(cmd) > 1:
                    self.update_config(edge_threshold=float(cmd[1]))
                    print(f"Threshold set to {cmd[1]}")
                    
                elif command == 'contrast' and len(cmd) > 1:
                    self.update_config(contrast=float(cmd[1]))
                    print(f"Contrast set to {cmd[1]}")
                    
                elif command == 'render':
                    os.system('cls' if os.name == 'nt' else 'clear')
                    if self.config.colorize:
                        print(self.render_colored())
                    else:
                        print(self.render())
                        
                elif command == 'save' and len(cmd) > 1:
                    if self.result is None:
                        self.render()
                    with open(cmd[1], 'w', encoding='utf-8') as f:
                        f.write(self.result.text)
                    print(f"Saved to {cmd[1]}")
                    
                else:
                    print("Unknown command. Type 'q' to quit.")
                    
            except KeyboardInterrupt:
                print("\nUse 'q' to quit.")
            except Exception as e:
                print(f"Error: {e}")


# =============================================================================
# PRESET CONFIGURATIONS
# =============================================================================

class Presets:
    """Predefined configuration presets."""
    
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
    def high_contrast_edge() -> AsciiArtConfig:
        """High contrast edge detection."""
        return AsciiArtConfig(
            mode=RenderMode.EDGE,
            edge_detector=EdgeDetector.CANNY,
            edge_threshold=0.15,
            edge_charset=CharacterSet.EDGE_DETAILED,
            contrast=1.4
        )
    
    @staticmethod
    def compact_braille() -> AsciiArtConfig:
        """Compact braille representation."""
        return AsciiArtConfig(
            mode=RenderMode.BRAILLE,
            max_width=60,
            dither_method=DitherMethod.FLOYD_STEINBERG,
            contrast=1.3
        )
    
    @staticmethod
    def social_media() -> AsciiArtConfig:
        """Configuration for social media sharing."""
        return AsciiArtConfig(
            width=40,
            charset=CharacterSet.STANDARD,
            contrast=1.2,
            char_aspect_ratio=0.5
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resize_to_fit(image: Image.Image, 
                  max_width: int = 800, 
                  max_height: int = 800) -> Image.Image:
    """Resize image to fit within max dimensions while maintaining aspect ratio."""
    width, height = image.size
    
    if width <= max_width and height <= max_height:
        return image
    
    ratio = min(max_width / width, max_height / height)
    new_size = (int(width * ratio), int(height * ratio))
    
    return image.resize(new_size, Image.Resampling.LANCZOS)


def create_comparison(image: Image.Image, 
                      width: int = 60) -> str:
    """Create a comparison of different rendering modes."""
    output = []
    
    output.append("=" * (width * 3 + 4))
    output.append("DENSITY MODE".center(width) + " | " + 
                  "BRAILLE MODE".center(width) + " | " + 
                  "EDGE MODE".center(width))
    output.append("=" * (width * 3 + 4))
    
    # Generate all modes
    density_result = image_to_ascii(image, width=width, mode='density')
    braille_result = image_to_ascii(image, width=width // 2, mode='braille')
    edge_result = image_to_ascii(image, width=width, mode='edge')
    
    # Combine side by side
    max_lines = max(len(density_result.lines), 
                    len(braille_result.lines), 
                    len(edge_result.lines))
    
    for i in range(max_lines):
        d_line = density_result.lines[i] if i < len(density_result.lines) else " " * width
        b_line = braille_result.lines[i] if i < len(braille_result.lines) else " " * (width // 2)
        e_line = edge_result.lines[i] if i < len(edge_result.lines) else " " * width
        
        # Pad lines to width
        d_line = d_line.ljust(width)[:width]
        b_line = b_line.ljust(width)[:width]
        e_line = e_line.ljust(width)[:width]
        
        output.append(f"{d_line} | {b_line} | {e_line}")
    
    output.append("=" * (width * 3 + 4))
    
    return '\n'.join(output)


def get_optimal_settings(image: Image.Image) -> AsciiArtConfig:
    """Analyze image and return optimal settings."""
    complexity = ComplexityAnalyzer.analyze(image)
    width, height = image.size
    aspect = height / width
    
    # Start with default config
    config = AsciiArtConfig()
    
    # Adjust based on complexity
    if complexity > 0.6:
        # High complexity - use detailed charset
        config.charset = CharacterSet.DETAILED
        config.max_width = 120
        config.sharpness = 1.2
    elif complexity < 0.3:
        # Low complexity - use simple charset
        config.charset = CharacterSet.SIMPLE
        config.max_width = 80
        config.contrast = 1.3
    else:
        config.charset = CharacterSet.STANDARD
        config.max_width = 100
    
    # Adjust for aspect ratio
    if aspect > 1.5:
        # Tall image
        config.max_width = 80
    elif aspect < 0.67:
        # Wide image
        config.max_width = 140
    
    # Check if edge mode might be better
    edge_score = ComplexityAnalyzer.calculate_edge_density(image)
    if edge_score > 0.4:
        # High edge content - might benefit from edge mode
        config.mode = RenderMode.EDGE
        config.edge_threshold = 0.1 + (1 - edge_score) * 0.2
    
    return config


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main classes
    'AsciiArtGenerator',
    'AsciiArtConfig',
    'AsciiArtResult',
    
    # Enums
    'RenderMode',
    'EdgeDetector',
    'DitherMethod',
    
    # Character sets
    'CharacterSet',
    
    # Processors
    'EdgeProcessor',
    'BrailleGenerator',
    'ComplexityAnalyzer',
    
    # Formatters
    'AnsiColorFormatter',
    'HtmlFormatter',
    
    # Convenience functions
    'image_to_ascii',
    'print_ascii',
    'save_html',
    
    # Utilities
    'BatchProcessor',
    'GifToAscii',
    'AsciiAnalyzer',
    'InteractiveMode',
    'Presets',
    
    # Helper functions
    'resize_to_fit',
    'create_comparison',
    'get_optimal_settings',
]


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == '__main__':
    main()

