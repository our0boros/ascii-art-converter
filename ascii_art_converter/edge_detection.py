#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Edge Detection
=====================================================
This module contains the EdgeProcessor class for detecting edges in images.
"""

from PIL import Image, ImageFilter
import numpy as np
from typing import Tuple
from enum import Enum
from ascii_art_converter.constants import EdgeDetector


class EdgeProcessor:
    """Process edges in images using various edge detection algorithms."""
    
    # Direction characters for edge detection (based on main.py implementation)
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
    def detect_edges(image: Image.Image, detector: EdgeDetector = EdgeDetector.SOBEL) -> Image.Image:
        """
        Detect edges in an image using the specified detector.
        
        Args:
            image: Input image
            detector: Edge detection algorithm to use
            
        Returns:
            Grayscale image with edges detected
        """
        gray = image.convert('L')
        
        if detector == EdgeDetector.SOBEL:
            return EdgeProcessor._sobel_edge_detection(gray)
        elif detector == EdgeDetector.CANNY:
            return EdgeProcessor._canny_edge_detection(gray)
        elif detector == EdgeDetector.LAPLACIAN:
            return EdgeProcessor._laplacian_edge_detection(gray)
        elif detector == EdgeDetector.FIND_EDGES:
            return EdgeProcessor._find_edges(gray)
        elif detector == EdgeDetector.NONE:
            return gray
        else:
            raise ValueError(f"Unknown edge detector: {detector}")
    
    @staticmethod
    def _sobel_edge_detection(image: Image.Image) -> Image.Image:
        """Sobel edge detection."""
        arr = np.array(image, dtype=np.float64)
        
        # Sobel kernels
        sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        
        try:
            from scipy import ndimage
            # Apply kernels
            gradient_x = ndimage.convolve(arr, sobel_x)
            gradient_y = ndimage.convolve(arr, sobel_y)
            
            # Calculate magnitude
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            
            # Normalize to 0-255
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8)
            
            return Image.fromarray(gradient_magnitude)
        except ImportError:
            # Fallback using PIL filters
            edges = image.filter(ImageFilter.FIND_EDGES)
            return edges
    
    @staticmethod
    def _canny_edge_detection(image: Image.Image) -> Image.Image:
        """Canny edge detection."""
        try:
            from scipy import ndimage
            from skimage import filters
            
            arr = np.array(image, dtype=np.float64)
            edges = filters.canny(arr)
            return Image.fromarray((edges * 255).astype(np.uint8))
        except ImportError:
            # Fallback
            edges = image.filter(ImageFilter.FIND_EDGES)
            edges = edges.filter(ImageFilter.SMOOTH)
            return edges
    
    @staticmethod
    def _laplacian_edge_detection(image: Image.Image) -> Image.Image:
        """Laplacian edge detection."""
        try:
            from scipy import ndimage
            
            arr = np.array(image, dtype=np.float64)
            laplacian = ndimage.laplace(arr)
            laplacian = np.absolute(laplacian)
            laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8)
            
            return Image.fromarray(laplacian)
        except ImportError:
            # Fallback
            return image.filter(ImageFilter.FIND_EDGES)
    
    @staticmethod
    def _find_edges(image: Image.Image) -> Image.Image:
        """Simple edge detection using PIL's FIND_EDGES filter."""
        return image.filter(ImageFilter.FIND_EDGES)
    
    @staticmethod
    def adjust_threshold(image: Image.Image, threshold: int = 128) -> Image.Image:
        """
        Adjust edge threshold.
        
        Args:
            image: Edge-detected image
            threshold: Threshold value (0-255)
            
        Returns:
            Binary image (edges black on white background)
        """
        arr = np.array(image)
        binary = (arr > threshold).astype(np.uint8) * 255
        return Image.fromarray(binary)
    
    @classmethod
    def get_edge_char(cls, magnitude: float, direction: float, threshold: float, charset: str) -> str:
        """
        Get appropriate character for edge based on magnitude and direction.
        
        Args:
            magnitude: Edge magnitude
            direction: Edge direction in radians
            threshold: Threshold value (0-1)
            charset: Character set to use
            
        Returns:
            Character representing the edge
        """
        if magnitude < threshold:
            return charset[0]  # None character at index 0
        
        # Normalize direction to 0-180 degrees
        deg = (np.degrees(direction) + 180) % 180
        
        # Map direction to character based on charset
        # Check if charset has at least 5 characters for direction mapping
        if len(charset) >= 5:
            # Use detailed charset with direction mapping
            if deg < 22.5 or deg >= 157.5:
                return charset[1]  # horizontal
            elif 22.5 <= deg < 67.5:
                return charset[3]  # diagonal /
            elif 67.5 <= deg < 112.5:
                return charset[2]  # vertical
            else:
                return charset[4]  # diagonal \
        else:
            # Simple charset - map magnitude to character
            idx = min(len(charset) - 1, int((1 - threshold + magnitude) * len(charset)))
            return charset[max(0, idx)]
    
    @classmethod
    def detect(cls, image: Image.Image, detector: EdgeDetector = EdgeDetector.SOBEL, sigma: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect edges and return magnitude and direction arrays.
        
        Args:
            image: Input image
            detector: Edge detection algorithm
            sigma: Gaussian blur sigma
            
        Returns:
            Tuple of (magnitude array, direction array)
        """
        gray = image.convert('L')
        
        if sigma > 0:
            gray = gray.filter(ImageFilter.GaussianBlur(radius=sigma))
        
        arr = np.array(gray, dtype=np.float64)
        
        if detector == EdgeDetector.SOBEL:
            # Sobel kernels
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            
            try:
                from scipy import ndimage
                # Apply kernels
                gradient_x = ndimage.convolve(arr, sobel_x)
                gradient_y = ndimage.convolve(arr, sobel_y)
            except ImportError:
                # Fallback implementation
                gradient_x = np.zeros_like(arr)
                gradient_y = np.zeros_like(arr)
                for i in range(1, arr.shape[0] - 1):
                    for j in range(1, arr.shape[1] - 1):
                        gradient_x[i, j] = (arr[i-1, j+1] + 2*arr[i, j+1] + arr[i+1, j+1]) - \
                                          (arr[i-1, j-1] + 2*arr[i, j-1] + arr[i+1, j-1])
                        gradient_y[i, j] = (arr[i+1, j-1] + 2*arr[i+1, j] + arr[i+1, j+1]) - \
                                          (arr[i-1, j-1] + 2*arr[i-1, j] + arr[i-1, j+1])
            
            # Calculate magnitude and direction
            magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            direction = np.arctan2(gradient_y, gradient_x)
            
            # Normalize magnitude to 0-255
            magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)
            
            return magnitude, direction
        elif detector == EdgeDetector.CANNY:
            # For Canny, return magnitude only (direction not calculated)
            try:
                from scipy import ndimage
                from skimage import filters
                
                edges = filters.canny(arr)
                magnitude = (edges * 255).astype(np.uint8)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
            except ImportError:
                # Fallback to FIND_EDGES
                edges = EdgeProcessor._find_edges(gray)
                magnitude = np.array(edges)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
        else:
            # For other detectors, use FIND_EDGES as fallback
            edges = EdgeProcessor.detect_edges(gray, detector)
            magnitude = np.array(edges)
            direction = np.zeros_like(magnitude, dtype=np.float64)
            
            return magnitude, direction
