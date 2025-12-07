#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Edge Detection
=====================================================
This module contains the EdgeProcessor class for detecting edges in images.
"""

from PIL import Image, ImageFilter
import numpy as np
from enum import Enum
from ascii_art_converter.constants import EdgeDetector


class EdgeProcessor:
    """Process edges in images using various edge detection algorithms."""
    
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
