#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Complexity Analysis
=========================================================
This module contains the ComplexityAnalyzer class for analyzing image complexity.
"""

from PIL import Image, ImageFilter
import numpy as np
import math


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
        try:
            from scipy import ndimage
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
