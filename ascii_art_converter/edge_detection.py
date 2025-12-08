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
        elif detector == EdgeDetector.PREWITT:
            return EdgeProcessor._prewitt_edge_detection(gray)
        elif detector == EdgeDetector.SCHARR:
            return EdgeProcessor._scharr_edge_detection(gray)
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
            gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8) if gradient_magnitude.max() > 0 else gradient_magnitude.astype(np.uint8)
            
            return Image.fromarray(gradient_magnitude)
        except Exception:
            # Fallback using PIL filters on any error
            print("Warning: Failed to use scipy.ndimage for Sobel. Falling back to PIL's FIND_EDGES filter.")
            edges = image.filter(ImageFilter.FIND_EDGES)
            return edges
    
    @staticmethod
    def _canny_edge_detection(image: Image.Image) -> Image.Image:
        """Canny edge detection."""
        try:
            from scipy import ndimage
            try:
                # Try importing from skimage.feature (newer versions)
                from skimage.feature import canny
                arr = np.array(image, dtype=np.float64)
                edges = canny(arr)
                return Image.fromarray((edges * 255).astype(np.uint8))
            except ImportError:
                # Fallback to skimage.filters (older versions)
                from skimage import filters
                arr = np.array(image, dtype=np.float64)
                edges = filters.canny(arr)
                return Image.fromarray((edges * 255).astype(np.uint8))
        except Exception:
            # Fallback on any error (import error, version incompatibility, etc.)
            print("Warning: Failed to use scipy.ndimage for Canny. Falling back to PIL's FIND_EDGES filter.")
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
            laplacian = (laplacian / laplacian.max() * 255).astype(np.uint8) if laplacian.max() > 0 else laplacian.astype(np.uint8)
            
            return Image.fromarray(laplacian)
        except Exception:
            # Fallback on any error (import error, version incompatibility, etc.)
            print("Warning: Failed to use scipy.ndimage for Laplacian. Falling back to PIL's FIND_EDGES filter.")
            return image.filter(ImageFilter.FIND_EDGES)
    
    @staticmethod
    def _prewitt_edge_detection(image: Image.Image) -> Image.Image:
        """Prewitt edge detection."""
        arr = np.array(image, dtype=np.float64)
        
        # Prewitt kernels
        prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
        
        try:
            from scipy import ndimage
            gradient_x = ndimage.convolve(arr, prewitt_x)
            gradient_y = ndimage.convolve(arr, prewitt_y)
        except Exception:
            # Fallback implementation
            gradient_x = EdgeProcessor._convolve2d(arr, prewitt_x)
            gradient_y = EdgeProcessor._convolve2d(arr, prewitt_y)
        
        # Calculate magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to 0-255
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8) if gradient_magnitude.max() > 0 else gradient_magnitude.astype(np.uint8)
        
        return Image.fromarray(gradient_magnitude)
    
    @staticmethod
    def _scharr_edge_detection(image: Image.Image) -> Image.Image:
        """Scharr edge detection."""
        arr = np.array(image, dtype=np.float64)
        
        # Scharr kernels
        scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
        scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        
        try:
            from scipy import ndimage
            gradient_x = ndimage.convolve(arr, scharr_x)
            gradient_y = ndimage.convolve(arr, scharr_y)
        except Exception:
            # Fallback implementation
            gradient_x = EdgeProcessor._convolve2d(arr, scharr_x)
            gradient_y = EdgeProcessor._convolve2d(arr, scharr_y)
        
        # Calculate magnitude
        gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        
        # Normalize to 0-255
        gradient_magnitude = (gradient_magnitude / gradient_magnitude.max() * 255).astype(np.uint8) if gradient_magnitude.max() > 0 else gradient_magnitude.astype(np.uint8)
        
        return Image.fromarray(gradient_magnitude)
    
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
            magnitude: Edge magnitude (0-1)
            direction: Edge direction in radians
            threshold: Threshold value (0-1)
            charset: Character set to use. Character sets are structured as:
                - Simple (4 chars): [none, weak, medium, strong] - only intensity mapping
                - Basic direction (5 chars): [none, horizontal, vertical, / diagonal, \ diagonal] - direction only
                - Extended (8+ chars): Various combinations of direction and intensity
                
        Returns:
            Character representing the edge
        """
        if magnitude < threshold:
            return charset[0]  # None character at index 0
        
        # Determine charset type based on length and common patterns
        charset_length = len(charset)
        
        # 1. Simple charset (4 or fewer characters): only intensity mapping
        if charset_length <= 4:
            # Map magnitude to character based on intensity
            idx = min(charset_length - 1, int(magnitude * charset_length))
            return charset[max(0, idx)]
        
        # Normalize direction to 0-180 degrees for consistent mapping
        deg = (np.degrees(direction) + 180) % 180
        
        # 2. Basic direction charset (5 characters): direction only
        if charset_length == 5:
            # Map direction to character: 0(no), 1(horizontal), 2(vertical), 3(/), 4(\)
            if deg < 22.5 or deg >= 157.5:
                return charset[1]  # horizontal
            elif 22.5 <= deg < 67.5:
                return charset[3]  # diagonal /
            elif 67.5 <= deg < 112.5:
                return charset[2]  # vertical
            else:
                return charset[4]  # diagonal \
        
        # 3. Extended charset (8+ characters): combine direction and intensity
        # Handle various extended charset types
        if charset_length >= 8:
            # Determine direction index first
            if deg < 22.5 or deg >= 157.5:
                dir_idx = 1  # horizontal
            elif 22.5 <= deg < 67.5:
                dir_idx = 3  # diagonal /
            elif 67.5 <= deg < 112.5:
                dir_idx = 2  # vertical
            else:
                dir_idx = 4  # diagonal \
            
            # Check for specific charset patterns
            
            # 8-character direction detail (0-none, 1-horiz, 2-vert, 3-/, 4-\, 5-thick horiz, 6-thick vert, 7-cross)
            if charset_length == 8 and charset[5] in ['═', '=', '━'] and charset[6] in ['┃', '|', '┇']:
                # Double the thickness based on magnitude
                if magnitude > 0.5:
                    if dir_idx == 1:  # horizontal
                        return charset[5]  # thick horizontal
                    elif dir_idx == 2:  # vertical
                        return charset[6]  # thick vertical
                    elif dir_idx in [3, 4]:  # diagonal
                        return charset[dir_idx]
                    else:
                        return charset[7]  # cross
                return charset[dir_idx]
            
            # 8-character strength-based (0-none, 1-weak, 2-vert, 3-/, 4-\, 5-medium, 6-strong, 7-very strong)
            elif charset_length == 8 and charset[5] in ['▒', '░', '·'] and charset[6] in ['▓', '▒', '●']:
                # Map direction first, then intensity for non-directional chars
                if dir_idx in [1, 2, 3, 4]:
                    # For directional chars, use intensity to determine thickness
                    if magnitude > 0.75:
                        return charset[7]  # very strong
                    elif magnitude > 0.5:
                        return charset[6]  # strong
                    elif magnitude > 0.25:
                        return charset[5]  # medium
                    return charset[dir_idx]
                else:
                    # For non-directional chars, use intensity mapping
                    idx = min(7, int(magnitude * 8))
                    return charset[idx]
            
            # Default extended charset: use direction mapping for first 5 chars, intensity for rest
            else:
                if magnitude > 0.75 and len(charset) > 5:
                    # Use stronger intensity characters from extended set
                    if dir_idx == 1:
                        return charset[5] if len(charset) > 5 else charset[dir_idx]  # horizontal 2
                    elif dir_idx == 2:
                        return charset[6] if len(charset) > 6 else charset[dir_idx]  # vertical 2
                    elif dir_idx in [3, 4]:
                        return charset[dir_idx]  # diagonal remains same
                    else:
                        return charset[7] if len(charset) > 7 else charset[dir_idx]  # cross
                return charset[dir_idx]
        
        # Fallback: simple magnitude mapping for any other charset
        idx = min(len(charset) - 1, int(magnitude * len(charset)))
        return charset[max(0, idx)]
    

    @staticmethod
    def _gradient_calculation(arr: np.ndarray, kernel_x: np.ndarray, kernel_y: np.ndarray, detector_name: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate gradient magnitude and direction using the specified kernels.
        
        Args:
            arr: Input image array
            kernel_x: Horizontal edge detection kernel
            kernel_y: Vertical edge detection kernel
            detector_name: Name of the detector (for warning messages)
            
        Returns:
            Tuple of (magnitude array, direction array)
        """
        try:
            from scipy import ndimage
            # Apply kernels
            gradient_x = ndimage.convolve(arr, kernel_x)
            gradient_y = ndimage.convolve(arr, kernel_y)
        except Exception:
            # Fallback implementation on any error
            print(f"Warning: Failed to use scipy.ndimage for {detector_name}. Falling back to manual implementation.")
            gradient_x = EdgeProcessor._convolve2d(arr, kernel_x)
            gradient_y = EdgeProcessor._convolve2d(arr, kernel_y)
        
        # Calculate magnitude and direction
        magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
        direction = np.arctan2(gradient_y, gradient_x)
        
        # Normalize magnitude to 0-255
        magnitude = (magnitude / magnitude.max() * 255).astype(np.uint8) if magnitude.max() > 0 else magnitude.astype(np.uint8)
        
        return magnitude, direction
        
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
            # Sobel edge detection
            sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
            sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
            return EdgeProcessor._gradient_calculation(arr, sobel_x, sobel_y, "Sobel")
        elif detector == EdgeDetector.CANNY:
            # For Canny, return magnitude only (direction not calculated)
            try:
                from scipy import ndimage
                try:
                    # Try importing from skimage.feature (newer versions)
                    from skimage.feature import canny
                    edges = canny(arr)
                except ImportError:
                    # Fallback to skimage.filters (older versions)
                    from skimage import filters
                    edges = filters.canny(arr)
                
                magnitude = (edges * 255).astype(np.uint8)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
            except Exception as e:
                print(e)
                # Fallback to FIND_EDGES on any error (import, version incompatibility, etc.)
                print("Warning: Failed to use scipy.ndimage for Canny. Falling back to PIL's FIND_EDGES filter.")
                edges = EdgeProcessor._find_edges(gray)
                magnitude = np.array(edges)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
        elif detector == EdgeDetector.PREWITT:
            # Prewitt edge detection
            prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
            prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
            return EdgeProcessor._gradient_calculation(arr, prewitt_x, prewitt_y, "Prewitt")
        elif detector == EdgeDetector.SCHARR:
            # Scharr edge detection
            scharr_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
            scharr_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
            return EdgeProcessor._gradient_calculation(arr, scharr_x, scharr_y, "Scharr")
        elif detector == EdgeDetector.LAPLACIAN:
            # Laplacian edge detection
            try:
                from scipy import ndimage
                laplacian = ndimage.laplace(arr)
                laplacian = np.absolute(laplacian)
                magnitude = (laplacian / laplacian.max() * 255).astype(np.uint8) if laplacian.max() > 0 else laplacian.astype(np.uint8)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
            except Exception:
                # Fallback implementation on any error
                print("Warning: Failed to use scipy.ndimage for Laplacian. Falling back to manual implementation.")
                # Manual Laplacian implementation
                laplacian_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
                laplacian = EdgeProcessor._convolve2d(arr, laplacian_kernel)
                laplacian = np.absolute(laplacian)
                magnitude = (laplacian / laplacian.max() * 255).astype(np.uint8) if laplacian.max() > 0 else laplacian.astype(np.uint8)
                direction = np.zeros_like(magnitude, dtype=np.float64)
                
                return magnitude, direction
        else:
            # For other detectors, use FIND_EDGES as fallback
            edges = EdgeProcessor.detect_edges(gray, detector)
            magnitude = np.array(edges)
            direction = np.zeros_like(magnitude, dtype=np.float64)
            
            return magnitude, direction
