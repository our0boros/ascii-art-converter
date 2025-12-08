"""ASCII Art Converter library.

This library provides tools to convert images to ASCII/Braille art with various options and modes.
"""

from .constants import (
    RenderMode,
    EdgeDetector,
    DitherMethod
)
from .config import AsciiArtConfig, AsciiArtResult
from .constants import BRAILLE_BASE, BRAILLE_DOTS

from .character_sets import CharacterSet

from .complexity import ComplexityAnalyzer
from .edge_detection import EdgeProcessor
from .braille import BrailleGenerator

from .generator import AsciiArtGenerator
from .formatters import AnsiColorFormatter, HtmlFormatter
from .batch import BatchProcessor
from .gif import GifToAscii
from .analyzer import AsciiAnalyzer
from .presets import Presets
from .utils import (
    image_to_ascii,
    print_ascii,
    save_html,
    demo,
    resize_to_fit,
    create_comparison,
    get_optimal_settings
)

# Use lazy import for interactive mode to avoid import warnings
import importlib

def __getattr__(name):
    if name == "InteractiveMode":
        from .interactive import InteractiveMode
        return InteractiveMode
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

# Define all public API
__all__ = [
    # Core types
    'AsciiArtGenerator',
    'AsciiArtConfig',
    'AsciiArtResult',
    
    # Enumerations
    'RenderMode',
    'EdgeDetector',
    'DitherMethod',
    'CharacterSet',
    
    # Constants
    'BRAILLE_BASE',
    'BRAILLE_DOTS',
    
    # Processors
    'EdgeProcessor',
    'BrailleGenerator',
    'ComplexityAnalyzer',
    
    # Formatters
    'AnsiColorFormatter',
    'HtmlFormatter',
    
    # Utilities
    'image_to_ascii',
    'print_ascii',
    'save_html',
    'demo',
    'resize_to_fit',
    'create_comparison',
    'get_optimal_settings',
    
    # Specialized classes
    'BatchProcessor',
    'GifToAscii',
    'AsciiAnalyzer',
    'InteractiveMode',
    'Presets',
]

# Version information
__version__ = '1.2.0'
__author__ = 'ASCII Art Converter Team'
__license__ = 'MIT'
