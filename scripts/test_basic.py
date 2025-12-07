#!/usr/bin/env python3
"""
Basic pytest tests for the ascii_art_converter package.
"""

import pytest
from PIL import Image
from io import BytesIO
from ascii_art_converter.generator import AsciiArtGenerator, AsciiArtConfig
from ascii_art_converter import __version__


def test_package_version():
    """Test that the package has a valid version."""
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_generator_import():
    """Test that AsciiArtGenerator can be imported and instantiated."""
    generator = AsciiArtGenerator()
    assert isinstance(generator, AsciiArtGenerator)


def test_config_creation():
    """Test that AsciiArtConfig can be created with default values."""
    config = AsciiArtConfig(width=80)
    assert isinstance(config, AsciiArtConfig)
    assert config.width == 80


def test_simple_conversion():
    """Test that a simple image conversion works."""
    # Create a simple test image (100x100 red square)
    image = Image.new('RGB', (100, 100), color='red')
    
    # Create config and generator
    config = AsciiArtConfig(width=40)
    generator = AsciiArtGenerator()
    
    # Perform conversion
    result = generator.convert(image, config)
    
    # Verify result
    assert result is not None
    assert result.width == 40
    assert result.height > 0
    assert len(result.text) > 0


def test_different_render_modes():
    """Test that different render modes work."""
    from ascii_art_converter.constants import RenderMode
    
    # Create a simple test image
    image = Image.new('RGB', (80, 80), color='blue')
    
    # Test different render modes
    for mode in [RenderMode.DENSITY, RenderMode.EDGE, RenderMode.BRAILLE]:
        config = AsciiArtConfig(width=30, mode=mode)
        generator = AsciiArtGenerator()
        result = generator.convert(image, config)
        assert result is not None
        assert len(result.text) > 0


def test_colorize_option():
    """Test that the colorize option works."""
    # Create a simple test image
    image = Image.new('RGB', (50, 50), color='green')
    
    # Test with colorize enabled
    config = AsciiArtConfig(width=20, colorize=True)
    generator = AsciiArtGenerator()
    result = generator.convert(image, config)
    
    # Verify color data is present
    assert result is not None
    assert result.colors is not None
    assert len(result.colors) == result.height
    assert len(result.colors[0]) == result.width
