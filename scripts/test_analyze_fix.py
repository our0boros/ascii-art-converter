#!/usr/bin/env python3
"""
Test script to verify the analyze functionality is working correctly.
"""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

try:
    from ascii_art_converter.cli import main
    import argparse
    print("✓ Successfully imported cli module")
except Exception as e:
    print(f"✗ Failed to import cli module: {e}")
    sys.exit(1)

try:
    from ascii_art_converter.analyzer import AsciiAnalyzer
    print("✓ Successfully imported analyzer module")
except Exception as e:
    print(f"✗ Failed to import analyzer module: {e}")
    sys.exit(1)

try:
    from ascii_art_converter.generator import AsciiArtGenerator
    from ascii_art_converter.character_sets import CharacterSet
    from ascii_art_converter.config import AsciiArtResult
    
    # Test AsciiAnalyzer with a simple AsciiArtResult
    result = AsciiArtResult(
        text="Hello World!\nTesting 123",
        lines=2,
        colors=None,
        width=12,
        height=2,
        original_size=(100, 100),
        complexity_score=1.0
    )
    
    # Test basic analysis
    analysis = AsciiAnalyzer.analyze(result)
    print(f"✓ Basic analysis works: {analysis.keys()}")
    print(f"  Lines: {analysis['line_count']}, Characters: {analysis['char_count']}")
    
    # Test with actual image conversion
    print("\nTesting with actual image conversion...")
    
    # Check if test image exists
    test_image_path = os.path.join(os.path.dirname(__file__), "data", "test.png")
    if not os.path.exists(test_image_path):
        print(f"⚠ Test image not found at {test_image_path}")
        print("Skipping image conversion test")
    else:
        print(f"✓ Test image found at {test_image_path}")
        
        # Test the CLI analyze functionality by running it
        import subprocess
        
        print("\nTesting CLI with analyze option...")
        cmd = [sys.executable, "-m", "ascii_art_converter.cli", test_image_path, "--analyze", "-w", "40"]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=os.path.dirname(__file__))
        
        if result.returncode == 0:
            print("✓ CLI analyze command executed successfully")
            # Print the analysis part of the output
            lines = result.stdout.splitlines()
            analysis_started = False
            for line in lines:
                if "Analysis:" in line:
                    analysis_started = True
                    print("\nAnalysis output:")
                elif analysis_started:
                    if line.strip():
                        print(line)
                    else:
                        # Stop at first empty line after analysis
                        break
        else:
            print(f"✗ CLI analyze command failed with exit code {result.returncode}")
            print(f"  Stderr: {result.stderr}")
    
    print("\n🎉 All tests completed!")
    
except Exception as e:
    print(f"✗ Test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
