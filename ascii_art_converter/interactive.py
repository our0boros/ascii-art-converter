#!/usr/bin/env python3
"""
Image to ASCII/Braille Art Converter - Interactive Mode
======================================================
This module contains the InteractiveMode class for real-time ASCII art parameter adjustment.
"""

from PIL import Image
import os
from typing import Optional
from ascii_art_converter.config import AsciiArtConfig, AsciiArtResult
from ascii_art_converter.generator import AsciiArtGenerator
from ascii_art_converter.constants import RenderMode
from ascii_art_converter.character_sets import CharacterSet
from ascii_art_converter.formatters import AnsiColorFormatter


class InteractiveMode:
    """Interactive ASCII art preview with parameter adjustment."""
    
    def __init__(self, image: Image.Image):
        """Initialize with an image."""
        self.image = image
        self.config = AsciiArtConfig()
        self.generator = AsciiArtGenerator()
        self.result = None
    
    def update_config(self, **kwargs):
        """Update configuration parameters."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
    
    def render(self) -> str:
        """Render current configuration."""
        self.result = self.generator.convert(self.image, self.config)
        return self.result.text
    
    def render_colored(self, color_mode: str = '24bit') -> str:
        """Render with colors."""
        self.config.colorize = True
        self.result = self.generator.convert(self.image, self.config)
        
        if self.result.colors:
            return AnsiColorFormatter.format_result(self.result, color_mode=color_mode)
        return self.result.text
    
    def run(self):
        """Run interactive mode."""
        
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


def main():
    """Main entry point for interactive mode."""
    import sys
    from PIL import Image
    
    if len(sys.argv) < 2:
        print("Usage: python -m ascii_art_converter.interactive <image_path>")
        sys.exit(1)
    
    image_path = sys.argv[1]
    
    try:
        image = Image.open(image_path)
        interactive = InteractiveMode(image)
        interactive.run()
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
