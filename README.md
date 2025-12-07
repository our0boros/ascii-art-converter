# ASCII Art Converter

```bash
   _____    __________________ .___.___     _____          __    
  /  _  \  /   _____/\_   ___ \|   |   |   /  _  \________/  |_  
 /  /_\  \ \_____  \ /    \  \/|   |   |  /  /_\  \_  __ \   __\ 
/    |    \/        \\     \___|   |   | /    |    \  | \/|  |   
\____|__  /_______  / \______  /___|___| \____|__  /__|   |__|   
        \/        \/         \/                  \/              
_________                                   __                   
\_   ___ \  ____   _______  __ ____________/  |_  ___________    
/    \  \/ /  _ \ /    \  \/ // __ \_  __ \   __\/ __ \_  __ \   
\     \___(  <_> )   |  \   /\  ___/|  | \/|  | \  ___/|  | \/   
 \______  /\____/|___|  /\_/  \___  >__|   |__|  \___  >__|      
        \/            \/          \/                 \/         
```

A powerful and flexible Python library and command-line tool for converting images to ASCII art, Braille art, and edge detection art with support for color, dithering, and various customization options.

```
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠛⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠏⠈⠀⡿⢻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢿⠛⠀⠀⢰⠁⠸⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⣿⠀⠀⠀⠀⠀⢀⡞⠈⢿⣿⣧⠉⣿⣿⣿⡿⠛⣻⣿⣿⣿⡟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡇⢿⠀⠀⠀⠀⢀⡍⠀⠀⠀⢿⡟⢾⣿⣿⣿⢇⣾⣿⣿⠿⣿⣿⣿⣿⣿⣿⡟⠻⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⡛⠀⠀⠀⠀⠛⠀⠀⠀⠀⠘⠀⢸⣿⣿⢃⣸⣿⣿⣿⣦⣌⢿⣿⣿⢻⣿⣷⢀⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⠿⠿⣿⣿⣿⡿⠀⠈⠙⣿⠂⠀⠀⠀⠀⠀⣾⣿⠇⡼⢹⣿⣿⣿⣿⣿⡟⣿⣿⡀⠻⠇⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⡿⠟⠋⠉⠀⠀⠀⠀⠀⠻⣛⡉⢀⣤⢆⡀⡀⢾⣿⣷⣦⠄⣸⣿⣿⠀⠃⠀⠹⢿⣿⣿⠋⠀⣿⣿⡆⢀⣶⡌⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣛⣁⣀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢸⣿⣿⢏⡴⠋⠀⠀⠉⠋⣰⣿⣿⡟⣰⣶⣦⣄⠀⢠⣴⣦⣄⣿⣿⣇⠘⠛⠉⠙⠻⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⠀⠀⠀⠀⠨⠛⠁⠊⠂⣴⣊⡤⣴⢂⣿⣿⣿⡇⠙⠿⠟⠛⡁⠈⠉⠉⢹⣿⣿⡇⠰⠁⠀⠀⠀⠀⠙⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠈⣿⣿⡇⢷⠘⣿⣿⠿⠁⠒⠿⢿⡿⠿⠟⠂⠀⢾⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠙⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣶⣾⣿⣿⣿⣿⣦⣀⣤⠖⠒⠀⠈⠛⠁⠀⠓⢦⠅⣀⣐⠚⠁⠀⠀⠲⠀⠀⠀⠢⡈⠡⠒⠀⠀⢀⣠⣤⣶⣶⣦⣤⣌⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠿⣋⣤⡴⠒⠀⣀⣤⣤⡄⢀⣴⣾⣿⣿⣿⠁⠀⣴⣾⣷⡀⢀⠂⢠⣤⣤⣤⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⢋⣴⣷⣿⠟⣿⠛⢀⠞⣵⣿⣿⣿⠃⡀⢀⢿⣿⣿⣿⡾⠟⣷⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣿⣿⡏⣠⣤⡉⠦⢀⣼⣿⣿⣿⠃⠈⡀⡘⠈⣿⣿⣿⣿⣠⣽⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠏⠘⣿⡿⠀⣸⣼⣿⣿⣿⠃⢀⣬⠻⢻⣦⡘⣿⣿⣿⡇⠻⡹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⠋⢀⡼⠋⣼⣿⣿⣿⣿⣿⣿⠇⣾⣿⣿⣿⣶⣷⢻⣿⣿⣿⢸⢀⠋⣹⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡟⠛⢿⣿⣿⣿⣿⡧⠈⠳⢁⣿⡟⣿⣿⣿⣯⣿⣿⣿⣿⣿⣿⣿⣿⠀⢿⣿⣿⠈⠛⠁⢿⠟⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡧⠈⢿⣟⠛⡛⠈⣡⣼⣦⠌⣿⢁⣿⣿⡏⣸⣿⣿⣿⣿⣿⣿⣿⠃⠀⠘⢛⣻⡇⠳⣄⠀⢠⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡔⠃⠀⠈⠉⣥⣶⠛⠿⠋⠀⠁⠾⡿⠟⠃⠁⠀⠈⠙⠛⠛⠋⠀⠀⠀⠀⠾⡿⠋⠀⠈⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣄⢰⣶⡄⠻⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⡀⠀⠀⠀⠀⣀⠀⡀⠀⠀⠀⠀⠀⠀⢠⣶⣾⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣭⣅⣠⣶⣦⣶⣤⠀⠤⠄⠀⠀⠀⢠⣤⣷⣶⣾⣶⣾⣿⡆⠠⠀⠀⠀⠐⠀⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣟⣛⣛⡀⠀⠀⠀⣀⣀⣈⣛⣛⣉⣯⣭⣉⣉⣁⣀⣀⣀⣀⣀⣀⣩⣭⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿
```

## Features

<img src="./README.assets/example.gif" alt="example" width="100%"/>

- **Multiple Rendering Modes**:
  - Density-based ASCII art
  - Braille pattern art
  - Edge detection art

- **Color Support**:
  - 24-bit true color
  - 256-color palette
  - 16-color palette
  - Various color sampling methods (center, average, dominant)

- **Customization Options**:
  - Adjustable character aspect ratio
  - Multiple character sets (standard, detailed, blocks, simple, etc.)
  - Custom character sets
  - Brightness, contrast, gamma, and sharpness adjustments
  - Dithering support (Floyd-Steinberg, ordered, Atkinson)

- **Interactive Mode**:
  - Real-time preview and adjustment
  - Live parameter tweaking
  - Convenient saving options

- **Batch Processing**:
  - Process multiple images at once
  - Consistent configuration across images

- **Output Formats**:
  - Plain text
  - HTML with styling
  - ANSI color codes for terminals

## Installation

### Prerequisites
- Python 3.7+
- Pillow (PIL fork) for image processing
- NumPy for numerical operations

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/our0boros/ascii-art-converter.git
cd ascii-art-converter
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Conversion
```bash
python -m ascii_art_converter.cli image.png
```

### Set Output Width
```bash
python -m ascii_art_converter.cli image.png -w 80
```

### Use Braille Mode
```bash
python -m ascii_art_converter.cli image.png -m braille
```

### Edge Detection
```bash
python -m ascii_art_converter.cli image.png -m edge -t 0.2
```

### Colored HTML Output
```bash
python -m ascii_art_converter.cli image.png -c -o output.html
```

### 256-Color Terminal Output
```bash
python -m ascii_art_converter.cli image.png --colorize --color-mode 256
```

## Command-Line Interface

### Usage
```
python -m ascii_art_converter.cli [-h] [-o OUTPUT] [-w WIDTH] [-H HEIGHT] [--max-width MAX_WIDTH]
                                  [--min-width MIN_WIDTH] [--char-ratio CHAR_RATIO] [-m {density,braille,edge}]
                                  [--charset CHARSET] [--custom-charset CUSTOM_CHARSET] [-i] [-t THRESHOLD]
                                  [--edge-detector {sobel,prewitt,laplacian,canny,scharr}] [--edge-sigma EDGE_SIGMA]
                                  [--edge-charset EDGE_CHARSET] [--braille-threshold BRAILLE_THRESHOLD]
                                  [--dither {none,floyd_steinberg,ordered,atkinson}] [-c] [--color-mode {24bit,256,16}]
                                  [--color-sample {center,average,dominant}] [--contrast CONTRAST]
                                  [--brightness BRIGHTNESS] [--gamma GAMMA] [--sharpness SHARPNESS]
                                  [--complexity-factor COMPLEXITY_FACTOR] [--demo] [-a] [-v]
                                  [input]
```

### Positional Arguments
```
input                 Input image file
```

### Optional Arguments
```
-h, --help            show this help message and exit
-o OUTPUT, --output OUTPUT
                      Output file (txt, html, or ansi)
-w WIDTH, --width WIDTH
                      Output width in characters
-H HEIGHT, --height HEIGHT
                      Output height in characters
--max-width MAX_WIDTH
                      Maximum auto width
--min-width MIN_WIDTH
                      Minimum auto width
--char-ratio CHAR_RATIO
                      Character aspect ratio (width/height)
-m {density,braille,edge}, --mode {density,braille,edge}
                      Rendering mode
--charset CHARSET     Character set: standard, detailed, blocks, simple, binary, dots, geometric
--custom-charset CUSTOM_CHARSET
                      Custom character string (dark to light)
-i, --invert          Invert brightness
-t THRESHOLD, --threshold THRESHOLD
                      Edge detection threshold (0-1)
--edge-detector {sobel,prewitt,laplacian,canny,scharr}
                      Edge detection algorithm
--edge-sigma EDGE_SIGMA
                      Gaussian blur sigma for edge detection
--edge-charset EDGE_CHARSET
                      Character set for edge mode
--braille-threshold BRAILLE_THRESHOLD
                      Threshold for braille dots (0-1)
--dither {none,floyd_steinberg,ordered,atkinson}
                      Dithering method for braille
-c, --colorize        Enable color output
--color-mode {24bit,256,16}
                      Terminal color mode
--color-sample {center,average,dominant}
                      Color sampling method
--contrast CONTRAST   Contrast adjustment (0.5-2.0)
--brightness BRIGHTNESS
                      Brightness adjustment (0.5-2.0)
--gamma GAMMA         Gamma correction
--sharpness SHARPNESS
                      Sharpness enhancement
--complexity-factor COMPLEXITY_FACTOR
                      Multiplier for auto-size complexity calculation
--demo                Run demo
-a, --analyze         Show analysis of the generated ASCII art
-v, --verbose         Verbose output
```

## Interactive Mode

The converter includes an interactive mode for real-time preview and adjustment of parameters:

```bash
python -m ascii_art_converter.interactive image.png
```

### Interactive Commands
```
Commands:
  w <num>     - Set width
  m <mode>    - Set mode (density/braille/edge)
  c <charset> - Set charset
  i           - Toggle invert
  color       - Toggle color
  t <num>     - Set threshold (edge mode)
  contrast <num> - Set contrast
  render      - Render current settings
  save <file> - Save to file
  q           - Quit
```

## Configuration Options

### Character Sets

The converter provides several built-in character sets:

- `standard`: Standard density-based character set
- `detailed`: More detailed character set for higher resolution
- `blocks`: Block characters for bold, contrasting art
- `simple`: Simplified character set for minimalistic art
- `binary`: Only two characters (0 and 1) for binary art
- `dots`: Dot-based characters
- `geometric`: Geometric shapes

### Edge Detectors

Available edge detection algorithms:

- Sobel (default)
- Prewitt
- Laplacian
- Canny
- Scharr

### Dithering Methods

Dithering options for Braille mode:

- None (default)
- Floyd-Steinberg
- Ordered
- Atkinson

## Advanced Usage

### Custom Character Set

Create custom ASCII art using your own character set ordered from darkest to lightest:

```bash
python -m ascii_art_converter.cli image.png --custom-charset " .:-=+*#%@"
```

### Complex Image Processing

Combine multiple options for advanced effects:

```bash
python -m ascii_art_converter.cli image.png \
    -w 120 \
    --char-ratio 0.45 \
    --mode braille \
    --dither floyd_steinberg \
    --colorize \
    --color-mode 256 \
    --contrast 1.2 \
    --brightness 1.1 \
    --sharpness 1.5
```

### Batch Processing

Use the batch processing functionality to convert multiple images with the same configuration:

```python
from ascii_art_converter.batch import BatchProcessor
from ascii_art_converter.generator import AsciiArtConfig
from ascii_art_converter.constants import RenderMode

# Create a batch processor
processor = BatchProcessor()

# Define configuration
config = AsciiArtConfig(
    width=80,
    mode=RenderMode.DENSITY,
    colorize=True,
    color_depth=8  # 8 for 256-color, 24 for true color
)

# Process all PNG files in a directory
processor.process_directory(
    input_dir="images/",
    output_dir="output/",
    config=config,
    extensions=[".png", ".jpg"]
)
```

## API Usage

The library can be used programmatically in your Python projects:

```python
from PIL import Image
from ascii_art_converter.generator import AsciiArtGenerator, AsciiArtConfig
from ascii_art_converter.constants import RenderMode

# Load an image
image = Image.open("data/test.png")

# Create configuration
config = AsciiArtConfig(
    width=100,
    mode=RenderMode.DENSITY,
    char_aspect_ratio=0.45,
    colorize=True,
    color_depth=24,  # 8 for 256-color, 24 for true color
    contrast=1.2
)

# Generate ASCII art
generator = AsciiArtGenerator()
result = generator.convert(image, config)

# Print the result
print(result.text)

# Save as HTML
from ascii_art_converter.formatters import HtmlFormatter
html_content = HtmlFormatter.format(result, image, True)
with open("output.html", "w", encoding="utf-8") as f:
    f.write(html_content)

# Save as ANSI
from ascii_art_converter.formatters import AnsiColorFormatter
ansi_content = AnsiColorFormatter.format_result(result, color_mode="256")
# with open("output.ansi", "w", encoding="utf-8") as f:
#     f.write(ansi_content)
print(ansi_content)
```

## Examples

### Basic ASCII Art
```bash
python -m ascii_art_converter.cli image.png -w 80
```

### Colored Braille Art
```bash
python -m ascii_art_converter.cli image.png -w 100 -m braille -c --color-mode 256
```

### Edge Detection Art
```bash
python -m ascii_art_converter.cli image.png -w 90 -m edge --edge-detector canny -t 0.15
```

### HTML Output with Custom Styling
```bash
python -m ascii_art_converter.cli image.png -w 120 --colorize -o output.html
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup

1. Fork the repository
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install development dependencies:
   ```bash
   pip install -e .[dev]
   ```
4. Run tests:
   ```bash
   pytest
   ```

### Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures and variables
- Write docstrings for all public functions and classes
- Add tests for new functionality

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Inspired by various ASCII art converters and image processing techniques
- Built on top of Pillow and NumPy for robust image manipulation
- Thanks to all contributors who have helped improve this project

## Support

If you encounter any issues or have questions, please open an issue on the [GitHub repository](https://github.com/our0boros/ascii-art-converter/issues).