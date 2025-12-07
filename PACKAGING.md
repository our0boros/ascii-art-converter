# Packaging and Installation Guide

This guide explains how to package the ASCII Art Converter project as a Python package and install it using pip.

## Prerequisites

Make sure you have the following tools installed:

- Python 3.7 or higher
- pip (Python package installer)
- setuptools (for packaging)
- wheel (for building wheel distributions)

You can install the packaging tools with:

```bash
pip install --upgrade setuptools wheel twine
```

## Building the Package

Follow these steps to build the package:

1. **Navigate to the project directory**:
   ```bash
   cd path/to/AsciiArtConverter
   ```

2. **Build the package**:
   ```bash
   python setup.py sdist bdist_wheel
   ```

   This command will create two types of distributions:
   - A source distribution (`.tar.gz` file) in the `dist/` directory
   - A wheel distribution (`.whl` file) in the `dist/` directory

## Installing the Package

### Option 1: Install from Local Wheel

The easiest way to install the package locally is using the wheel file:

```bash
pip install dist/ascii_art_converter-1.0.0-py3-none-any.whl
```

### Option 2: Install in Development Mode

For development purposes, you can install the package in editable mode:

```bash
pip install -e .
```

This allows you to make changes to the source code and immediately see the changes without needing to reinstall the package.

### Option 3: Install from Source Distribution

You can also install from the source distribution:

```bash
pip install dist/ascii_art_converter-1.0.0.tar.gz
```

### Option 4: Install from Git Repository

If you have the project hosted in a Git repository, you can install it directly:

```bash
pip install git+https://github.com/yourusername/ascii-art-converter.git
```

## Verifying the Installation

After installation, you can verify that the package is installed correctly:

1. **Check if the package is installed**:
   ```bash
   pip list | grep ascii-art-converter
   ```

2. **Test the command-line tools**:
   ```bash
   ascii-art-converter --help
   ascii-interactive --help
   ```

3. **Test the Python import**:
   ```bash
   python -c "from ascii_art_converter import AsciiArtGenerator; print('Import successful')"
   ```

## Uninstalling the Package

To uninstall the package:

```bash
pip uninstall ascii-art-converter
```

## Updating the Package

To update the package after making changes:

1. **Increment the version number** in `setup.py`
2. **Rebuild the package**:
   ```bash
   python setup.py sdist bdist_wheel
   ```
3. **Install the updated version**:
   ```bash
   pip install --upgrade dist/ascii_art_converter-X.Y.Z-py3-none-any.whl
   ```

## Distributing the Package

### Upload to PyPI (Optional)

If you want to make your package available on PyPI (Python Package Index), you can use twine:

1. **Create an account** on [PyPI](https://pypi.org/) and [TestPyPI](https://test.pypi.org/)

2. **Upload to TestPyPI first** (for testing):
   ```bash
   twine upload --repository testpypi dist/*
   ```

3. **Install from TestPyPI** to verify:
   ```bash
   pip install --index-url https://test.pypi.org/simple/ ascii-art-converter
   ```

4. **Upload to PyPI** (once tested):
   ```bash
   twine upload dist/*
   ```

## Additional Notes

- **Versioning**: Follow [Semantic Versioning](https://semver.org/) for version numbers
- **Dependencies**: Keep dependencies minimal and well-specified
- **Documentation**: Update the README.md and docstrings as you make changes
- **Testing**: Add tests to ensure the package works as expected

## Troubleshooting

### Common Issues

1. **Missing dependencies**:
   - Ensure all required packages are listed in `install_requires` in `setup.py`

2. **Command not found** after installation:
   - Make sure the Python scripts directory is in your PATH
   - On Windows: `%APPDATA%\Python\Python3X\Scripts`
   - On Unix/Linux: `~/.local/bin`

3. **Permission errors**:
   - Use `--user` flag with pip to install for current user only:
     ```bash
     pip install --user ascii-art-converter
     ```
   - Or use sudo (not recommended) on Unix/Linux:
     ```bash
     sudo pip install ascii-art-converter
     ```

4. **Import errors**:
   - Check that the package is properly installed
   - Verify the import statement matches the package structure

For more information on Python packaging, refer to the [official Python Packaging Guide](https://packaging.python.org/).
