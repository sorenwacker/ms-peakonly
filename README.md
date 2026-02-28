# ms-peakonly

[![GitHub stars](https://img.shields.io/github/stars/sorenwacker/ms-peakonly)](https://github.com/sorenwacker/ms-peakonly/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Neural network-based peak detection for untargeted metabolomics LC-MS data.

This package provides a high-level Python API for the [peakonly](https://github.com/Arseha/peakonly) neural network, making it easy to integrate into data analysis pipelines and Jupyter notebooks.

## Features

- Automatic peak detection using pre-trained neural networks
- Batch processing of multiple mzML files
- Simple API for integration into existing workflows
- Automatic model download on first use
- Returns results as pandas DataFrames

## Installation

```bash
pip install git+https://github.com/sorenwacker/ms-peakonly
```

### Dependencies

- PyTorch >= 1.2.0
- pandas
- numpy
- pymzML
- scipy
- matplotlib

## Quick Start

```python
from ms_peakonly import PeakOnly
from glob import glob

# Get list of mzML files
files = glob("data/*.mzML")

# Initialize (downloads models automatically on first use)
po = PeakOnly(model_dir="models/")

# Process files and get peak table
peaks = po.process(files)

# Results are returned as a pandas DataFrame
print(peaks.head())
```

## How It Works

The neural network approach identifies chromatographic peaks directly from raw LC-MS data without requiring traditional signal processing steps like smoothing or baseline correction. The model was trained on manually annotated peaks from diverse metabolomics datasets.

## Input Format

- **Supported format:** mzML files
- Files should contain centroided MS1 data

## Output

The `process()` method returns a pandas DataFrame containing detected peaks with columns for:

- Retention time
- m/z value
- Intensity
- Peak boundaries
- Quality scores

## Citation

If you use this package in your research, please cite the original peakonly paper:

> Kumler, W., et al. (2023). Neural network-based peak detection for untargeted metabolomics.

## Acknowledgments

This package wraps the neural network implementation from [Arseha/peakonly](https://github.com/Arseha/peakonly).

## License

MIT License - see [LICENSE](LICENSE) for details.
