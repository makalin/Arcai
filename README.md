# Arcai
AI-Powered Archaeological Site Detection

![Arcai Logo](logo.png)

**Arcai** (AI + Archaeology) is an AI-powered virtual archaeologist designed to analyze satellite imagery and identify potential archaeological sites. The system leverages modern machine learning, computer vision, and geospatial technologies to uncover hidden traces of ancient civilizations.

---

## 🚀 Features
- **Automatic Detection**: AI-powered identification of archaeological sites from satellite imagery
- **Pre-trained Models**: ArcaiNet v1 and v2 models for rapid analysis
- **Interactive Web Interface**: User-friendly web application for image upload and analysis
- **Multiple Output Formats**: Visualizations, interactive maps, and GIS-compatible exports
- **Comprehensive Documentation**: Detailed guides for data preparation, training, and deployment

---

## 🛰️ Tech Stack
- **Python 3.8+** - Core programming language
- **TensorFlow/Keras** - Deep learning framework
- **OpenCV** - Computer vision and image processing
- **GeoPandas** - Geospatial data handling
- **Flask** - Web application framework
- **Folium** - Interactive web mapping
- **Matplotlib/Seaborn** - Data visualization

---

## 📂 Project Structure
```
Arcai/
├── 📄 README.md                    # Main project documentation
├── 📄 LICENSE                      # MIT License
├── 📄 requirements.txt             # Python dependencies
├── 🐍 setup.py                     # Installation script
├── 🌐 web_interface.py             # Flask web application
├── 📊 data/                        # Dataset directory
│   ├── 📄 README.md               # [Data documentation](data/README.md)
│   ├── 📁 positive/               # Images with archaeological sites
│   ├── 📁 negative/               # Images without sites
│   └── 📁 sample_images/          # Example images for testing
├── 🤖 models/                      # Model files
│   └── 📄 README.md               # [Model documentation](models/README.md)
├── 📈 outputs/                     # Detection results
│   └── 📄 README.md               # [Output documentation](outputs/README.md)
├── 🔬 src/                         # Source code
│   ├── 📄 __init__.py             # Package initialization
│   ├── 🎯 detect.py               # Main detection script
│   ├── 🧠 model.py                # ArcaiNet model architecture
│   ├── 🔧 preprocessing.py        # Image preprocessing utilities
│   ├── 🚀 train.py                # Model training script
│   └── 📊 visualization.py        # Visualization utilities
├── 📓 notebooks/                   # Jupyter notebooks
│   ├── 📄 01_data_exploration.ipynb
│   └── 📄 02_model_training.ipynb
├── 🌐 templates/                   # Web interface templates
│   ├── 📄 index.html              # Main upload page
│   └── 📄 results.html            # Results display page
└── 🎨 static/                      # Static web assets
    ├── 📁 css/
    └── 📁 js/
```

---

## ⚡ Quick Start

### 1️⃣ Automated Setup
```bash
# Clone the repository
git clone https://github.com/makalin/Arcai.git
cd Arcai

# Run automated setup
python setup.py
```

### 2️⃣ Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p data/{positive,negative,sample_images} models outputs uploads
```

### 3️⃣ Add Training Data
- Place satellite images with archaeological sites in `data/positive/`
- Place images without sites in `data/negative/`
- See [Data Documentation](data/README.md) for format requirements

### 4️⃣ Train the Model
```bash
python src/train.py --data-dir data/ --epochs 50 --model-type v1
```

### 5️⃣ Run Detection
```bash
# Command line detection
python src/detect.py --input data/sample_images/sample_satellite.tif

# Web interface
python web_interface.py
# Open http://localhost:5000
```

---

## 📊 Example Outputs

The system generates multiple output formats:

- **Detection Visualizations**: PNG images with bounding boxes
- **Interactive Maps**: HTML files with Folium maps
- **Confidence Plots**: Bar charts of detection scores
- **HTML Reports**: Comprehensive analysis reports
- **GeoJSON Exports**: GIS-compatible data format

See [Output Documentation](outputs/README.md) for detailed descriptions.

---

## 🌍 Dataset

Arcai includes a curated set of satellite images for testing and training. The dataset supports:

- **Formats**: GeoTIFF, JPEG, PNG
- **Resolution**: 1-5m per pixel (optimal)
- **Channels**: RGB and multi-spectral
- **Organization**: Positive/negative sample directories

See [Data Documentation](data/README.md) for detailed format requirements and organization guidelines.

---

## 🤖 Models

### ArcaiNet v1
- **Architecture**: Custom CNN with 4 convolutional blocks
- **Performance**: ~85-90% accuracy
- **Use Case**: General archaeological site detection

### ArcaiNet v2
- **Architecture**: Enhanced CNN with attention mechanisms
- **Performance**: ~88-93% accuracy
- **Use Case**: High-accuracy detection with attention to important features

See [Model Documentation](models/README.md) for detailed architecture and performance information.

---

## 🌐 Web Interface

Arcai includes a modern web interface with:

- **Drag-and-drop upload**: Easy satellite image upload
- **Real-time processing**: Live detection with progress indicators
- **Interactive results**: Clickable maps and downloadable outputs
- **Responsive design**: Works on desktop and mobile devices

Start the web interface:
```bash
python web_interface.py
```

---

## 📓 Jupyter Notebooks

Interactive tutorials and examples:

- **[Data Exploration](notebooks/01_data_exploration.ipynb)**: Analyze dataset characteristics
- **[Model Training](notebooks/02_model_training.ipynb)**: Step-by-step training guide

---

## 📌 Roadmap

- [x] Core detection system
- [x] Web interface
- [x] Multiple output formats
- [x] Comprehensive documentation
- [ ] Expand dataset with more regions
- [ ] Improve detection accuracy for desert and forest-covered areas
- [ ] Add real-time satellite feed processing
- [ ] Mobile application
- [ ] Cloud deployment options

---

## 📄 License

MIT License — see `LICENSE` file for details.

---

## 🙌 Contributions

Contributions, issues, and feature requests are welcome!

### How to Contribute
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Development Setup
```bash
# Clone your fork
git clone https://github.com/yourusername/Arcai.git
cd Arcai

# Install in development mode
pip install -e .

# Run tests
python -m pytest tests/
```

---

## 🌟 Acknowledgments

Arcai draws inspiration from the work of:
- Digital archaeologists and remote sensing specialists
- Open geospatial communities
- Archaeological survey teams worldwide
- Open source machine learning communities

---

## 📞 Support

- **Documentation**: Check the README files in each directory
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join community discussions for questions and ideas

For detailed information about specific components, see:
- [Data Documentation](data/README.md)
- [Model Documentation](models/README.md)
- [Output Documentation](outputs/README.md)
