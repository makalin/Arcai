# Arcai

**Arcai** (AI + Archaeology) is an AI-powered virtual archaeologist designed to analyze satellite imagery and identify potential archaeological sites. The system leverages modern machine learning, computer vision, and geospatial technologies to uncover hidden traces of ancient civilizations.

---

## 🚀 Features
- Automatic detection of potential archaeological sites from satellite imagery.
- Pre-trained AI models for rapid analysis.
- Dataset of satellite images for research and testing.
- Interactive visualization of detected sites.
- Geospatial data export for use in GIS tools.

---

## 🛰️ Tech Stack
- **Python**
- **TensorFlow**
- **OpenCV**
- **GeoPandas**
- **Matplotlib / Folium** (for visualization)

---

## 📂 Project Structure
```

arcai/
├── data/               # Example satellite images and labels
├── models/             # Pre-trained model files
├── notebooks/          # Jupyter notebooks for experiments
├── src/                # Source code for training/inference
├── outputs/            # Visualizations and detection results
└── README.md           # Project documentation

````

---

## ⚡ Quick Start
1️⃣ Clone the repo:
```bash
git clone https://github.com/makalin/Arcai.git
cd Arcai
````

2️⃣ Install dependencies:

```bash
pip install -r requirements.txt
```

3️⃣ Run detection on sample data:

```bash
python src/detect.py --input data/sample_image.tif
```

---

## 📊 Example Output

<img src="outputs/example_detection.png" width="600"/>

---

## 🌍 Dataset

Arcai includes a curated set of satellite images (public domain / CC) for testing purposes. See [`data/`](data/) folder for details.

---

## 🤖 Models

* **ArcaiNet v1**: Custom TensorFlow model trained on labeled satellite data.
* Pre-trained weights available in the [`models/`](models/) folder.

---

## 📌 Roadmap

* [ ] Expand dataset with more regions.
* [ ] Improve detection accuracy for desert and forest-covered areas.
* [ ] Add web-based interactive map.

---

## 📄 License

MIT License — see `LICENSE` file for details.

---

## 🙌 Contributions

Contributions, issues, and feature requests are welcome!
Feel free to submit a pull request or open an issue.

---

## 🌟 Acknowledgments

Arcai draws inspiration from the work of digital archaeologists and open geospatial communities.
