# Foggy Object Detection using Deep Neural Networks

This project focuses on **object detection in foggy weather conditions** using a combination of **AOD-Net for dehazing** and **YOLOv8** for detection.  
It was presented and published at the **2025 Global Conference in Emerging Technology (GINOTECH), IEEE Pune Section**.

---

## 🚀 Features
- Video dehazing using AOD-Net and CycleGAN
- Object detection using YOLOv8
- Flask-based web application with simple UI
- Upload and process videos

---

## 📂 Project Structure
├── app2.py # Flask application
├── dehaze_aodnet.py # AOD-Net model for dehazing
├── dehaze_cyclegan.py # CycleGAN-based dehazing
├── domain_adaptation.py # Domain adaptation code
├── model.py # Deep learning model functions
├── video.py # Video processing script
├── templates/ # HTML templates
├── static/ # CSS, outputs, uploads
└── requirements.txt # Required dependencies


---

## ⚡ Installation
```bash
git clone https://github.com/Reddy-py/foggy_object_detection.git
cd foggy_object_detection
pip install -r requirements.txt

python app2.py

Then open your browser at http://127.0.0.1:5000/

📜 Publication

This work is published in:
IEEE Pune Section, GINOTECH 2025 — "Object Detection in Foggy Conditions using Deep Neural Network"

Medapati Siva Sankar Rama Reddy
GitHub: Reddy-py


---

## 2️⃣ Add a **LICENSE**
- Right-click project root → **New → File** → name it `LICENSE`  
- Paste this (MIT License — widely used):  

```text
MIT License

Copyright (c) 2025 Medapati Siva Sankar Rama Reddy

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

git add README.md LICENSE
git commit -m "Add README and License"
git push origin main
