# 🖋️ Handwritten Number Recognition System (HNRS)
**Course:** COS30018 - Intelligent Systems (Swinburne University of Technology)  
**Assignment:** Project Assignment - Option B [cite: 2, 3]  
**Due Date:** 11:59 pm 02/11/2025 (End of Week 12) [cite: 4]  

---

## 🌟 Project Overview
This project implements a system capable of recognizing handwritten numbers through a multi-stage pipeline[cite: 8, 11]. It goes beyond simple digit recognition by handling multi-digit construction and (as an extension) simple arithmetic expressions[cite: 11, 35].

### 🎯 Key Objectives
* **Image Acquisition:** Loading individual images or auto-creating images from folders[cite: 20].
* **Preprocessing:** Standardizing input via grayscaling and resizing[cite: 21].
* **Segmentation:** Partitioning multi-digit numbers into individual digits[cite: 25].
* **Classification:** Using Machine Learning (CNNs and other models) to recognize digits[cite: 26, 28].
* **GUI:** Providing a user-friendly interface to control hyper-parameters and visualize results[cite: 17, 18].

---

## 📂 Repository Structure
We follow a modular structure to allow parallel development as per the project plan[cite: 65].

```text
/
├── .gitignore               # Standard Python ignores
├── README.md                # Project Overview & Integration Rules
├── requirements.txt         # Required libraries (numpy, pandas, torch/tensorflow, etc.) [cite: 14]
├── src/                     # Source Code
│   ├── preprocessing/       # Task 1: Image Preprocessing [cite: 21]
│   ├── segmentation/        # Task 2: Image Segmentation [cite: 25]
│   ├── models/              # Task 3: ML Model Training & Logic [cite: 26]
│   └── gui/                 # GUI Implementation [cite: 17]
├── data/                    # Dataset storage (MNIST) [cite: 15]
├── documents/               # Project reports, assignment brief, and review notes
└── documents/week*-reports/ # Weekly progress documentation and report drafts

---

## 🚀 Run the Project
Use the package entry points to run preprocessing, training, prediction, and the GUI in a consistent workflow.

- Preprocess images:
  ```bash
  python -m src preprocess data/input_folder output_folder --method otsu --size 28 28
  ```
  
  Preprocessing Task 2: tested on 5+ images from data/sample/part1/image.
  Outputs are 28×28 PNGs (visually clean and centered); pipeline does not crash on blank or very-dark images.
- Train the CNN model on MNIST:
  ```bash
  python -m src train --epochs 5
  ```
- Predict a single image using the saved model:
  ```bash
  python -m src predict data/sample.png
  ```
- Launch the GUI:
  ```bash
  streamlit run src/gui/app.py
  ```

## 🔗 Entry-point modules
The project now supports module-based execution using Python's `-m` switch:
- `python -m src` — unified workflow runner
- `python -m src.preprocessing` — preprocessing CLI
- `python -m src.models.training` — model training
- `python -m src.models.test` — model evaluation/test harness
- `python -m src.gui` — prints the GUI launch command
