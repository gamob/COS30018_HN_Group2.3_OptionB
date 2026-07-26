#  Handwritten Number Recognition System (HNRS)
**Course:** COS30018 - Intelligent Systems [cite: 1]  
**Due Date:** 11:59 pm 02/11/2025 (End of Week 12) [cite: 4]  
**Team:** [Group of 3-4 Students] [cite: 6]

---

##  Universal Integration Protocol
To ensure that our individual "scraps" (Tasks 1, 2, 3, and GUI) match up perfectly during the final integration phase (Task 4), all team members **must** adhere to these interface standards. [cite: 66, 60]

###  The Data Flow Pipeline
We use **NumPy arrays** as our universal language. The data must flow as follows:

| Stage | Component | Input Format | Output Format | Responsible Member |
| :--- | :--- | :--- | :--- | :--- |
| **Start** | **GUI** | User Upload/Folder | `str` (File Path) | Member 4 |
| **Task 1** | **Preprocessing** | `str` (File Path) | `np.array` (Cleaned Image) | Member 1 |
| **Task 2** | **ML Recognition** | `np.array` (Cleaned Image) | `list[int]` (Predicted Numbers) | Member 3 |
| **End** | **GUI** | `list[int]` | UI Display / Visuals | Member 4 |

---

##  Detailed Module Requirements

### 1. Image Preprocessing (`src/preprocessing/`)
* **Goal:** Research/experiment with at least 2 techniques (e.g., grayscaling, binarization, resizing). [cite: 21, 59]
* **Mandatory Function:** `process_image(image_path: str) -> np.array`
* **Constraint:** Output must be standardized (e.g., $28 \times 28$ pixels) to match ML model input. [cite: 21]
### 2. ML Model Representation & Training (`src/models/`)
* **Goal:** Compare different techniques (e.g., CNNs vs. others) using the MNIST dataset. [cite: 28, 60]
* **Mandatory Function:** `predict_digits(digit_list: list[np.array]) -> list[int]`
* **Constraint:** Accuracy and performance must be evaluated on both single and multi-digit images. [cite: 31]

### 3. System GUI (`src/gui/`)
* **Goal:** Allow user input, show output, and enable parameter setting/visualization. [cite: 17, 18]
* **Requirement:** Must support loading from a file or automatic creation from a folder. [cite: 20]

---

##  Weekly Progress & GitHub Rules
* **Individual Contribution:** Every student must contribute code and/or documents AND commit to GitHub weekly. Failure to do so will result in a penalty (up to -80 marks!). [cite: 67]
* **Branching:** Work in your specific folder (`src/task_name/`). Use branches for major changes.
* **Testing:** Each module should include a `if __name__ == "__main__":` block to demonstrate it works independently before integration. [cite: 60]
* **Coding Standards:** Follow good programming practices with clear, helpful comments. [cite: 60]

---

##  Extensions (Aiming for D/HD)
We are aiming for **Extension Option 2**: Recognition of simple arithmetic expressions (digits $0-9$ and $+$, $-$, $*$, $/$, $(, )$) and calculating the result. [cite: 35, 67]
