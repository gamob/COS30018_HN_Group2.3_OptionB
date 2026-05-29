### Problem 1:

Imports use relative paths (preprocessing.preprocessing) instead of package paths (src.preprocessing.preprocessing)
When running python -m src, Python cannot resolve these imports
Result: ModuleNotFoundError: No module named 'preprocessing'
Root Cause:
The script is written for direct execution but used as a package entry point.

## Fix:

 CORRECT approach - use absolute imports from src package
from src.preprocessing.preprocessing import process_path, preprocess_image_steps
from src.models.model import load_digit_cnn_model, predict_digit
from src.models.training.train_cnn import train_and_save_model
Or set PYTHONPATH:

bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python -m src preprocess ...
ISSUE #2: Model Path Bug in src/models/test/test_model.py [BLOCKING]

File: src/models/test/test_model.py (line 8)


model_path = root / "models" / "digit_cnn_model.h5"  
Problem:

root already resolves to project root (2 parents up from test file)
Adding /models/ again creates: project_root/models/digit_cnn_model.h5
Correct path should be: project_root/src/models/digit_cnn_model.h5
Result: FileNotFoundError: Model file not found
## Correct Fix:


model_path = root / "src" / "models" / "digit_cnn_model.h5"
ISSUE #3: Missing Sample Data Initialization [BLOCKING]

Location: data/sample/ directory

### Problem 2:

Repository README indicates sample datasets should exist in data/sample/
Data structure: part1/, part2/, part3/ with images and manifests
Current state: data/sample/ directory exists but is EMPTY
Scripts reference create_sample_datasets.py but it doesn't exist in repo
Result:

Cannot test preprocessing on part1 data
Cannot test segmentation on part2 data
Cannot test model training on real sample images
Required Action:

Either commit sample dataset files to data/sample/
Or create scripts/create_sample_datasets.py to generate them on first run
ISSUE #4: Hard-coded keras Imports (Deprecated) [HIGH]

Files: src/models/training/train_cnn.py, src/models/model.py

## Fix:
from keras.datasets import mnist 
from keras.models import Sequential  

### Problem 3:

Keras was integrated into TensorFlow since v2.4
Standalone Keras package is deprecated
Modern code should use: from tensorflow import keras
Result: Incompatibility with newer TensorFlow versions (may fail on import)

## Fix:

Python
from tensorflow import keras
from keras.datasets import mnist

### Problem 4: GUI Import Path Errors [MEDIUM]

File: src/gui/app.py (lines 14-15)


from preprocessing.preprocessing import preprocess_image_steps 
from models.model import load_digit_cnn_model  
Problem:

Same import path issue as in src/__main__.py
When running Streamlit, relative imports will fail
Result: ModuleNotFoundError when launching GUI
## Fix:

Python
from src.preprocessing.preprocessing import preprocess_image_steps
from src.models.model import load_digit_cnn_model