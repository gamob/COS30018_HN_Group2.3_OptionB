from pathlib import Path
from src.models.model import load_digit_cnn_model, predict_digit
from src.preprocessing.preprocessing import preprocess_image_steps
m = load_digit_cnn_model("src/models/digit_cnn_model.h5")
p = Path("data/unseen_data")
for f in sorted(p.rglob("*.png")):
    steps = preprocess_image_steps(str(f), size=(28,28), method="otsu", normalize=True)
    print(f.name, predict_digit(m, steps["final"]))