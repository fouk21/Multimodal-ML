# 🧠 Multimodal CLIP Inference Project

This is a small deep learning project that uses OpenAI's CLIP model to match images with textual descriptions — a classic example of **multimodal learning** (image + text).

---

## 📦 How it works

Given an image and a list of text prompts, the model returns probabilities for which prompt best matches the image using **zero-shot classification**.

It also visualizes the embeddings using **PCA**.

---

## ▶️ Run it

```bash
# Install dependencies
pip install -r requirements.txt

# Run example (with your own image)
python main.py path/to/image.jpg "a photo of a dog" "a photo of a cat"
```

---

## 📊 Example output

```
a photo of a dog: 0.9231
a photo of a cat: 0.0769
```

### 🔍 Embedding Visualization:

The script generates a plot like this:

![Embeddings](embedding_visualization.png)

---

## 📁 Files

- `main.py`: Runs inference and visualization
- `model_utils.py`: Loads CLIP and handles preprocessing
- `requirements.txt`: Dependencies
- `embedding_visualization.png`: (auto-generated)

---

## 🚀 Future Ideas

- Add Streamlit UI
- Compare multiple images
- Use t-SNE for nonlinear embedding projection
