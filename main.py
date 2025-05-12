from model_utils import load_model, preprocess_image, encode_texts
import torch
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def main():
    if len(sys.argv) < 3:
        print("Usage: python main.py <image_path> <prompt1> <prompt2> ...")
        return

    image_path = sys.argv[1]
    text_prompts = sys.argv[2:]

    model, preprocess, device = load_model()
    image_input = preprocess_image(image_path, preprocess).to(device)
    text_inputs = encode_texts(model, device, text_prompts)

    with torch.no_grad():
        image_features = model.encode_image(image_input)
        text_features = model.encode_text(text_inputs)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        probs = similarities[0].cpu().numpy()

    for prompt, prob in zip(text_prompts, probs):
        print(f"{prompt}: {prob:.4f}")

    # Visualize embeddings
    visualize_embeddings(image_features.cpu(), text_features.cpu(), text_prompts)

def visualize_embeddings(image_feat, text_feats, labels):
    all_feats = torch.cat([image_feat, text_feats], dim=0).numpy()
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(all_feats)

    plt.figure(figsize=(8,6))
    plt.scatter(reduced[1:, 0], reduced[1:, 1], label="Text", c="blue")
    plt.scatter(reduced[0, 0], reduced[0, 1], label="Image", c="red", marker='x', s=100)
    for i, label in enumerate(labels):
        plt.text(reduced[i+1, 0], reduced[i+1, 1], label, fontsize=9)
    plt.legend()
    plt.title("PCA of CLIP Image and Text Embeddings")
    plt.savefig("embedding_visualization.png")
    plt.show()

if __name__ == "__main__":
    main()
