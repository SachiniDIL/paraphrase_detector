from sentence_transformers import SentenceTransformer, util

# Load pre-trained model (this happens once)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Ask user for input
sentence1 = input("Enter the first sentence: ")
sentence2 = input("Enter the second sentence: ")

# Convert the sentences to vector form
embeddings = model.encode([sentence1, sentence2], convert_to_tensor=True)

# Measure how similar the two vectors are
similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])

# Show result
score = similarity.item()
print(f"\nSimilarity Score: {score:.2f}")

# Explain result
if score > 0.8:
    print("🟡 These sentences are likely paraphrased.")
elif score > 0.5:
    print("🟠 These sentences might be somewhat similar.")
else:
    print("🟢 These sentences are likely different.")
