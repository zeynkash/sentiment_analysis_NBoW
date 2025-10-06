# 🧠 Sentiment Analysis using NBoW (Neural Bag-of-Words)

This project implements a **Neural Bag-of-Words (NBoW)** model using **PyTorch** for sentiment analysis on text data.  
It demonstrates how to train a simple yet effective neural network that averages word embeddings to classify text as **positive** or **negative**.

---

## 📚 Overview

The NBoW model is a lightweight text classifier that:
- Converts each word into an embedding vector.
- Averages all embeddings in a sentence (bag-of-words approach).
- Passes the averaged vector through a linear layer to predict sentiment.

Although simple, this project provides a clear foundation for understanding how text classification works with embeddings before moving to more advanced architectures like RNNs or Transformers.

---

## ⚙️ Features

- Custom `Dataset` and `DataLoader` for text preprocessing  
- `get_collate_fn()` for dynamic padding within batches  
- `train()` and `evaluate()` loops for model optimization and validation  
- Pretrained **Word2Vec (Google News)** embeddings integration  
- Model saving and loading using `state_dict`  
- Sentiment prediction for custom user input  

---

## 🧩 Model Architecture

Input Text → Token IDs → Embedding Layer → Mean Pooling → Linear Layer → Output Sentiment


- **Embedding layer:** Converts tokens into vectors  
- **Mean pooling:** Averages embeddings across the sequence  
- **Fully connected (Linear) layer:** Maps pooled vector to sentiment logits  

---

## 🧠 Example Inference

```python
new_text = "I absolutely didn't love this movie!"
sentiment, confidence = predict_sentiment(new_text, model, tokenizer, vocab, device)

print(f"Text: '{new_text}'")
print(f"Predicted: {sentiment}")
print(f"Confidence: {confidence:.3f}")

```

Output:
```python
Text: 'I absolutely didn't love this movie!'
Predicted: Positive
Confidence: 0.994
```
---

🧰 Requirements

Python 3.9+

PyTorch

tqdm

gensim

numpy

Install dependencies:
```python
pip install torch tqdm gensim numpy
```
---

## 🚀 Training

1. Prepare your dataset and vocabulary.

2. Train the model:
train_loss, train_acc = train(train_data_loader, model, criterion, optimizer, device)
valid_loss, valid_acc = evaluate(valid_data_loader, model, criterion, device)

3. The best model is saved automatically as nbow.pt.


----

## 📈 Evaluation Metrics

During training, the script tracks and stores:

Training & validation loss

Training & validation accuracy

Results are saved in the metrics dictionary for easy visualization later.

---

## 💾 Model Saving & Loading

```python
# Save
torch.save(model.state_dict(), "nbow.pt")

# Load
model.load_state_dict(torch.load("nbow.pt"))
```


---

## 🔍 Future Improvements

Use BiLSTM or Transformer-based models

Apply attention mechanisms for better context capture

Experiment with max pooling instead of mean pooling

Add negation-aware pre-processing

---

## 🧑‍💻 Author

Zeyn Kash
AI & Software Engineering Student at Istanbul University
