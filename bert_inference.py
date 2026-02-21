import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

MODEL_NAME = "bhadresh-savani/distilbert-base-uncased-emotion"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(device)
model.eval()

labels = ["sadness", "joy", "love", "anger", "fear", "surprise"]

def predict_top_k(text, k=3):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=1)

    probs = probs[0]
    top_probs, top_indices = torch.topk(probs, k)

    results = {}
    for i in range(k):
        emotion = labels[top_indices[i]]
        confidence = float(top_probs[i])
        results[emotion] = confidence

    return results
