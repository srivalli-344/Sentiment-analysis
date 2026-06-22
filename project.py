# Install necessary libraries

from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

# Load sentiment analysis model (RoBERTa)
sentiment_model_name = "cardiffnlp/twitter-roberta-base-sentiment"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_model_name)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_model_name)
sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model, tokenizer=sentiment_tokenizer)

# Load sarcasm detection model
sarcasm_model_name = "mrm8488/t5-base-finetuned-sarcasm-twitter"
sarcasm_pipeline = pipeline("text-classification", model=sarcasm_model_name)

# Function to classify with sarcasm awareness
def classify_with_context(sentence):
    sarcasm_result = sarcasm_pipeline(sentence)[0]
    sentiment_result = sentiment_pipeline(sentence)[0]

    if sarcasm_result['label'].lower() == "sarcasm":
        print(f"[!] Sarcasm detected in: {sentence}")
        if sentiment_result['label'] == "LABEL_2":
            sentiment_result['label'] = "negative"
        elif sentiment_result['label'] == "LABEL_0":
            sentiment_result['label'] = "positive"

    return {
        "sentence": sentence,
        "sarcasm": sarcasm_result['label'],
        "sentiment": sentiment_result['label'],
        "confidence": round(sentiment_result['score'], 2)
    }

# Test it
test_sentences = [
    "I love how everything is going wrong today!",
    "Oh great, another Monday!",
    "I'm thrilled to be stuck in traffic for 2 hours.",
    "I had a genuinely great time at the event!"
]

for sentence in test_sentences:
    result = classify_with_context(sentence)
    print(result)
