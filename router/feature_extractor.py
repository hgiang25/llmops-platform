class FeatureExtractor:
    def __init__(self, model_name="bert-base-uncased"):
        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name)
        pass

    def extract(self, prompt: str):
        # Mock feature extraction (e.g., generating BERT embeddings)
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model(**inputs)
        # return outputs.last_hidden_state.mean(dim=1).detach().numpy()
        return [0.1, 0.2, 0.3]  # Mock embedding
