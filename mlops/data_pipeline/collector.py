import json

class DataCollector:
    def __init__(self, log_file="data/raw/prompts_log.jsonl"):
        self.log_file = log_file

    def log_request(self, prompt, route, difficulty_score):
        record = {
            "prompt": prompt,
            "route": route,
            "difficulty_score": difficulty_score
        }
        # with open(self.log_file, "a") as f:
        #     f.write(json.dumps(record) + "\n")
        print(f"Logged request: {record}")
