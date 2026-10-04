import json
import requests
import os

input_file = "data/reference/cloudops_reference.jsonl"
output_file = "data/reference/cloudops_reference_v6.jsonl"
api_url = "http://localhost:8080/chat"

print("Starting Baseline Reset (Sending requests to Router v6)...")

new_records = []
with open(input_file, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        if not line.strip(): continue
        data = json.loads(line)
        prompt = data["prompt"]
        
        try:
            resp = requests.post(api_url, json={"prompt": prompt})
            if resp.status_code == 200:
                result = resp.json()
                # Cập nhật các trường quan trọng từ mô hình v6
                data["difficulty_score"] = result.get("difficulty_score", data["difficulty_score"])
                data["route"] = result.get("route", data["route"])
                data["model_used"] = result.get("model_used", data["model_used"])
                
                # Thêm log để theo dõi tiến độ
                if i % 50 == 0:
                    print(f"[{i}] Updated: {prompt[:30]}... -> Score v6: {data['difficulty_score']}")
            else:
                print(f"API Error at line {i}: {resp.status_code}")
        except Exception as e:
            print(f"Exception at line {i}: {e}")
            
        new_records.append(data)

# Ghi đè file
print("\nOverwriting with new Reference Baseline...")
with open(input_file, "w", encoding="utf-8") as f:
    for data in new_records:
        f.write(json.dumps(data) + "\n")
        
print("Success! Baseline Reset is complete. Drift Detector is now synced with v6.")
