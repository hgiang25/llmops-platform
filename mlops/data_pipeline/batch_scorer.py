"""
Batch Scorer — Chấm điểm Ground Truth (LLM-as-a-Judge).

Script này đóng vai trò như một cronjob định kỳ.
Nó đọc các log từ DataCollector (chưa có ground truth),
sử dụng LLM-as-a-Judge (GPT-4) hoặc hàm mô phỏng để chấm điểm
và phân loại prompt thành 3 nhãn:
0: weak, 1: strong_disaggregated, 2: strong_external.

Dữ liệu đầu ra được lưu vào data/current/ để làm dữ liệu retrain/drift check.
"""

import json
import os
import random
from pathlib import Path

def mock_llm_as_a_judge(prompt: str) -> int:
    """
    Mô phỏng LLM-as-a-Judge chấm điểm độ khó của câu hỏi.
    Thực tế ở đây bạn có thể call OpenAI API.
    """
    length = len(prompt)
    if length < 50:
        return 0  # weak
    elif length < 150:
        # Nếu có từ khóa lập trình, ưu tiên model mạnh hơn
        if any(keyword in prompt.lower() for keyword in ["code", "kubernetes", "docker", "python", "aws"]):
            return 2  # strong_external
        return 1  # strong_disaggregated
    else:
        return 2  # strong_external

def run_batch_scoring(input_log_path: str = "data/raw/prompts_log.jsonl", 
                      output_path: str = "data/current/cloudops_current.jsonl") -> dict:
    print(f"Starting batch scoring job...")
    print(f"Input: {input_log_path}")
    print(f"Output: {output_path}")
    
    if not os.path.exists(input_log_path):
        print(f"File {input_log_path} không tồn tại. Tạo file rỗng để test.")
        # Tạo mock data nếu chưa có
        Path("data/raw").mkdir(parents=True, exist_ok=True)
        with open(input_log_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"prompt": "Hello", "route": "weak"}) + "\n")
            f.write(json.dumps({"prompt": "How to deploy Kubernetes cluster?", "route": "strong"}) + "\n")

    scored_records = []
    
    with open(input_log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
            except Exception:
                continue
            
            prompt = record.get("prompt", "")
            
            # Chấm điểm Ground Truth bằng LLM-as-a-Judge
            label = mock_llm_as_a_judge(prompt)
            
            route_map = {0: "weak", 1: "strong_disaggregated", 2: "strong_external"}
            
            record["ground_truth_label"] = label
            record["ground_truth_route"] = route_map[label]
            # Giữ lại difficulty_score cho tương thích với evaluator cũ nếu cần,
            # nhưng model classification sẽ học theo ground_truth_label
            record["difficulty_score"] = float(label) / 2.0 
            
            scored_records.append(record)

    # Lưu lại thành file current dataset
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for record in scored_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Đã chấm điểm xong {len(scored_records)} records.")
    return {
        "status": "completed",
        "scored_records": len(scored_records),
        "output_path": output_path
    }

if __name__ == "__main__":
    run_batch_scoring()
