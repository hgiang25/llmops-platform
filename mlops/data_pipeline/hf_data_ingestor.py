import json
import os
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Ensure root is in sys path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# pyrefly: ignore [missing-import]
from datasets import load_dataset
import random

CLOUDOPS_KEYWORDS = [
    "kubernetes", "docker", "linux", "terraform", "aws", "azure", 
    "gcp", "network", "server", "database", "cloud", "ubuntu", 
    "centos", "nginx", "apache", "sql", "nosql", "dns", "tcp",
    "ssl", "tls", "load balancer", "firewall", "vpc", "subnet",
    "oom", "crashloopbackoff", "pod", "deployment", "helm"
]

def simulate_strong_llm_judge(prompt: str) -> float:
    """
    Simulates a Strong LLM (like GPT-4) judging the difficulty of a prompt.
    In reality, this would be an API call to OpenAI/Anthropic.
    """
    prompt_lower = prompt.lower()
    
    # Base score based on length
    word_count = len(prompt_lower.split())
    base_score = min(0.3 + (word_count / 100.0) * 0.4, 0.7)
    
    # Add points for technical complexity (keywords)
    tech_score = sum(0.05 for kw in CLOUDOPS_KEYWORDS if kw in prompt_lower)
    
    final_score = min(base_score + tech_score, 1.0)
    
    # Add a bit of random noise (human/model variance)
    final_score = final_score + random.uniform(-0.05, 0.05)
    return round(max(0.0, min(1.0, final_score)), 4)

def determine_domain(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(kw in prompt_lower for kw in ["kubernetes", "docker", "pod", "helm"]):
        return "kubernetes"
    elif any(kw in prompt_lower for kw in ["network", "dns", "tcp", "ssl", "vpc"]):
        return "network"
    elif any(kw in prompt_lower for kw in ["linux", "ubuntu", "centos"]):
        return "linux"
    elif any(kw in prompt_lower for kw in ["terraform", "yaml", "config"]):
        return "configuration"
    elif any(kw in prompt_lower for kw in ["error", "fail", "crash", "oom"]):
        return "troubleshooting"
    return "general_cloud"

def ingest_huggingface_data(
    dataset_name: str = "tatsu-lab/alpaca", 
    split: str = "train",
    max_samples: int = 1000,
    output_file: str = "data/raw/cloudops_real_dataset.jsonl"
):
    print(f"Starting Data Ingestion Pipeline from HuggingFace ({dataset_name})...")
    print("Mode: STREAMING (Optimized for Big Data, No RAM Overflow)")
    
    # 1. Load dataset with streaming=True
    # Streaming allows us to download and process one row at a time, avoiding OOM on large datasets
    dataset = load_dataset(dataset_name, split=split, streaming=True)
    
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    collected_count = 0
    start_time = time.time()
    
    with open(path, "w", encoding="utf-8") as f:
        for idx, row in enumerate(dataset):
            # Extract prompt depending on dataset structure
            
            prompt = ""
            response = ""
            
            # Alpaca format
            if "instruction" in row and "output" in row:
                prompt = row.get("instruction", "")
                response = row.get("output", "")
            # UltraChat format
            elif "data" in row and isinstance(row["data"], list) and len(row["data"]) >= 2:
                prompt = row["data"][0]
                response = row["data"][1]
            
            if not prompt or not response:
                continue
                
            prompt_lower = prompt.lower()
            
            # 2. Filter: Only keep CloudOps related data
            if not any(kw in prompt_lower for kw in CLOUDOPS_KEYWORDS):
                continue
                
            # 3. Labeling: Simulate LLM-as-a-Judge for difficulty score
            difficulty_score = simulate_strong_llm_judge(prompt)
            domain = determine_domain(prompt)
            
            # Route assignment (for the dataset label)
            if difficulty_score < 0.4:
                route = "weak"
                model_used = "cloudops-llm-7b-finetuned"
            elif difficulty_score < 0.7:
                route = "strong_disaggregated"
                model_used = "cloudops-llm-70b-disaggregated"
            else:
                route = "strong_external"
                model_used = "gpt-4o"
                
            # Create standardized record
            record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "prompt": prompt,
                "route": route,
                "difficulty_score": difficulty_score,
                "model_used": model_used,
                "response_time_ms": round(random.uniform(50, 1500), 2),
                "token_count": len(prompt.split()) + len(response.split()),
                "response_text": response[:100] + "...", # Truncated for space
                "prompt_length": len(prompt),
                "prompt_word_count": len(prompt.split()),
                "domain": domain,
                "source": "huggingface_ingestion"
            }
            
            # 4. Write chunk to disk immediately (JSONL format)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush() # Ensure it's written to disk
            
            collected_count += 1
            
            if collected_count % 100 == 0:
                print(f"Processed and saved {collected_count} CloudOps records...")
                
            if collected_count >= max_samples:
                break
                
    elapsed = time.time() - start_time
    print(f"\nIngestion Complete!")
    print(f"Successfully filtered and saved {collected_count} records.")
    print(f"Time taken: {elapsed:.2f} seconds.")
    print(f"Output file: {path.absolute()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HuggingFace Data Ingestor")
    parser.add_argument("--dataset", type=str, default="tatsu-lab/alpaca", help="HF Dataset name")
    parser.add_argument("--max_samples", type=int, default=500, help="Maximum samples to collect")
    
    args = parser.parse_args()
    ingest_huggingface_data(dataset_name=args.dataset, max_samples=args.max_samples)
