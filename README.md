# LLMOps Platform

This is a comprehensive LLMOps platform focused on managing the lifecycle of Large Language Models (LLMs) with Difficulty-Aware Routing, Disaggregated Inference, and an end-to-end MLOps pipeline.

## Features
- **Difficulty-Aware Routing**: Routes prompts to either weak (local) or strong (external) models based on complexity.
- **Disaggregated Inference**: Separates prefill and decode stages for optimal GPU utilization.
- **LMCache**: KV Cache sharing between instances to speed up inference.
- **MLOps Lifecycle**: End-to-end pipeline for data collection, drift detection, evaluation, retraining (QLoRA), and model registry (MLflow).

## Prerequisites
- Python 3.12+
- Docker & Docker Compose (For metrics, redis, and MLflow)
- `venv` (Virtual Environment)

## Getting Started

1. Clone the repository and navigate to the project directory.
2. Create and activate a virtual environment:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
   ```
3. Install dependencies:
   ```powershell
   pip install -r requirements.txt
   ```
4. Start infrastructure services (Redis, Prometheus, Grafana, MLflow):
   ```powershell
   docker-compose up -d
   ```

## Running the Platform

You need to open **2 separate terminals** and activate the virtual environment in both before running the services.

**Terminal 1 (Run API Gateway):**
```powershell
.\venv\Scripts\activate
uvicorn api_gateway.main:app --reload --port 8000
```

**Terminal 2 (Run Streamlit Frontend):**
```powershell
.\venv\Scripts\activate
streamlit run frontend/app.py
```

---

## Hướng dẫn trải nghiệm và Test MLOps Pipeline (Tutorial)

Để hiểu rõ cách hệ thống MLOps vận hành khép kín, bạn hãy thực hiện theo kịch bản (scenario) sau trên giao diện Streamlit (`http://localhost:8501`):

### Bước 1: Trải nghiệm Inference (Giao tiếp với mô hình)
1. Mở trang **"💬 Inference"** trên thanh điều hướng bên trái.
2. Nhập một vài câu hỏi (prompt) từ ngắn đến dài. Ví dụ:
   - *Câu ngắn:* "Hello, how are you?"
   - *Câu dài (CloudOps):* "How to troubleshoot a Kubernetes pod stuck in CrashLoopBackOff state due to OOMKilled?"
3. Nhấn **"🚀 Send (Auto Route)"**.
4. **Điều gì xảy ra ngầm?** 
   - Hệ thống (API Gateway) sẽ tự động chấm điểm độ khó (Difficulty Score) của câu hỏi.
   - Nếu dễ (Score < 0.4), nó đẩy vào mô hình nhỏ (Weak Model - 7B).
   - Nếu khó (Score > 0.7), nó đẩy ra API ngoài (Strong Model - GPT-4o).
   - Quan trọng nhất: Mọi prompt, kết quả, thời gian phản hồi, mô hình được chọn đều được **DataCollector ngầm lưu lại** vào thư mục `data/raw/` dưới dạng file JSONL.

### Bước 2: Xem dữ liệu đã thu thập
1. Chuyển sang trang **"📋 Data Logs"** hoặc **"📊 MLOps Dashboard"**.
2. Nhấn **"Load Logs"**. Bạn sẽ thấy danh sách các câu hỏi bạn vừa chat ở Bước 1.
3. *Ý nghĩa:* Trong thực tế, dữ liệu này chính là nguồn để ta giám sát xem hành vi người dùng có thay đổi không, và dùng để huấn luyện lại mô hình trong tương lai.

### Bước 3: Phát hiện Data Drift (Sự sai lệch dữ liệu)
1. Chuyển sang trang **"🔍 Drift Detection"**.
2. Nhấn **"Run Drift Detection"**.
3. **Điều gì xảy ra ngầm?**
   - Hệ thống sử dụng thư viện **Evidently AI** để so sánh tập dữ liệu hiện tại (Current Data) với tập dữ liệu gốc ban đầu (Reference Data).
   - Nếu người dùng đột ngột hỏi những chủ đề mới, hoặc câu hỏi dài hơn bất thường, phân phối thống kê sẽ thay đổi -> Hệ thống cảnh báo **DRIFT DETECTED ⚠️**.
   - Bạn có thể xem chi tiết cột nào bị lệch (ví dụ: `prompt_length`, `difficulty_score`).

### Bước 4: Chạy Pipeline Huấn luyện lại (Retraining)
Khi hệ thống bị Drift, ta cần huấn luyện lại Router Model hoặc LLM để thích ứng.
1. Chuyển sang trang **"🔄 Retraining Pipeline"**.
2. Tích chọn **"Force retrain"** và **"Generate synthetic data"** (để giả lập có dữ liệu mới).
3. Nhấn **"▶️ Run Full Pipeline"**.
4. **Điều gì xảy ra ngầm (Đọc Log trên màn hình)?**
   - **EVAL (Đánh giá cũ):** Chạy bài test F1 Score và Cost Savings trên model hiện tại.
   - **TRAIN (Huấn luyện):** Chạy thuật toán QLoRA fine-tuning (hiện đang chạy giả lập Mock Mode vì chưa cắm GPU). Nó sẽ in ra Loss giảm dần qua 3 Epochs.
   - **EVAL (Đánh giá mới):** Chạy lại bài test trên model vừa train xong.
   - **COMPARE (So sánh):** So sánh F1 Score. Nếu model mới xịn hơn (F1 cao hơn mức cho phép), nó khuyên **DEPLOY**. Nếu bằng hoặc kém hơn, nó khuyên **KEEP_CURRENT**.
   - **REGISTER (Đăng ký):** Nếu đạt chuẩn DEPLOY, mô hình mới sẽ được tự động đóng gói và đẩy lên kho **MLflow Registry** kèm theo các chỉ số (metrics) để theo dõi version.

### Bước 5: Xem kho mô hình trên MLflow
1. Mở trình duyệt vào trang quản lý MLflow: `http://localhost:5000` (yêu cầu Docker đang chạy `docker-compose up -d`).
2. Tại đây, bạn sẽ thấy lịch sử các lần chạy Pipeline, các biểu đồ loss, metrics (F1_score), và danh sách các Model Versions đã được đăng ký để sẵn sàng đưa ra môi trường Production.
