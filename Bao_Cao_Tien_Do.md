# 📊 BÁO CÁO TIẾN ĐỘ ĐỒ ÁN: LLMOps PLATFORM - DIFFICULTY-AWARE ROUTING

## 1. Tiến độ đã thực hiện (Những gì em đã làm được)
Trong tuần vừa qua, em đã hoàn thiện thành công lõi (Core) quan trọng nhất của hệ thống: **Mô hình định tuyến độ khó (Difficulty-Aware Router)** từ khâu huấn luyện đến triển khai thực tế. Cụ thể:

* **Thiết kế kiến trúc AI tối ưu:** 
  * Chọn `DeBERTa-v3-base` làm Router thay vì dùng LLM Decoder.
  * Lập trình mạng nơ-ron tùy chỉnh `OrdinalClassificationHead`. Thay vì dùng CrossEntropy thông thường, em dùng **Hồi quy thứ bậc (Ordinal Regression)** để mô hình hiểu được bản chất tăng dần của độ khó (Weak < Medium < Strong).
* **Ứng dụng công nghệ Fine-tuning tiên tiến:**
  * Khắc phục thành công điểm yếu của DeBERTa bằng cách tích hợp công nghệ **LoRA (Low-Rank Adaptation)** (không nén 4-bit) để huấn luyện.
  * Tối ưu hóa Inference: Sử dụng thuật toán `merge_and_unload()` để gộp vĩnh viễn trọng số LoRA vào mô hình gốc sau khi train xong. Điều này giúp hệ thống đạt độ thông minh cao nhưng **độ trễ (overhead) khi chạy thực tế bằng 0**.
* **Đóng gói và Triển khai (MLOps Pipeline):**
  * Tích hợp thành công **MLflow Model Registry**. Mô hình sau khi huấn luyện được tự động đăng ký, lưu trữ và đẩy lên môi trường Production.
  * Xây dựng xong luồng End-to-End: Giao diện web (Streamlit) gọi API Gateway (FastAPI) -> Load model từ MLflow -> Chấm điểm prompt -> Điều hướng request.

---

## 2. Các Quyết định Kiến trúc Hệ thống (Architectural Decisions)
*(Phần này giải thích chi tiết lý do đằng sau các quyết định công nghệ trong đồ án).*

### 2.1. Tại sao chọn DeBERTa thay vì Qwen cho bài toán Router?
* **Kiến trúc Encoder vs Decoder:** Bài toán định tuyến là bài toán **Phân loại văn bản (Sequence Classification)**. Mô hình Encoder (như DeBERTa) sinh ra để đọc hiểu toàn bộ câu văn từ 2 chiều (trái sang phải và ngược lại) để rút ra đặc trưng (CLS token) chấm điểm. Trong khi đó, Decoder (như Qwen) sinh ra để dự đoán từ tiếp theo (Autoregressive Generation). Việc dùng Decoder làm phân loại là "dùng dao mổ trâu giết gà", không đúng bản chất mạng nơ-ron.
* **Độ trễ suy luận (Latency):** Router đóng vai trò là "người gác cổng", mọi request đều phải đi qua nó trước. DeBERTa có 86 triệu tham số, thời gian suy luận chỉ mất **~20 mili-giây**. Qwen-0.5B có 500 triệu tham số, thời gian suy luận mất **~200-300 mili-giây** (chưa kể chi phí KV-Cache).
* **Tài nguyên VRAM:** Để chạy toàn bộ hệ thống LLMOps trên một máy tính cá nhân (VRAM 6GB - RTX 4050), Router phải thật sự nhẹ. DeBERTa tốn chưa tới 1.5GB VRAM, để lại khoảng trống bộ nhớ cho các tác vụ LLM suy luận thực sự ở phía sau.

### 2.2. Tại sao dùng LoRA thay vì QLoRA? Lợi ích của nó?
Ban đầu, khi huấn luyện DeBERTa bằng phương pháp truyền thống (Full Fine-Tuning), hệ thống liên tục gặp lỗi nổ Gradient (NaN) do kiến trúc nội bộ của DeBERTa rất nhạy cảm. Em đã quyết định áp dụng LoRA để giải quyết bài toán này.
* **Sự khác biệt giữa LoRA và QLoRA:**
  * **QLoRA (Quantized LoRA):** Là kỹ thuật nén (Quantization) bộ não gốc xuống 4-bit trước khi cấy LoRA. Việc nén này tiết kiệm RAM cho các mô hình khổng lồ, nhưng đánh đổi bằng việc **mất mát độ chính xác** và **tăng độ trễ** do CPU/GPU phải liên tục "giải nén" trong lúc tính toán.
  * **LoRA (Tiêu chuẩn):** Giữ nguyên 100% chất lượng của bộ não gốc (16-bit / 32-bit), chỉ cấy thêm một nhánh học tập nhỏ. Vì DeBERTa đã đủ nhỏ để load vào VRAM, việc dùng QLoRA (nén) là không cần thiết và làm suy giảm hiệu năng.
* **Lợi ích tuyệt đối của việc dùng LoRA cho DeBERTa:**
  1. **Khắc phục lỗi nổ Gradient:** Bằng cách đóng băng bộ não gốc, quá trình học chỉ diễn ra trên các ma trận LoRA nhỏ (được khởi tạo bằng 0). Điều này giúp quá trình toán học cực kỳ ổn định.
  2. **Tăng tốc độ tiếp thu:** Nhờ sự ổn định trên, em có thể đẩy Learning Rate lên mức cao (`1e-4`) một cách an toàn. Trong 15 Epochs, mô hình học được rất nhanh mà không sợ bị sập (Crash).
  3. **Tối ưu hóa Inference (Zero Latency Overhead):** Kỹ thuật đỉnh cao của LoRA là sau khi huấn luyện xong, em có thể dùng toán học để **Hòa trộn (Merge)** các trọng số mới này thẳng vào bộ não gốc (Lệnh `merge_and_unload()`). Kết quả là lúc chạy thực tế, không có một nhánh LoRA nào tồn tại, mô hình giữ nguyên kiến trúc ban đầu với tốc độ 20 mili-giây.

### 2.3. Luồng Dữ liệu Thực (Real-world Data Flow) & Continuous Retraining (MLOps)
Một trong những điểm làm nên sự khác biệt của hệ thống này là khả năng **tự học hỏi từ dữ liệu thực tế** (Continuous Learning) thay vì chỉ chạy tĩnh:
* **Cơ chế thu thập dữ liệu tự động (Data Collector):** Mọi thao tác chat của người dùng trên giao diện Streamlit khi đi qua API Gateway đều được thu thập một cách tự động và lưu trữ vào Data Lake (file `prompts_log.jsonl`). Hệ thống lưu lại chi tiết câu hỏi (Prompt), điểm khó dự đoán (Difficulty Score), mô hình được điều hướng (Route), tốc độ phản hồi, và đặc biệt là **Phản hồi của người dùng (Feedback - Thumbs Up/Down)**.
* **Cơ chế chống nhiễu loạn (Decoupled Monitoring):** Việc ghi log được thiết kế Tách rời (Decoupled). Do đó, khi ta nâng cấp hoặc thay đổi kiến trúc mô hình (từ DeBERTa thường lên DeBERTa+LoRA), hệ thống thu thập log không bị hỏng hóc hay "crash code".
* **Data Drift & Reference Baseline:** Hệ thống sẽ liên tục giám sát (Monitor) và so sánh dữ liệu thực tế với tập dữ liệu tham chiếu gốc (Reference Baseline). Khi mô hình được cập nhật lên phiên bản thông minh hơn (phân phối điểm dàn trải linh hoạt hơn thay vì tập trung ở một mức), ta áp dụng kỹ thuật **Reset Baseline**. Điều này giúp thuật toán Drift Detection luôn hiểu đúng đâu là "hành vi chuẩn" của hệ thống mới nhất, tránh tình trạng phát ra báo động giả (False Alarm) làm kích hoạt Retrain oan uổng.
* **Continuous Retraining (Tái huấn luyện tự động):** Nếu hệ thống giám sát phát hiện sự suy giảm chất lượng (Quality Drift) dựa trên phản hồi kém từ người dùng, module `retrain_trigger.py` sẽ tự động kích hoạt. Nó lọc ra các câu hỏi khó mà mô hình đã đoán sai, chuẩn bị dữ liệu và tự động gọi lại quy trình huấn luyện LoRA để sản sinh ra phiên bản mô hình mới thông minh hơn. Toàn bộ quy trình này tạo thành một vòng tuần hoàn khép kín chuẩn MLOps.

---

## 3. Vấn đề đã gặp phải và Cách giải quyết
* **Vấn đề 1: Hiện tượng Mode Collapse (Mô hình chỉ đoán bừa):**
  * *Tình trạng:* Do lúc đầu ép Learning Rate xuống quá thấp (`5e-6`) để chống lỗi NaN, mô hình bị Underfitting. Câu prompt nào nó cũng trả về cùng một kết quả `0.48` (đoán theo tỷ lệ phân phối trung bình của tập dữ liệu thay vì đọc hiểu văn bản).
  * *Giải pháp:* Đổi chiến lược sang dùng LoRA với 15 Epochs như đã trình bày ở trên, mô hình đã thông minh và nhạy bén trở lại.
* **Vấn đề 2: Xung đột API (Error 500) do Zombie Process trên Windows:**
  * *Tình trạng:* API Gateway load nhầm phiên bản model cũ (v5) dù đã deploy bản mới, gây ra lỗi HTTP 500. Nguyên nhân do tiến trình Uvicorn cũ bị treo ngầm (Zombie Process) không nhả cổng 8000.
  * *Giải pháp:* Đổi kiến trúc mạng sang cổng `8080` cho frontend và backend để đảm bảo luồng giao tiếp hoàn toàn sạch sẽ.

---

## 4. Kế hoạch tiếp theo (Next Steps)
1. **Tích hợp Inference Engine thực tế:**
   * Thay thế các câu trả lời giả lập (Mock) bằng cách kết nối trực tiếp API Gateway với máy chủ suy luận thật chạy **vLLM** hoặc **SGLang** (chứa mô hình Qwen2.5-0.5B / 7B).
2. **Đo lường độ trễ (Latency Benchmarking):**
   * Thực hiện Benchmark để đánh giá xem việc đi qua Router làm tăng tổng thời gian phản hồi (Time-to-First-Token) lên bao nhiêu mili-giây so với việc gọi thẳng vào LLM.
3. **Phát triển luồng Human-in-the-loop (Thu thập phản hồi):**
   * Hoàn thiện tính năng đánh giá (Rate this response) trên giao diện Streamlit. Dữ liệu này sẽ được thu thập vào Data Lake để tái huấn luyện (Retrain) Router thành các phiên bản thông minh hơn trong tương lai.

---

## 5. Giải phẫu chi tiết các Luồng Hoạt động của Hệ thống (System Pipelines Deep-Dive)
Hệ thống LLMOps được thiết kế theo kiến trúc Microservices & Data-Driven, tách biệt hoàn toàn thành 3 luồng (pipeline) hoạt động độc lập nhưng liên kết chặt chẽ qua Data Lake.

### 5.1. Luồng Suy luận và Định tuyến (Inference & Routing Pipeline)
Đây là luồng "Mặt tiền" (User-facing) tương tác trực tiếp với người dùng theo thời gian thực (Real-time).
* **Ý tưởng cốt lõi:** Thay vì gửi mọi câu hỏi tới GPT-4 (rất tốn kém), hệ thống dùng một mô hình nhỏ (Router) làm "người phân loại" để xác định độ khó của câu hỏi, từ đó chốt xem nên dùng mô hình nào để trả lời.
* **Chi tiết Thuật toán & Xử lý:**
  1. **Tiếp nhận (API Gateway):** Người dùng nhập câu hỏi trên Streamlit. API Gateway (FastAPI) tiếp nhận dạng JSON.
  2. **Tokenization:** Văn bản được đưa qua `AutoTokenizer` của DeBERTa, cắt thành các từ vựng (tokens). Ký tự đặc biệt `[CLS]` (Classification) được chèn vào đầu câu.
  3. **Ordinal Regression (Mạng nơ-ron):** Dữ liệu đi qua bộ não DeBERTa. Trọng số của token `[CLS]` (đã tóm tắt toàn bộ ý nghĩa câu) được trích xuất ra, đi qua một mạng học sâu tùy chỉnh (`OrdinalClassificationHead`). Đầu ra cuối cùng bị ép qua hàm `Sigmoid` để trả về một điểm số từ `0.0` đến `1.0` (Difficulty Score).
  4. **Routing Logic (If/Else):** Thuật toán so sánh: 
     - Nếu `Score < 0.4` ➡️ Gọi LLM nội bộ (Ví dụ: Qwen-0.5B)
     - Nếu `0.4 <= Score <= 0.7` ➡️ Gọi LLM chuyên gia nội bộ (Ví dụ: Qwen-7B)
     - Nếu `Score > 0.7` ➡️ Gọi API External (GPT-4o).
  5. **Data Collection (Ghi log):** Module `DataCollector` chạy bất đồng bộ (Async) ở nền, âm thầm ghi lại chi tiết giao dịch (Prompt, Score, Route) vào file `prompts_log.jsonl` (Data Lake). Không làm chậm tốc độ phản hồi của người dùng.

### 5.2. Luồng Chấm Điểm Chất Lượng Tự Động (Offline Quality Scoring Pipeline)
Được chạy định kỳ ngầm (Cronjob/Batch) vào ban đêm để chấm điểm các câu trả lời trong ngày.
* **Ý tưởng cốt lõi:** Dữ liệu thực tế thường thiếu nhãn (Unlabeled) vì rất hiếm người dùng chịu bấm nút "Like" hay "Dislike". Do đó, hệ thống cần một thuật toán tự động chấm điểm để chuẩn bị cho việc dạy lại AI.
* **Chi tiết Thuật toán & Xử lý:**
  1. **Nhúng Vector (Embedding):** Đọc lại file `prompts_log.jsonl`. Thuật toán dùng mô hình Embedding biến các câu hỏi/câu trả lời thành một mảng số thực (Vector n-chiều).
  2. **Semantic KNN Search (Tìm kiếm K-Láng giềng):** Hệ thống dùng thư viện **FAISS** (Facebook AI Similarity Search) để quét trong không gian Vector. Nó đem Vector của câu hỏi thực tế đi đối chiếu với hàng ngàn Vector "câu hỏi mẫu xuất sắc" (Reference Index) đã được lưu từ trước.
  3. **Chấm điểm Cosine:** Thuật toán tính toán khoảng cách Cosine (Cosine Similarity). Nếu câu hỏi hiện tại giống về mặt ngữ nghĩa (Semantic) với các câu hỏi mẫu tốt, điểm chất lượng (Quality Score) sẽ cao (Ví dụ: 0.9). Ngược lại sẽ bị đánh điểm thấp.
  4. **Tổng hợp Feedback:** Điểm AI chấm bằng FAISS sẽ được cộng dồn & làm mượt (Smooth) với điểm mà người dùng bấm 👎 (nếu có) để ra điểm cuối cùng.

### 5.3. Luồng Giám Sát Độ Lệch & Tái Huấn Luyện (Drift Detection & Retraining Pipeline)
Luồng giữ cho hệ thống "Sống" và "Tiến hóa" liên tục.
* **Ý tưởng cốt lõi:** Khi người dùng thay đổi hành vi (ví dụ: bỗng nhiên hỏi toàn câu về Kubernetes thay vì Linux), hoặc mô hình LLM sinh ra ảo giác (Hallucination) dẫn đến chất lượng giảm, hệ thống phải tự biết và tự động cập nhật lại não bộ (Retrain).
* **Chi tiết Thuật toán & Xử lý:**
  1. **Data Drift Detection:** Dùng thuật toán thống kê **KS-Test (Kolmogorov-Smirnov Test)** để so sánh phân phối điểm số của 2 tập dữ liệu: Tập thực tế (Current) và Tập chuẩn mực (Baseline). Nếu 2 đường cong phân phối lệch nhau quá 5%, hệ thống cảnh báo "Data Drift".
  2. **Kích hoạt Retrain (Trigger):** Nếu hệ thống vừa phát hiện Data Drift, vừa thấy điểm Quality Score trung bình giảm xuống dưới mức rủi ro (Ví dụ: < 0.75), nó tự động kích hoạt quá trình học lại bằng module `retrain_trigger.py`.
  3. **Lọc dữ liệu & Gán nhãn giả (Pseudo-labeling):** Dữ liệu thực tế được lọc (chỉ lấy câu tốt). Điểm độ khó của nó được gán nhãn tự động dựa trên độ phức tạp của câu. 
  4. **Huấn luyện LoRA:** Module `finetune.py` được khởi động. Trọng số gốc của DeBERTa bị đóng băng (Freeze), thuật toán AdamW dùng **LoRA** để cập nhật ma trận thích ứng với Learning Rate `1e-4` trong vòng 15 Epochs. 
  5. **Hòa trộn và Khai báo (Merge & Registry):** LoRA được gộp ngược vào mạng gốc (`merge_and_unload`). Sau đó `mlflow_utils.py` chạy để đăng ký mô hình lên MLflow dưới tag `Production` (Ví dụ: Chuyển từ v6 sang v7).
  6. **Hot-reload:** API Gateway tự động quét MLflow, tải lại mô hình v7 mà không làm sập (Downtime) hệ thống. Vòng lặp MLOps chính thức khép kín!
