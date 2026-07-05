# Coach On Tap AI Agent (Agentic RAG)

Hệ thống AI Chatbot thông minh cho Coach On Tap, sử dụng kiến trúc Agentic RAG với LangGraph, Gemini và AWS S3.

## Kiến trúc hệ thống
- **Ingestion Flow:** Tự động hóa việc bóc tách PDF từ S3 Data Lake (AWS) qua LlamaParse, tạo embedding bằng Gemini và lưu trữ Vector Index trên S3.
- **User Query Flow:** Luồng xử lý câu hỏi thông qua Guardrails, Analyzer, và RAG Core điều phối bởi LangGraph.

## Cài đặt

```bash
pip install -e .
```

## Khởi động

### 1. Chạy Backend API (FastAPI)
```bash
python api/main.py
```

### 2. Chạy Giao diện người dùng (Streamlit)
```bash
streamlit run main.py
```

### 3. Chạy Ingestion Watcher (Theo dõi dữ liệu mới trên S3)
```bash
python source/ingestion/s3_watcher.py
```

## Cấu trúc thư mục
- `api/`: Backend API.
- `source/agent/`: Logic điều phối Agent (LangGraph).
- `source/rag_core/`: Truy vấn và tổng hợp câu trả lời.
- `source/ingestion/`: Xử lý dữ liệu đầu vào.