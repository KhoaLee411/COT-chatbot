"""
Streamlit frontend cho Coach On Tap AI Agent.
"""
import streamlit as st
import requests
from loguru import logger

import random
import numpy as np
import google.generativeai as genai
from config.settings import API_URL, BACKEND_HOST, FAQ_PATH, GEMINI_API_KEY, EMBEDDING_MODEL

genai.configure(api_key=GEMINI_API_KEY)


# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Coach On Tap - AI Agent",
    initial_sidebar_state="collapsed",
)

ss = st.session_state

if "messages" not in ss:
    ss.messages = [
        {
            "role": "assistant",
            "content": "Xin chào! Tôi là trợ lý AI của Coach On Tap. Tôi có thể giúp gì cho bạn hôm nay?",
        }
    ]


# ── API client ────────────────────────────────────────────────────────────────

def stream_chat(query: str, history: list):
    """Gọi API và yield từng token."""
    try:
        with requests.post(
            f"{API_URL}/chat",
            json={"query": query, "chat_history": history},
            stream=True,
            timeout=60,
        ) as response:
            if response.status_code == 200:
                for chunk in response.iter_content(chunk_size=None):
                    if chunk:
                        yield chunk.decode("utf-8")
            else:
                yield f"[Lỗi {response.status_code}]: {response.text}"
    except requests.exceptions.ConnectionError:
        yield f"[Không thể kết nối tới server tại {API_URL}]"
    except Exception as e:
        logger.error(f"Stream error: {e}")
        yield f"[Lỗi kết nối: {e}]"


# ── FAQ Suggestions ───────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def fetch_faqs():
    """Fetch FAQ từ backend và trả về danh sách các dict {question, answer} phẳng."""
    try:
        url = f"{BACKEND_HOST}{FAQ_PATH}"
        res = requests.get(url, timeout=10)
        if res.status_code == 200:
            topics = res.json()
            flat_faqs = []
            for topic in topics:
                items = topic.get("items") or []
                for item in items:
                    q = str(item.get("question", "")).strip()
                    a = str(item.get("answer", "")).strip()
                    # Lọc bớt các câu rác
                    if q and a and len(a) > 10 and "placeholder" not in q.lower():
                        flat_faqs.append({"question": q, "answer": a})
                        
            # Embed FAQs
            questions = [f["question"] for f in flat_faqs]
            if questions:
                try:
                    result = genai.embed_content(
                        model=EMBEDDING_MODEL,
                        content=questions,
                        task_type="retrieval_document"
                    )
                    embeddings = result['embedding']
                    for i, faq in enumerate(flat_faqs):
                        faq["embedding"] = embeddings[i]
                except Exception as e:
                    logger.error(f"Lỗi embedding FAQs: {e}")
                    
            return flat_faqs
    except Exception as e:
        logger.error(f"Lỗi khi lấy FAQ: {e}")
    return []

def get_suggestions(history, flat_faqs, num=3):
    """Lấy danh sách câu hỏi gợi ý bằng embedding similarity."""
    if not flat_faqs:
        return []
        
    user_msgs = [m["content"] for m in history if m["role"] == "user"]
    if not user_msgs:
        return random.sample(flat_faqs, min(num, len(flat_faqs)))
        
    last_msg = user_msgs[-1]
    
    try:
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=last_msg,
            task_type="retrieval_query"
        )
        query_emb = res['embedding']
    except Exception as e:
        logger.error(f"Lỗi embedding query: {e}")
        return random.sample(flat_faqs, min(num, len(flat_faqs)))
        
    scored_faqs = []
    for faq in flat_faqs:
        emb = faq.get("embedding")
        if emb:
            score = np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb))
            scored_faqs.append((score, faq))
        else:
            scored_faqs.append((0, faq))
            
    scored_faqs.sort(key=lambda x: x[0], reverse=True)
    return [f for s, f in scored_faqs[:num]]


# ── UI ────────────────────────────────────────────────────────────────────────

def main():
    st.title("Coach On Tap AI Agent")

    for message in ss.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Xử lý lấy FAQ và hiển thị gợi ý
    flat_faqs = fetch_faqs()
    
    # Lưu suggestions vào session_state để không bị random lại khi ấn nút (gây mất click)
    if "current_suggestions" not in ss or ss.get("last_msg_count") != len(ss.messages):
        ss.current_suggestions = get_suggestions(ss.messages, flat_faqs)
        ss.last_msg_count = len(ss.messages)

    if ss.current_suggestions:
        st.write("💡 **Gợi ý câu hỏi phổ biến:**")
        # Chia cột để hiển thị các nút bấm ngang hàng
        cols = st.columns(len(ss.current_suggestions))
        for i, faq in enumerate(ss.current_suggestions):
            # Lấy 60 ký tự đầu làm label cho ngắn gọn nếu quá dài
            label = faq["question"][:60] + "..." if len(faq["question"]) > 60 else faq["question"]
            
            # Key dựa trên số message hiện tại để tránh trùng lặp khi re-render
            if cols[i].button(label, key=f"sug_{len(ss.messages)}_{i}", use_container_width=True, help=faq["question"]):
                # Khi click: thêm luôn cặp Q&A vào lịch sử và render lại
                ss.messages.append({"role": "user", "content": faq["question"]})
                ss.messages.append({"role": "assistant", "content": faq["answer"]})
                st.rerun()

    if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
        ss.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            # Lịch sử không bao gồm message vừa thêm (đã append ở trên)
            api_history = ss.messages[:-1]
            full_response = st.write_stream(stream_chat(prompt, api_history))
            ss.messages.append({"role": "assistant", "content": full_response})
            st.rerun()


if __name__ == "__main__":
    main()
