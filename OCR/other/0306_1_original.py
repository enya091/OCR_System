import streamlit as st
import fitz
import requests
import numpy as np
import cv2
from PIL import Image
import easyocr

# --- 1. 載入深度學習 OCR 引擎 ---
@st.cache_resource
def load_ocr_engine():
    return easyocr.Reader(['ch_tra', 'en'], gpu=False)

reader = load_ocr_engine()

# --- 2. 專屬糾錯規則字典 (動態路由的核心) ---
ROUTER_RULES = {
    "FINANCIAL": "特別注意：此為財務、轉帳或收據文件。若文本中出現類似「5100」、「51,000」或「S1000」等不合理數字組合，請高度懷疑是 OCR 將金錢符號「$」誤認為「5」或「S」。請利用上下文（如台幣、轉帳成功）自動在底層修復為正確金額（如 $1,000）。",
    "HARDWARE": "特別注意：此為硬體設備、螢幕標籤或機身銘牌。文本可能包含高密度的多國語言與安規標章（如 CE, FCC）。請忽略無意義的單一字母噪點或破碎字元，專注於提取並校正型號（Model）、電壓（Rating）、序號等關鍵規格。",
    "GENERAL": "特別注意：請利用上下文語意邏輯，自動修正任何因掃描品質不佳或反光造成的錯別字或形近字，確保資訊通順合理。"
}

# --- 3. 核心 API 呼叫 ---
def ask_together_text(api_key, model_id, messages, max_tokens=2048, temperature=0.2):
    url = "https://api.together.xyz/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    try:
        response = requests.post(url, json=payload, headers=headers)
        result = response.json()
        if response.status_code != 200:
            return f"API 錯誤: {result.get('error', {}).get('message', '請檢查權限')}"
        return result['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"異常: {str(e)}"

# --- 4. 文件分類路由器 (Router) ---
def classify_document(api_key, model_id, text):
    """
    輕量級分類器：讀取文本前 500 字，判斷文件類型以套用專屬規則
    """
    sample_text = text[:500]
    prompt = f"""
    請判斷以下文本的類型。你只能輸出以下三個英文單字之一，絕對不要輸出其他任何說明文字：
    1. FINANCIAL (包含轉帳截圖、發票、收據、報價單等與金額高度相關者)
    2. HARDWARE (包含螢幕標籤、產品銘牌、安規認證、硬體規格等)
    3. GENERAL (合約、一般文章、其他無法歸類者)
    
    文本內容：
    {sample_text}
    """
    messages = [{"role": "user", "content": prompt}]
    # 使用極低的 token 限制和溫度來確保只回傳分類標籤
    category = ask_together_text(api_key, model_id, messages, max_tokens=10, temperature=0.0)
    
    if "FINANCIAL" in category.upper(): return "FINANCIAL"
    if "HARDWARE" in category.upper(): return "HARDWARE"
    return "GENERAL"

# --- 5. 影像辨識模組 ---
def extract_text_from_image(img_array):
    result = reader.readtext(
        img_array, 
        detail=0, 
        paragraph=True, 
        adjust_contrast=True
    )
    return "\n".join(result)

# --- 6. 介面設定 ---
st.set_page_config(page_title="動態路由文件分析系統", layout="wide")

if "full_text" not in st.session_state:
    st.session_state.full_text = ""
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_category" not in st.session_state:
    st.session_state.doc_category = "GENERAL"

with st.sidebar:
    st.header("系統設定")
    tg_api_key = st.text_input("Together AI API Key", type="password")
    
    st.divider()
    st.header("推理模型")
    selected_model = st.selectbox("選擇推理模型", [
        "meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    ])
    
    st.divider()
    st.write("OCR 引擎狀態: EasyOCR (已就緒)")
    
    if st.button("清除對話紀錄"):
        st.session_state.chat_history = []
        st.session_state.doc_category = "GENERAL"
        st.rerun()

# --- 7. 主介面佈局 ---
col_file, col_chat = st.columns([1, 1])

with col_file:
    st.header("1. 智慧文件解析")
    uploaded_file = st.file_uploader("上傳 PDF 或照片檔", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if st.button("執行解析並啟動動態路由"):
            if not tg_api_key:
                st.error("請先輸入 API Key")
            else:
                with st.spinner("系統正在萃取內容並進行分類..."):
                    all_content = []
                    
                    if file_extension == 'pdf':
                        doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
                        progress_bar = st.progress(0)
                        
                        for i in range(len(doc)):
                            page = doc[i]
                            page_text = page.get_text().strip()
                            
                            if len(page_text) < 50:
                                pix = page.get_pixmap(matrix=fitz.Matrix(3, 3))
                                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                                img_cv = cv2.cvtColor(img_data, cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
                                ocr_text = extract_text_from_image(img_cv)
                                page_text = f"[由 EasyOCR 引擎萃取]\n{ocr_text}"
                            else:
                                page_text = f"[原生文字層提取]\n{page_text}"

                            all_content.append(f"--- 第 {i+1} 頁 ---\n{page_text}")
                            progress_bar.progress((i + 1) / len(doc))
                            
                    elif file_extension in ['png', 'jpg', 'jpeg']:
                        pil_img = Image.open(uploaded_file).convert('RGB')
                        raw_img_array = np.array(pil_img)
                        ocr_text = extract_text_from_image(raw_img_array)
                        all_content.append(f"--- 照片檔案內容 ---\n[由 EasyOCR 引擎萃取]\n{ocr_text}")
                        st.image(pil_img, caption="原始上傳影像", use_container_width=True)

                    st.session_state.full_text = "\n\n".join(all_content)
                    
                    # 🚀 啟動動態路由分類
                    st.session_state.doc_category = classify_document(tg_api_key, selected_model, st.session_state.full_text)
                    st.success(f"解析完成！系統判定文件類型為：{st.session_state.doc_category}")
                    
                    initial_prompt = "我已經上傳了一份文件，請先幫我做一個簡短的內容總結。"
                    st.session_state.chat_history.append({"role": "user", "content": initial_prompt})
                    
                    # 組合泛化指令 + 動態路由規則
                    base_instruction = "Identity: 你是專業的文件內容分析官。必須使用「繁體中文」回答。你接收到的文本包含由 OCR 自動萃取的內容，可能存在視覺形近字錯誤（Visual Homoglyph Errors）。請勿向使用者提及解析技術、錯字或修正過程，直接給出合乎邏輯的正確資訊。"
                    specific_instruction = ROUTER_RULES.get(st.session_state.doc_category, ROUTER_RULES["GENERAL"])
                    
                    system_msg = {
                        "role": "system", 
                        "content": f"{base_instruction}\n\n{specific_instruction}\n\nContext:\n{st.session_state.full_text}"
                    }
                    
                    chat_msgs = [system_msg, {"role": "user", "content": initial_prompt}]
                    ans = ask_together_text(tg_api_key, selected_model, chat_msgs)
                    st.session_state.chat_history.append({"role": "assistant", "content": ans})
                    st.rerun()

with col_chat:
    st.header("2. 內容診斷助理")
    # 顯示當前套用的路由策略
    st.caption(f"當前啟動策略：**{st.session_state.doc_category}** 模型糾錯機制")
    
    chat_container = st.container(height=550)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    if prompt := st.chat_input("詢問關於這份文件的任何問題..."):
        if not tg_api_key:
            st.error("請輸入 API Key")
        elif not st.session_state.full_text:
            st.warning("請先在左側完成文件解析")
        else:
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with chat_container.chat_message("user"):
                st.markdown(prompt)
            
            # 對話中持續套用泛化指令 + 動態路由規則
            base_instruction = "Identity: 你是專業的文件內容分析官。必須使用「繁體中文」回答。你接收到的文本包含由 OCR 自動萃取的內容，可能存在視覺形近字錯誤。請勿向使用者提及修正過程，直接給出合乎邏輯的正確資訊。"
            specific_instruction = ROUTER_RULES.get(st.session_state.doc_category, ROUTER_RULES["GENERAL"])
            
            system_msg = {
                "role": "system", 
                "content": f"{base_instruction}\n\n{specific_instruction}\n\nContext:\n{st.session_state.full_text}"
            }
            
            history_slice = st.session_state.chat_history[-5:]
            messages_to_send = [system_msg] + history_slice
            
            with chat_container.chat_message("assistant"):
                with st.spinner("分析中..."):
                    response = ask_together_text(tg_api_key, selected_model, messages_to_send)
                    st.markdown(response)
                    st.session_state.chat_history.append({"role": "assistant", "content": response})
            st.rerun()