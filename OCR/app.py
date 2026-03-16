from __future__ import annotations

import streamlit as st

from analysis.approval_extractor import extract_approval_entries_from_pdf_bytes
from analysis.document_parser import parse_uploaded_file
from analysis.router import classify_document
from config import PROVIDER_MODELS, get_default_provider, get_provider_api_key, load_env_file
from llm_clients.factory import ask_model_text
from prompt.rules import BASE_INSTRUCTION, ROUTER_RULES
from security.auth import validate_api_key

MAX_CONTEXT_CHARS = 18000
MAX_CHAT_HISTORY = 6


def _humanize_error(exc: Exception) -> str:
    message = str(exc)
    lower = message.lower()

    if (
        "nodename nor servname provided" in lower
        or "name or service not known" in lower
        or "failed to establish a new connection" in lower
    ):
        return (
            "無法連線到 LLM 服務（DNS 解析失敗）。請檢查網路、VPN/代理設定、或本機 DNS。"
            "你也可以先在側邊欄切換成 Google Gemini。"
        )

    if "timed out" in lower:
        return "連線逾時。請確認網路狀態，或稍後再試一次。"

    return message


def _build_verified_approvals_text(approval_entries: list[dict[str, str | int]]) -> str:
    lines: list[str] = []
    for entry in approval_entries:
        role = str(entry.get("職位", "")).strip()
        name = str(entry.get("姓名", "")).strip()
        note = str(entry.get("手寫內容", "")).strip()
        page = str(entry.get("頁碼", "")).strip()

        if not role:
            continue

        human = name or "（姓名待確認）"
        tail = f"；手寫內容：{note}" if note else ""
        lines.append(f"- 第{page}頁｜{role}：{human}{tail}")

    return "\n".join(lines)


def _build_system_message(
    doc_category: str,
    full_text: str,
    approval_entries: list[dict[str, str | int]] | None = None,
) -> dict[str, str]:
    specific_instruction = ROUTER_RULES.get(doc_category, ROUTER_RULES["GENERAL"])
    safe_context = full_text[:MAX_CONTEXT_CHARS]
    verified_section = ""
    if approval_entries:
        verified_lines = _build_verified_approvals_text(approval_entries)
        if verified_lines:
            verified_section = (
                "\n\n簽核欄位校正結果（高優先級事實，若與 OCR 原文衝突請以此為準）：\n"
                f"{verified_lines}"
            )

    return {
        "role": "system",
        "content": (
            f"{BASE_INSTRUCTION}\n\n"
            f"{specific_instruction}\n\n"
            "額外規則：若提問涉及簽核流程、人名、職位、日期，必須優先使用"
            "「簽核欄位校正結果」，不得被 OCR 亂序文字覆蓋。\n\n"
            f"Context:\n{safe_context}{verified_section}"
        ),
    }


def _initialize_session_state() -> None:
    default_provider = get_default_provider()

    if "full_text" not in st.session_state:
        st.session_state.full_text = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "doc_category" not in st.session_state:
        st.session_state.doc_category = "GENERAL"
    if "provider" not in st.session_state:
        st.session_state.provider = default_provider
    if "model" not in st.session_state:
        st.session_state.model = PROVIDER_MODELS[st.session_state.provider][0]
    if "preview_images" not in st.session_state:
        st.session_state.preview_images = []
    if "approval_entries" not in st.session_state:
        st.session_state.approval_entries = []


def render_app() -> None:
    load_env_file()

    st.set_page_config(page_title="動態路由文件分析系統", layout="wide")
    _initialize_session_state()

    with st.sidebar:
        st.header("系統設定")

        provider_options = list(PROVIDER_MODELS.keys())
        if st.session_state.provider not in provider_options:
            st.session_state.provider = get_default_provider()
        provider_index = provider_options.index(st.session_state.provider)
        provider = st.selectbox("LLM 提供商", provider_options, index=provider_index)
        st.session_state.provider = provider

        env_api_key = get_provider_api_key(provider)
        manual_api_key = st.text_input(
            f"{provider} API Key (可留空使用 .env)",
            type="password",
        )

        active_api_key = manual_api_key.strip() or env_api_key
        if env_api_key:
            st.caption("已偵測 .env API Key，可直接使用。")

        st.divider()
        st.header("推理模型")

        model_options = PROVIDER_MODELS[provider]
        if st.session_state.model not in model_options:
            st.session_state.model = model_options[0]

        model_index = model_options.index(st.session_state.model)
        selected_model = st.selectbox("選擇推理模型", model_options, index=model_index)
        st.session_state.model = selected_model

        st.divider()
        st.write("OCR 引擎狀態: EasyOCR (懶載入，首次辨識時啟動)")

        if st.button("清除對話紀錄"):
            st.session_state.chat_history = []
            st.session_state.doc_category = "GENERAL"
            st.session_state.full_text = ""
            st.session_state.preview_images = []
            st.session_state.approval_entries = []
            st.rerun()

    col_file, col_chat = st.columns([1, 1])

    with col_file:
        st.header("1. 智慧文件解析")
        uploaded_file = st.file_uploader("上傳 PDF 或照片檔", type=["pdf", "png", "jpg", "jpeg"])
        file_extension = (
            uploaded_file.name.rsplit(".", 1)[-1].lower()
            if uploaded_file and "." in uploaded_file.name
            else ""
        )

        if uploaded_file and st.button("執行解析並啟動動態路由"):
            if not validate_api_key(active_api_key):
                st.error("請先輸入 API Key（或在 .env 設定）。")
            else:
                progress = st.progress(0.0)
                try:
                    with st.spinner("系統正在萃取內容並進行分類..."):
                        full_text, previews = parse_uploaded_file(uploaded_file, progress_callback=progress.progress)
                        st.session_state.full_text = full_text
                        st.session_state.preview_images = previews
                        st.session_state.approval_entries = []
                        if file_extension == "pdf":
                            st.session_state.approval_entries = extract_approval_entries_from_pdf_bytes(
                                uploaded_file.getvalue()
                            )
                        st.session_state.doc_category = classify_document(
                            provider=provider,
                            api_key=active_api_key,
                            model_id=selected_model,
                            text=full_text,
                        )

                        st.session_state.chat_history = []
                        initial_prompt = "我已經上傳了一份文件，請先幫我做一個簡短的內容總結。"
                        st.session_state.chat_history.append({"role": "user", "content": initial_prompt})

                        system_msg = _build_system_message(
                            st.session_state.doc_category,
                            st.session_state.full_text,
                            st.session_state.approval_entries,
                        )
                        initial_answer = ask_model_text(
                            provider=provider,
                            api_key=active_api_key,
                            model_id=selected_model,
                            messages=[system_msg, {"role": "user", "content": initial_prompt}],
                        )
                        st.session_state.chat_history.append({"role": "assistant", "content": initial_answer})

                    st.success(f"解析完成！系統判定文件類型為：{st.session_state.doc_category}")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"處理失敗：{_humanize_error(exc)}")
                finally:
                    progress.empty()

        if st.session_state.full_text:
            st.subheader("辨識結果")
            st.text_area(
                "辨識到的文字內容",
                value=st.session_state.full_text,
                height=320,
                disabled=True,
            )

            if st.session_state.approval_entries:
                st.subheader("簽核欄位辨識")
                st.dataframe(
                    st.session_state.approval_entries,
                    use_container_width=True,
                    hide_index=True,
                )

            if st.session_state.preview_images:
                st.subheader("圖片內容預覽")
                for idx, image in enumerate(st.session_state.preview_images, start=1):
                    st.image(image, caption=f"上傳圖片 {idx}", use_container_width=True)

    with col_chat:
        st.header("2. 內容診斷助理")
        st.caption(f"當前啟動策略：**{st.session_state.doc_category}** 模型糾錯機制")

        chat_container = st.container(height=550)
        with chat_container:
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        prompt = st.chat_input("詢問關於這份文件的任何問題...")
        if prompt:
            if not validate_api_key(active_api_key):
                st.error("請輸入有效的 API Key（或在 .env 設定）。")
            elif not st.session_state.full_text:
                st.warning("請先在左側完成文件解析。")
            else:
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                with chat_container.chat_message("user"):
                    st.markdown(prompt)

                system_msg = _build_system_message(
                    st.session_state.doc_category,
                    st.session_state.full_text,
                    st.session_state.approval_entries,
                )
                history_slice = st.session_state.chat_history[-MAX_CHAT_HISTORY:]
                messages_to_send = [system_msg] + history_slice

                try:
                    with chat_container.chat_message("assistant"):
                        with st.spinner("分析中..."):
                            response = ask_model_text(
                                provider=provider,
                                api_key=active_api_key,
                                model_id=selected_model,
                                messages=messages_to_send,
                            )
                            st.markdown(response)
                            st.session_state.chat_history.append(
                                {"role": "assistant", "content": response}
                            )
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"回覆失敗：{_humanize_error(exc)}")


if __name__ == "__main__":
    render_app()
