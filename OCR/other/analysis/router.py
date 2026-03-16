# analysis/router.py

from llm_clients.together_ai import ask_together_text
from prompt.rules import ROUTER_PROMPT_TEMPLATE

def classify_document(api_key: str, model_id: str, text: str) -> str:
    """
    輕量級分類器：讀取文本前 500 字，判斷文件類型以套用專屬規則
    """
    sample_text = text[:500]
    prompt_content = ROUTER_PROMPT_TEMPLATE.format(sample_text=sample_text)
    
    messages = [{"role": "user", "content": prompt_content}]
    
    # 使用極低的 token 限制和溫度來確保只回傳分類標籤
    category = ask_together_text(
        api_key=api_key, 
        model_id=model_id, 
        messages=messages, 
        max_tokens=10, 
        temperature=0.0
    )
    
    if "FINANCIAL" in category.upper(): 
        return "FINANCIAL"
    if "HARDWARE" in category.upper(): 
        return "HARDWARE"
    return "GENERAL"