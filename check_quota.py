from docuquery.config import load_settings
from docuquery.llm import GeminiLLM

s = load_settings()
for name in (s.gen_model, s.judge_model):
    llm = GeminiLLM(s.api_key, name, min_interval=0)
    # one single call straight to the API (no retry loop), so the real error shows up
    try:
        out = llm.client.models.generate_content(model=name, contents="Reply with the single word OK")
        print(name, "-> works:", (out.text or "")[:30])
    except Exception as e:
        print(name, "-> FAILED:", type(e).__name__)
        print(str(e)[:1500])
    print("-" * 40)