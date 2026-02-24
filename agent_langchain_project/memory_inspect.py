# memory_inspect.py
# 用于解析并打印 C:\Users\86137\.agent_vector_store.json 的元数据和文本预览
import json, os, pprint

PATH = r"C:\Users\86137\.agent_vector_store.json"

def main():
    if not os.path.exists(PATH):
        print("文件不存在:", PATH)
        return
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("读取 JSON 失败:", e)
        return

    texts = data.get("texts") or []
    metas = data.get("metadatas") or []
    n = max(len(texts), len(metas))
    print("文件:", PATH)
    print("文本条目数:", len(texts))
    print("元数据条目数:", len(metas))
    print("总条目(推算):", n)
    print("---- 最后 20 条（若有） ----")
    start = max(0, n - 20)
    for i in range(start, n):
        t = texts[i] if i < len(texts) else ""
        m = metas[i] if i < len(metas) else {}
        print("---- idx", i, "----")
        pprint.pprint({"meta": m, "text_preview": (t[:400] + ("..." if len(t) > 400 else ""))})
    # 额外：查找 meta 中是否含 'persona' 或 type == 'persona'
    found = []
    for i, m in enumerate(metas):
        try:
            if isinstance(m, dict):
                if "persona" in json.dumps(m, ensure_ascii=False).lower():
                    found.append((i, m))
                if m.get("type") == "persona":
                    found.append((i, m))
        except Exception:
            continue
    print("---- persona-like found count:", len(found))
    for idx, mm in found[:10]:
        print("idx", idx, "meta:", mm)

if __name__ == "__main__":
    main()