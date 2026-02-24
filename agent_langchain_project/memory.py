import os
import json
import time
import tempfile
from typing import Optional, List, Dict, Any
import logging
from .context_request_id import get_request_id

logger = logging.getLogger("agent.memory")

# 提供两种内存后端：FaissMemory（如果可用）和 JsonMemoryFallback
class JsonMemoryFallback:
    def __init__(self, path: str = "~/.agent_vector_store_fallback.json"):
        self.path = os.path.expanduser(path)
        self.texts: List[str] = []
        self.metadatas: List[dict] = []
        self.max_items = int(os.getenv("AGENT_MAX_MEMORY_ITEMS", "10000"))
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.texts = data.get("texts", [])
                    self.metadatas = data.get("metadatas", [])
            except Exception:
                logger.exception("memory.load_fail", extra={"path": self.path})

    def add(self, text: str, meta: Optional[dict] = None):
        # trim if necessary
        try:
            if len(self.texts) >= self.max_items:
                # remove oldest
                self.texts.pop(0)
                self.metadatas.pop(0)
        except Exception:
            pass

        idx = len(self.texts)
        self.texts.append(text)
        self.metadatas.append(meta or {"timestamp": time.time()})
        # prefer context request_id, fallback to meta's request_id if available
        rid = get_request_id() or (meta.get("request_id") if isinstance(meta, dict) else None)
        logger.info("memory.add", extra={"request_id": rid, "index": idx, "meta_summary": (meta or {})})
        return idx

    def search(self, query: str, k: int = 5):
        res = []
        ql = query.lower()
        for t, m in zip(self.texts, self.metadatas):
            try:
                if ql in t.lower() or any(ql in str(v).lower() for v in (m or {}).values()):
                    res.append({"text": t, "meta": m, "score": 1.0})
            except Exception:
                continue
        rid = get_request_id()
        logger.debug("memory.search", extra={"request_id": rid, "query_summary": query[:200], "hits": len(res)})
        return res[:k]

    def persist(self):
        try:
            dirp = os.path.dirname(self.path)
            if dirp:
                os.makedirs(dirp, exist_ok=True)
            # 原子写入
            fd, tmppath = tempfile.mkstemp(dir=dirp or None, prefix=".tmp.", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump({"texts": self.texts, "metadatas": self.metadatas}, f, ensure_ascii=False, indent=2)
            os.replace(tmppath, self.path)
            # prefer context request_id; if None, try to pick last saved meta request_id
            rid = get_request_id()
            if rid is None and self.metadatas:
                last_meta = self.metadatas[-1]
                if isinstance(last_meta, dict):
                    rid = last_meta.get("request_id")
            logger.info("memory.persist.success", extra={"request_id": rid, "path": self.path, "n_items": len(self.texts)})
            return True
        except Exception:
            logger.exception("memory.persist.fail", extra={"request_id": get_request_id(), "path": self.path})
            return False

    def info(self):
        return {"n_vectors": len(self.texts), "path": self.path}

def make_long_memory(index_path: str, model_path: str):
    """
    尝试构造 FaissMemory（如果环境满足），否则返回 JsonMemoryFallback。
    这里保持实现简单：如果 faiss/sentence_transformers 可用则构造相似功能，否则回退。
    """
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss  # type: ignore
    except Exception as e:
        logger.warning("faiss_unavailable", extra={"error": str(e)})
        return JsonMemoryFallback(path=index_path + ".json")

    # 简化的 FaissMemory（使用 IndexFlatIP + texts/metadatas parallel）
    class FaissMemorySimple:
        def __init__(self, index_path=index_path, model_name=model_path, local_only=True):
            self.index_path = os.path.expanduser(index_path)
            self._meta_path = self.index_path + ".meta.json"
            self._idx_path = self.index_path + ".index"
            self.model_name = model_name
            self.np = np
            self.faiss = faiss
            self.model = SentenceTransformer(model_name)
            self.dim = self.model.get_sentence_embedding_dimension()
            # 使用 IndexIDMap 保持 id 映射
            base_index = faiss.IndexFlatIP(self.dim)
            self.index = faiss.IndexIDMap(base_index)
            self.texts: List[str] = []
            self.metadatas: List[dict] = []
            # try load meta
            if os.path.exists(self._meta_path):
                try:
                    with open(self._meta_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        self.texts = data.get("texts", [])
                        self.metadatas = data.get("metadatas", [])
                except Exception:
                    logger.exception("faiss.meta_load_fail", extra={"meta_path": self._meta_path})
            # try load index file (best-effort)
            if os.path.exists(self._idx_path):
                try:
                    self.index = faiss.read_index(self._idx_path)
                except Exception:
                    logger.exception("faiss.index_load_fail", extra={"idx_path": self._idx_path})

        def _embed(self, texts: List[str]):
            embs = self.model.encode(texts, show_progress_bar=False)
            arr = self.np.array(embs, dtype="float32")
            norms = self.np.linalg.norm(arr, axis=1, keepdims=True) + 1e-12
            arr = arr / norms
            return arr

        def add(self, text: str, meta: Optional[dict] = None):
            vec = self._embed([text])
            new_id = len(self.texts)
            try:
                self.index.add_with_ids(vec, self.np.array([new_id], dtype="int64"))
            except Exception:
                try:
                    self.index.add(vec)
                except Exception:
                    pass
            self.texts.append(text)
            self.metadatas.append(meta or {"timestamp": time.time()})
            rid = get_request_id() or (meta.get("request_id") if isinstance(meta, dict) else None)
            logger.info("faiss.add", extra={"request_id": rid, "index": new_id})
            return new_id

        def search(self, query: str, k: int = 5):
            if len(self.texts) == 0:
                return []
            qv = self._embed([query])
            D, I = self.index.search(qv, k)
            results = []
            for score, idx in zip(D[0], I[0]):
                if idx < 0 or idx >= len(self.texts):
                    continue
                results.append({"text": self.texts[idx], "meta": self.metadatas[idx], "score": float(score)})
            logger.debug("faiss.search", extra={"request_id": get_request_id(), "query": query[:200], "n_results": len(results)})
            return results

        def persist(self):
            try:
                # write index (best-effort)
                try:
                    self.faiss.write_index(self.index, self._idx_path)
                except Exception:
                    pass
                dirp = os.path.dirname(self._meta_path)
                if dirp:
                    os.makedirs(dirp, exist_ok=True)
                fd, tmppath = tempfile.mkstemp(dir=dirp or None, prefix=".meta.", suffix=".json")
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump({"texts": self.texts, "metadatas": self.metadatas}, f, ensure_ascii=False, indent=2)
                os.replace(tmppath, self._meta_path)
                rid = get_request_id()
                if rid is None and self.metadatas:
                    last_meta = self.metadatas[-1]
                    if isinstance(last_meta, dict):
                        rid = last_meta.get("request_id")
                logger.info("faiss.persist.success", extra={"request_id": rid, "meta_path": self._meta_path})
                return True
            except Exception:
                logger.exception("faiss.persist.fail", extra={"request_id": get_request_id(), "meta_path": self._meta_path})
                return False

        def info(self):
            return {"n_vectors": len(self.texts), "meta_path": self._meta_path, "idx_path": getattr(self, "_idx_path", None)}

    return FaissMemorySimple()

# 便于外部调用的单例初始化器（cli 会调用）
_long_memory = None
def init_long_memory(index_path: str, model_path: str):
    global _long_memory
    if _long_memory is None:
        _long_memory = make_long_memory(index_path, model_path)
        # log init (may not have request_id)
        rid = get_request_id()
        logger.info("long_memory.init", extra={"request_id": rid, "index_path": index_path, "model_path": model_path})
    return _long_memory

def get_long_memory():
    return _long_memory