import contextvars
import uuid

# request id 存储在 contextvars 中，async/多线程安全
_request_id_var = contextvars.ContextVar("request_id", default=None)

def new_request_id() -> str:
    rid = str(uuid.uuid4())
    _request_id_var.set(rid)
    return rid

def get_request_id() -> str | None:
    return _request_id_var.get()