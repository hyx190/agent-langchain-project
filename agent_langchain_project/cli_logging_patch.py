import logging

class RequestIdFilter(logging.Filter):
    """确保每个 LogRecord 都有 request_id 属性，避免格式化失败。"""
    def filter(self, record):
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        return True

def register_request_id_filter():
    root = logging.getLogger()
    root.addFilter(RequestIdFilter())
    for h in root.handlers:
        h.addFilter(RequestIdFilter())