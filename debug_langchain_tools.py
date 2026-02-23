# 调试脚本：诊断 build_langchain_tools 无法构建 LangChain 工具的原因
import traceback
from tongyi_agent import tools

print("=== build_tools() 列表 ===")
try:
    base = tools.build_tools()
    print("build_tools count:", len(base))
    for t in base:
        print(" -", t.name, "->", getattr(t.func, "__name__", repr(t.func)))
except Exception:
    print("调用 build_tools() 异常：")
    traceback.print_exc()

print("\n=== langchain.tools.Tool 检查 ===")
try:
    from langchain.tools import Tool as LC_Tool
    print("Imported langchain.tools.Tool:", LC_Tool)
    print("hasattr(from_function):", hasattr(LC_Tool, "from_function"))
    # show signature if possible
    try:
        import inspect
        print("Tool.from_function signature:", inspect.signature(LC_Tool.from_function) if hasattr(LC_Tool, "from_function") else "no from_function")
    except Exception:
        pass
except Exception:
    print("无法 import langchain.tools.Tool:")
    traceback.print_exc()

print("\n=== 尝试调用 build_langchain_tools() 并捕获异常 ===")
try:
    lct = tools.build_langchain_tools()
    print("build_langchain_tools returned count:", len(lct))
    # 如果非空，打印每个工具的 repr 或 name
    for i, lt in enumerate(lct):
        try:
            name = getattr(lt, "name", None) or getattr(lt, "__name__", repr(lt))
        except Exception:
            name = repr(lt)
        print(f" LC tool #{i} -> {name}")
except Exception:
    print("build_langchain_tools() 抛出异常：")
    traceback.print_exc()

print("\n=== 逐个尝试单独包裹 (更详细的错误捕获) ===")
try:
    from langchain.tools import Tool as LC_Tool
    base = tools.build_tools()
    for t in base:
        print(f"\n-- 尝试包装工具: {t.name} ({getattr(t.func,'__name__',repr(t.func))})")
        try:
            if hasattr(LC_Tool, "from_function"):
                lc = LC_Tool.from_function(func=t.func, name=t.name, description=t.description, return_direct=True)
            else:
                lc = LC_Tool(func=t.func, name=t.name, description=t.description)
            print("  包装成功:", lc)
        except Exception:
            print("  包装失败，traceback:")
            traceback.print_exc()
except Exception:
    print("在逐个包装时发生异常：")
    traceback.print_exc()