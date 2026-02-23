# debug_llm.py — 放在项目根并在虚拟环境中运行
import inspect, sys, traceback
import tongyi_agent.llm as mod
from tongyi_agent.llm import DashScopeLLM

print("tongyi_agent.llm module file:", getattr(mod, "__file__", None))
print("DashScopeLLM class file:", inspect.getsourcefile(DashScopeLLM))
print("DashScopeLLM repr:", DashScopeLLM)

try:
    print("\n尝试实例化 DashScopeLLM(api_key='x') ...")
    inst = DashScopeLLM(api_key="x")
    print("实例化成功：", type(inst))
    print("__pydantic_fields_set__ exists?:", hasattr(inst, "__pydantic_fields_set__"))
except Exception:
    print("实例化时抛出异常：")
    traceback.print_exc()
    sys.exit(1)