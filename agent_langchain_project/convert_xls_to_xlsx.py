#!/usr/bin/env python3
import sys
import pandas as pd

def usage():
    print("用法: python convert_xls_to_xlsx.py <src.xls> <dst.xlsx>")
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()
    src = sys.argv[1]
    dst = sys.argv[2]
    print(f"读取: {src}")
    df = pd.read_excel(src, engine="xlrd", dtype=str)
    df.to_excel(dst, index=False, engine="openpyxl")
    print(f"已保存: {dst}")
