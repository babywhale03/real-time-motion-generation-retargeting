#!/usr/bin/env python3
# pkl 파일을 탐색

import sys, os, joblib, dill

def load_pkl(path):
    for loader in [joblib.load, lambda p: dill.load(open(p, 'rb'))]:
        try: return loader(path)
        except: pass
    raise ValueError("Unsupported or corrupted pickle format.")

def explore(obj, indent=0, key_path=""):
    pad = "  " * indent
    typename = type(obj).__name__
    prefix = f"{pad}{key_path}: {typename}"

    if isinstance(obj, dict):
        print(f"{prefix} {{")
        for k, v in obj.items():
            explore(v, indent + 1, str(k))
        print(f"{pad}}}")
    elif isinstance(obj, list):
        print(f"{prefix} [list of {len(obj)} items]")
        if obj and indent < 5:
            explore(obj[0], indent + 1, "[0]")
    elif hasattr(obj, 'shape'):
        print(f"{prefix} [shape: {tuple(obj.shape)}]")
        if obj.size <= 50:  # 너무 크면 생략
            print(f"{pad}  values: {obj}")
        else:
            print(f"{pad}  values (truncated): {obj[:5]} ...")
    elif isinstance(obj, (str, int, float)):
        print(f"{prefix} = {repr(obj)}")
    else:
        print(f"{prefix} = {repr(obj)[:60]}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python explore_tree.py <file.pkl>")
        return
    path = sys.argv[1]
    if not os.path.exists(path): print("File not found"); return
    try:
        data = load_pkl(path)
        st = data.get("0_0", {}).get("skeleton_tree", None)
        if st is None:
            print("'skeleton_tree' not found under key '0_0'")
            return
        print("# Exploring skeleton_tree:\n")
        explore(st)
    except Exception as e:
        print("Failed to load:", e)

if __name__ == "__main__":
    main()
