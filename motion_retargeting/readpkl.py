# import pickle
# import pandas as pd

# try:
#     with open('CLoSD_t2m_finetune_1env_sit.pkl', 'rb') as f:
#         data = pickle.load(f)
#         if isinstance(data, dict):
#             print("전체 key 목록:")
#             for key in data.keys():
#                 print("-", key)
#         else:
#             print("데이터는 dict 타입이 아닙니다. 타입:", type(data))

# except Exception as e:
#     print("오류 발생:", e)


import sys, os, joblib, dill

def load_pkl(path):
    for loader in [joblib.load, lambda p: dill.load(open(p, 'rb'))]:
        try: return loader(path)
        except: pass
    raise ValueError("Unsupported or corrupted pickle format.")

def inspect(obj, indent=0, max_depth=2):
    pad = "  " * indent
    if isinstance(obj, dict):
        for k, v in obj.items():
            print(f"{pad}- {k}: {type(v).__name__}")
            if indent < max_depth: inspect(v, indent+1, max_depth)
    elif isinstance(obj, list):
        print(f"{pad}- list[{len(obj)}]: {type(obj[0]).__name__ if obj else 'empty'}")
    elif hasattr(obj, 'shape'):
        print(f"{pad}- shape: {tuple(obj.shape)}")
    else:
        print(f"{pad}- {repr(obj)[:60]}")

def main():
    if len(sys.argv) != 2:
        print("Usage: python inspect_pkl.py <file.pkl>")
        return
    path = sys.argv[1]
    if not os.path.exists(path): print("File not found"); return
    try:
        obj = load_pkl(path)
        print(f"# Loaded: {type(obj).__name__}")
        inspect(obj)

        # edit
        # print(obj['0_0']['skeleton_tree']['node_names'])
        print(obj['0_0'].keys())
    except Exception as e:
        print("Failed to load:", e)

if __name__ == "__main__":
    main()

