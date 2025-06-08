import os
import sys
import json
import joblib
import dill

def load_pkl(path):
    for loader in [joblib.load, lambda p: dill.load(open(p, 'rb'))]:
        try:
            return loader(path)
        except Exception:
            pass
    raise ValueError("Unsupported or corrupted pickle format.")

def save_to_file(data, output_path):
    # 파일 확장자에 따라 저장 방식 선택
    ext = os.path.splitext(output_path)[1].lower()

    with open(output_path, 'w', encoding='utf-8') as f:
        if ext == '.json':
            try:
                json.dump(data, f, ensure_ascii=False, indent=4)
            except TypeError:
                f.write(str(data))  # JSON 직렬화 실패 시 fallback
        else:
            f.write(str(data))

    print(f"저장 완료: {output_path}")


if __name__ == "__main__":
    input_path = 'CLoSD_t2m_finetune_1env_sit.pkl'        
    output_path = 'CLoSD_t2m_finetune_1env_sit.txt'        

    data = load_pkl(input_path)
    save_to_file(data, output_path)
