import numpy as np
data = np.load("results.npy", allow_pickle=True).item()
print("motion shape:", data['motion'].shape)
print("first frame data (one joint):", data['motion'][0][:, :, 0])  # 첫 프레임

# import numpy as np

# # 파일 경로 지정
# file_path = 'results.npy'

# # 파일 로드
# data = np.load(file_path, allow_pickle=True)

# # 내용 출력
# print("데이터 타입:", type(data))
# print("데이터 형태:", data.shape)
# print("데이터 내용:\n", data)