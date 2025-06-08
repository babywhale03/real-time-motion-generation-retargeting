# import numpy as np

# def parse_bvh_file(file_path):
#     """BVH 파일을 파싱하여 hierarchy와 motion 데이터를 분리합니다."""
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#     motion_start = lines.index("MOTION\n")
#     hierarchy = lines[:motion_start]
#     motion_data = lines[motion_start + 3:]  # skip MOTION, Frames:, Frame Time:

#     motion = np.array([list(map(float, line.strip().split())) for line in motion_data], dtype=np.float32)
#     return hierarchy, motion

# def write_bvh(output_path, hierarchy_lines, motion_data, frame_time=0.0333333):
#     """정규화된 motion 데이터를 기존 hierarchy와 함께 BVH 형식으로 저장합니다."""
#     with open(output_path, 'w') as f:
#         # Write hierarchy
#         for line in hierarchy_lines:
#             f.write(line)
#         # Write motion header
#         f.write("MOTION\n")
#         f.write(f"Frames: {len(motion_data)}\n")
#         f.write(f"Frame Time: {frame_time}\n")
#         # Write each frame
#         for frame in motion_data:
#             frame_str = ' '.join(f"{v:.6f}" for v in frame)
#             f.write(frame_str + "\n")

# # 파일 경로 설정
# closd_path = "closdtest3.bvh"
# lafan_path = "./demo_dir/Lafan1/Lafan1_example.bvh"
# output_path = "closdtest3_normalized.bvh"

# # BVH 파일 로드
# closd_hierarchy, closd_motion = parse_bvh_file(closd_path)
# lafan_hierarchy, lafan_motion = parse_bvh_file(lafan_path)

# # 정규화를 위한 통계 계산
# closd_mean = np.mean(closd_motion, axis=0)
# closd_std = np.std(closd_motion, axis=0)
# lafan_mean = np.mean(lafan_motion, axis=0)
# lafan_std = np.std(lafan_motion, axis=0)

# # 스케일 및 오프셋 계산
# scale = lafan_std / (closd_std + 1e-8)
# offset = lafan_mean - closd_mean * scale

# # 정규화 적용
# normalized_closd_motion = closd_motion * scale + offset

# # 결과 저장
# write_bvh(output_path, closd_hierarchy, normalized_closd_motion)
# print(f"정규화된 BVH 파일이 저장되었습니다: {output_path}")








# offset 수정, rotation 까지 정규화
# import numpy as np

# def parse_bvh_file(file_path):
#     """BVH 파일을 파싱하여 hierarchy와 motion 데이터를 분리합니다."""
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#     motion_start = lines.index("MOTION\n")
#     hierarchy = lines[:motion_start]
#     motion_data = lines[motion_start + 3:]  # skip MOTION, Frames:, Frame Time:

#     motion = np.array([list(map(float, line.strip().split())) for line in motion_data], dtype=np.float32)
#     return hierarchy, motion

# def write_bvh(output_path, hierarchy_lines, motion_data, frame_time=0.0333333):
#     """정규화된 motion 데이터를 지정된 hierarchy와 함께 BVH 형식으로 저장합니다."""
#     with open(output_path, 'w') as f:
#         # Write Lafan-style hierarchy
#         for line in hierarchy_lines:
#             f.write(line)
#         # Write motion section
#         f.write("MOTION\n")
#         f.write(f"Frames: {len(motion_data)}\n")
#         f.write(f"Frame Time: {frame_time}\n")
#         for frame in motion_data:
#             frame_str = ' '.join(f"{v:.6f}" for v in frame)
#             f.write(frame_str + "\n")

# # 파일 경로 설정
# closd_path = "closdtest5.bvh"
# lafan_path = "./demo_dir/Lafan1/Lafan1_example.bvh"
# output_path = "closdtest5_normalized.bvh"

# # BVH 로딩
# closd_hierarchy, closd_motion = parse_bvh_file(closd_path)
# lafan_hierarchy, lafan_motion = parse_bvh_file(lafan_path)

# # 통계 계산
# closd_mean = np.mean(closd_motion, axis=0)
# closd_std = np.std(closd_motion, axis=0)
# lafan_mean = np.mean(lafan_motion, axis=0)
# lafan_std = np.std(lafan_motion, axis=0)
# # print("Mean/Std computed.")

# # 정규화 수행
# scale = lafan_std / (closd_std + 1e-8)
# offset = lafan_mean - closd_mean * scale
# normalized_closd_motion = closd_motion * scale + offset

# # Lafan hierarchy로 저장
# write_bvh(output_path, lafan_hierarchy, normalized_closd_motion)
# print(f"정규화된 BVH 파일이 저장되었습니다: {output_path}")







# position, rotation 모두 정규화화
import numpy as np

def parse_bvh_file(file_path):
    """BVH 파일을 파싱하여 hierarchy와 motion 데이터를 분리합니다."""
    with open(file_path, "r") as f:
        lines = f.readlines()

    motion_start = lines.index("MOTION\n")
    hierarchy = lines[:motion_start]
    motion_data = lines[motion_start + 3:]  # skip MOTION, Frames:, Frame Time:

    motion = np.array([list(map(float, line.strip().split())) for line in motion_data], dtype=np.float32)
    return hierarchy, motion

def write_bvh(output_path, hierarchy_lines, motion_data, frame_time=0.0333333):
    """정규화된 motion 데이터를 기존 hierarchy와 함께 BVH 형식으로 저장합니다."""
    with open(output_path, 'w') as f:
        # Write hierarchy
        for line in hierarchy_lines:
            f.write(line)
        # Write motion header
        f.write("MOTION\n")
        f.write(f"Frames: {len(motion_data)}\n")
        f.write(f"Frame Time: {frame_time}\n")
        # Write each frame
        for frame in motion_data:
            frame_str = ' '.join(f"{v:.6f}" for v in frame)
            f.write(frame_str + "\n")

# 파일 경로
closd_path = "closdtest5.bvh"
lafan_path = "./demo_dir/Lafan1/Lafan1_example.bvh"
output_path = "closdtest5_normalized.bvh"

# 파일 파싱
closd_hierarchy, closd_motion = parse_bvh_file(closd_path)
lafan_hierarchy, lafan_motion = parse_bvh_file(lafan_path)

# 통계 계산
closd_mean = np.mean(closd_motion, axis=0)
closd_std = np.std(closd_motion, axis=0)
lafan_mean = np.mean(lafan_motion, axis=0)
lafan_std = np.std(lafan_motion, axis=0)

# 정규화 수행
scale = lafan_std / (closd_std + 1e-8)
offset = lafan_mean - closd_mean * scale
normalized_closd_motion = closd_motion * scale + offset

# 결과 저장
write_bvh(output_path, lafan_hierarchy, normalized_closd_motion)

print(f"✅ 정규화된 BVH 파일이 저장되었습니다: {output_path}")









# position만 정규화
# import numpy as np

# def parse_bvh_file(file_path):
#     """BVH 파일을 파싱하여 hierarchy와 motion 데이터를 분리합니다."""
#     with open(file_path, "r") as f:
#         lines = f.readlines()

#     motion_start = lines.index("MOTION\n")
#     hierarchy = lines[:motion_start]
#     motion_data = lines[motion_start + 3:]  # skip MOTION, Frames:, Frame Time:

#     motion = np.array([list(map(float, line.strip().split())) for line in motion_data], dtype=np.float32)
#     return hierarchy, motion

# def write_bvh(output_path, hierarchy_lines, motion_data, frame_time=0.0333333):
#     """정규화된 motion 데이터를 기존 hierarchy와 함께 BVH 형식으로 저장합니다."""
#     with open(output_path, 'w') as f:
#         # Write hierarchy
#         for line in hierarchy_lines:
#             f.write(line)
#         # Write motion header
#         f.write("MOTION\n")
#         f.write(f"Frames: {len(motion_data)}\n")
#         f.write(f"Frame Time: {frame_time}\n")
#         # Write each frame
#         for frame in motion_data:
#             frame_str = ' '.join(f"{v:.6f}" for v in frame)
#             f.write(frame_str + "\n")

# # 파일 경로
# closd_path = "closdtest5.bvh"
# lafan_path = "./demo_dir/Lafan1/Lafan1_example.bvh"
# output_path = "closdtest5_normalized.bvh"

# # 파일 파싱
# closd_hierarchy, closd_motion = parse_bvh_file(closd_path)
# lafan_hierarchy, lafan_motion = parse_bvh_file(lafan_path)

# # 통계 계산 (position 채널만: 0~2)
# closd_pos_mean = np.mean(closd_motion[:, :3], axis=0)
# closd_pos_std = np.std(closd_motion[:, :3], axis=0)
# lafan_pos_mean = np.mean(lafan_motion[:, :3], axis=0)
# lafan_pos_std = np.std(lafan_motion[:, :3], axis=0)

# # position 정규화
# scale = lafan_pos_std / (closd_pos_std + 1e-8)
# offset = lafan_pos_mean - closd_pos_mean * scale

# # 전체 motion 복사 후 position만 정규화 적용
# normalized_closd_motion = np.copy(closd_motion)
# normalized_closd_motion[:, :3] = closd_motion[:, :3] * scale + offset

# # 결과 저장 (hierarchy는 원래 cloSD BVH 기준)
# write_bvh(output_path, closd_hierarchy, normalized_closd_motion)

# print(f"✅ position만 정규화된 BVH 파일이 저장되었습니다: {output_path}")
