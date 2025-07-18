import numpy as np
import joblib
import dill
import transforms3d
from transforms3d.quaternions import quat2mat
from transforms3d.euler import mat2euler
import re


HIERARCHY_STR = """HIERARCHY
ROOT Hips
{
    OFFSET -0.0018 -0.2233 0.0282
    CHANNELS 6 Xposition Yposition Zposition Zrotation Yrotation Xrotation 
    JOINT LeftUpLeg
    {
        OFFSET -0.0068 0.0695 -0.0914
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT LeftLeg
        {
            OFFSET -0.0045 0.0343 -0.3752
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT LeftFoot
            {
                OFFSET -0.0437 -0.0136 -0.398
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT LeftToe
                {
                    OFFSET 0.1193 0.0264 -0.0558
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {
                        OFFSET 0.000000 0.000000 0.000000
                    }
                }
            }
        }
    }
    JOINT RightUpLeg
    {
        OFFSET -0.0043 -0.0677 -0.0905
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT RightLeg
        {
            OFFSET -0.0089 -0.0383 -0.3826
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT RightFoot
            {
                OFFSET -0.0423 0.0158 -0.3984
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT RightToe
                {
                    OFFSET 0.1233 -0.0254 -0.0481
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    End Site
                    {
                        OFFSET 0.000000 0.000000 0.000000
                    }
                }
            }
        }
    }
    JOINT Spine
    {
        OFFSET -0.0267 -0.0025 0.109
        CHANNELS 3 Zrotation Yrotation Xrotation
        JOINT Spine1
        {
            OFFSET 0.0011 0.0055 0.1352
            CHANNELS 3 Zrotation Yrotation Xrotation
            JOINT Spine2
            {
                OFFSET 0.0254 0.0015 0.0529
                CHANNELS 3 Zrotation Yrotation Xrotation
                JOINT Neck
                {
                    OFFSET -0.0429 -0.0028 0.2139
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT Head
                    {
                        OFFSET 0.0513 0.0052 0.065
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        End Site
                        {
                            OFFSET 0.000000 0.000000 0.000000
                        }
                    }
                }
                JOINT LeftShoulder
                {
                    OFFSET -0.0341 0.0788 0.1217
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT LeftArm
                    {
                        OFFSET -0.0089 0.091 0.0305
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        JOINT LeftForeArm
                        {
                            OFFSET -0.0275 0.2596 -0.0128
                            CHANNELS 3 Zrotation Yrotation Xrotation
                            JOINT LeftHand
                            {
                                OFFSET -0.0149 0.084 -0.0082
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                End Site
                                {
                                    OFFSET 0.000000 0.000000 0.000000
                                }
                            }
                        }
                    }
                }
                JOINT RightShoulder
                {
                    OFFSET -0.0386 -0.0818 0.1188
                    CHANNELS 3 Zrotation Yrotation Xrotation
                    JOINT RightArm
                    {
                        OFFSET -0.0091 -0.096 0.0326
                        CHANNELS 3 Zrotation Yrotation Xrotation
                        JOINT RightForeArm
                        {
                            OFFSET -0.0214 -0.2537 -0.0133
                            CHANNELS 3 Zrotation Yrotation Xrotation
                            JOINT RightHand
                            {
                                OFFSET -0.0103 -0.0846 -0.0061
                                CHANNELS 3 Zrotation Yrotation Xrotation
                                End Site
                                {
                                    OFFSET 0.000000 0.000000 0.000000
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}"""


scale = 100


def convert_hierarchy_to_cm(hierarchy_str):
    def scale_offset(match):
        nums = [float(x) * scale for x in match.group(1).split()]
        return f"OFFSET {' '.join(f'{x:.4f}' for x in nums)}"
    return re.sub(r'OFFSET ([\-0-9.eE ]+)', scale_offset, hierarchy_str)
HIERARCHY_STR = convert_hierarchy_to_cm(HIERARCHY_STR)


def save_bvh(file_path, hierarchy, frame_time, positions, rotations):
    with open(file_path, 'w') as f:
        f.write(hierarchy + '\n')
        f.write("MOTION\n")
        f.write(f"Frames: {len(positions)}\n")
        f.write(f"Frame Time: {frame_time:.6f}\n")
        for pos, rot in zip(positions, rotations):
            # values = [*pos]
            values = [p * scale for p in pos]  
            for r in rot:
                values.extend(np.degrees(r[::-1]))  
            f.write(' '.join(f'{v:.6f}' for v in values) + '\n')


def quat_to_euler(q):
    R = quat2mat([q[3], q[0], q[1], q[2]])  

    return mat2euler(R, axes='rxyz')  


def combine_quats(q1, q2):
    q1 = np.array([q1[3], q1[0], q1[1], q1[2]])
    q2 = np.array([q2[3], q2[0], q2[1], q2[2]])
    q_combined = transforms3d.quaternions.qmult(q2, q1)
    return [q_combined[1], q_combined[2], q_combined[3], q_combined[0]]


def load_pkl(path):
    for loader in [joblib.load, lambda p: dill.load(open(p, 'rb'))]:
        try:
            return loader(path)
        except:
            pass
    raise ValueError("Unsupported or corrupted pickle format.")


def export_bvh_from_pkl(pkl_path, out_bvh_path):
    data = load_pkl(pkl_path)
    sample = data['0_0']
    body_quat = sample['body_quat'].numpy()
    trans = sample['trans'].numpy()
    fps = sample['fps']
    frame_time = 1.0 / fps

    positions = []
    rotations = []

    for frame_idx in range(body_quat.shape[0]):
        frame_trans = trans[frame_idx]  # root 위치
        positions.append([frame_trans[0], frame_trans[1], frame_trans[2]])

        joints = body_quat[frame_idx]
        new_joint_rots = []

        for i in range(24):
            if i == 18: continue  # skip, will be merged into 17
            if i == 23: continue  # skip, will be merged into 22
            # if i == 17:
            #     q = combine_quats(joints[17], joints[18])
            # elif i == 22:
            #     q = combine_quats(joints[22], joints[23])
            else:
                q = joints[i]


            euler = quat_to_euler(q)
            new_joint_rots.append(euler)

        rotations.append(new_joint_rots)

    save_bvh(out_bvh_path, HIERARCHY_STR, frame_time, positions, rotations)


export_bvh_from_pkl("CLoSD_t2m_finetune_1env_sit.pkl", "closdtest_norm_xyz_sit.bvh")
print("closdtest_norm_xyz_sit.bvh!!!")

