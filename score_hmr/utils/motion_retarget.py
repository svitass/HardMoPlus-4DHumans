target_body_joints = ["pelvis",  # 0
                      "left_hip",  # 1
                      "right_hip",  # 2
                      "left_knee",  # 3
                      "right_knee",  # 4
                      "left_ankle",  # 5
                      "right_ankle",  # 6
                      "neck",  # 7
                      "head",  # 8
                      "left_shoulder",  # 9
                      "right_shoulder",  # 10
                      "left_elbow",  # 11
                      "right_elbow",  # 12
                      "left_wrist",  # 13
                      "right_wrist",  # 14
                      "nose",  # 15
                      "right_eye",  # 16
                      "left_eye",  # 17
                      "right_ear",  # 18
                      "left_ear",  # 19
                      "left_big_toe",  # 20
                      "left_small_toe",  # 21
                      "left_heel",  # 22
                      "right_big_toe",  # 23
                      "right_small_toe",  # 24
                      "right_heel",  # 25
                      "left_thumb",  # 26
                      "left_index",  # 27
                      "left_middle",  # 28
                      "left_ring",  # 29
                      "left_pinky",  # 30
                      "right_thumb",  # 31
                      "right_index",  # 32
                      "right_middle",  # 33
                      "right_ring",  # 34
                      "right_pinky"  # 35
                      ]
target_kps_names = [
    "pelvis", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "neck", "head",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_index1",
    "left_index2", "left_index3", "left_middle1", "left_middle2", "left_middle3", "left_pinky1", "left_pinky2",
    "left_pinky3", "left_ring1", "left_ring2", "left_ring3", "left_thumb1", "left_thumb2", "left_thumb3",
    "right_index1", "right_index2", "right_index3", "right_middle1", "right_middle2", "right_middle3", "right_pinky1",
    "right_pinky2", "right_pinky3", "right_ring1", "right_ring2", "right_ring3", "right_thumb1", "right_thumb2", "right_thumb3",
    "nose", "right_eye", "left_eye", "right_ear", "left_ear", "left_big_toe", "left_small_toe", "left_heel",
    "right_big_toe", "right_small_toe", "right_heel", "left_thumb", "left_index", "left_middle",
    "left_ring", "left_pinky", "right_thumb", "right_index", "right_middle", "right_ring", "right_pinky"
]
target_idx = [target_kps_names.index(joint_name) for joint_name in target_body_joints]
