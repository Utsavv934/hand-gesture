"""
ECHO HANDS — Geometric Sign Recognition Prototype & Quick Test Runner
"""

import numpy as np
from recognizer import recognize_sign, compute_hand_geometry, TrajectoryTracker


def build_synthetic_hand(handedness="Right"):
    pts = np.zeros((21, 3), dtype=np.float32)
    pts[0] = [0.0, 0.0, 0.0]  # wrist
    # Thumb: 1, 2, 3, 4
    pts[1] = [-0.15, 0.25, 0.0]
    pts[2] = [-0.30, 0.45, 0.0]
    pts[3] = [-0.35, 0.65, 0.0]
    pts[4] = [-0.25, 0.80, 0.0]
    # MCPs
    pts[5] = [-0.25, 0.95, 0.0]
    pts[9] = [0.0, 1.0, 0.0]
    pts[13] = [0.22, 0.95, 0.0]
    pts[17] = [0.40, 0.85, 0.0]
    # Straight fingers by default
    for mcp_idx, x_off in [(5, -0.25), (9, 0.0), (13, 0.22), (17, 0.40)]:
        y = pts[mcp_idx][1]
        pts[mcp_idx + 1] = [x_off, y + 0.35, 0.0]
        pts[mcp_idx + 2] = [x_off, y + 0.55, 0.0]
        pts[mcp_idx + 3] = [x_off, y + 0.70, 0.0]
    return pts


def set_finger_curl(pts, f_idx, curl=True):
    base_y = pts[f_idx][1]
    x_off = pts[f_idx][0]
    if curl:
        pts[f_idx + 1] = [x_off, base_y + 0.20, 0.0]
        pts[f_idx + 2] = [x_off, base_y + 0.10, 0.0]
        pts[f_idx + 3] = [x_off, base_y - 0.10, 0.0]
    else:
        pts[f_idx + 1] = [x_off, base_y + 0.35, 0.0]
        pts[f_idx + 2] = [x_off, base_y + 0.55, 0.0]
        pts[f_idx + 3] = [x_off, base_y + 0.70, 0.0]


if __name__ == "__main__":
    print("=" * 65)
    print("ECHO HANDS — Geometric Sign Recognition Verification")
    print("=" * 65)

    # Sign A
    pts_A = build_synthetic_hand("Right")
    for f in [5, 9, 13, 17]:
        set_finger_curl(pts_A, f, curl=True)
    pts_A[4] = [-0.35, 1.15, 0.0]
    letter_A, conf_A, reason_A, _ = recognize_sign(pts_A, tracker=TrajectoryTracker())
    print(f"Result A -> Letter: '{letter_A}', Confidence: {conf_A:.2f}, Reason: {reason_A}")

    # Sign C (Open Arch)
    pts_C = build_synthetic_hand("Right")
    pts_C[4] = [-0.25, 0.80, 0.0]
    for f, arch_y, tip_drop in [(5, 0.30, 0.0), (9, 0.30, 0.0), (13, 0.25, -0.05), (17, 0.20, -0.10)]:
        base_y = pts_C[f][1]
        x_off = pts_C[f][0]
        pts_C[f + 1] = [x_off, base_y + arch_y, 0.0]
        pts_C[f + 2] = [x_off + 0.05, base_y + arch_y + 0.10, 0.0]
        pts_C[f + 3] = [x_off + 0.10, base_y + arch_y + tip_drop, 0.0]
    letter_C, conf_C, reason_C, _ = recognize_sign(pts_C, tracker=TrajectoryTracker())
    print(f"Result C -> Letter: '{letter_C}', Confidence: {conf_C:.2f}, Reason: {reason_C}")

    # Sign O (Closed Loop)
    pts_O = build_synthetic_hand("Right")
    pts_O[4] = [-0.10, 0.82, 0.0]
    pts_O[6] = [-0.25, 1.20, 0.0]; pts_O[7] = [-0.18, 1.05, 0.0]; pts_O[8] = [-0.10, 0.85, 0.0]
    pts_O[10] = [0.0, 1.25, 0.0];  pts_O[11] = [-0.05, 1.05, 0.0]; pts_O[12] = [-0.08, 0.87, 0.0]
    pts_O[14] = [0.18, 1.15, 0.0]; pts_O[15] = [0.12, 1.00, 0.0]; pts_O[16] = [0.05, 0.88, 0.0]
    pts_O[18] = [0.32, 1.05, 0.0]; pts_O[19] = [0.25, 0.95, 0.0]; pts_O[20] = [0.15, 0.86, 0.0]
    letter_O, conf_O, reason_O, _ = recognize_sign(pts_O, tracker=TrajectoryTracker())
    print(f"Result O -> Letter: '{letter_O}', Confidence: {conf_O:.2f}, Reason: {reason_O}")

    # Sign B (4 fingers straight)
    pts_B = build_synthetic_hand("Right")
    pts_B[4] = [-0.10, 0.50, 0.0]
    letter_B, conf_B, reason_B, _ = recognize_sign(pts_B, tracker=TrajectoryTracker())
    print(f"Result B -> Letter: '{letter_B}', Confidence: {conf_B:.2f}, Reason: {reason_B}")

    print("=" * 65)
    print("All tests executed successfully.")
