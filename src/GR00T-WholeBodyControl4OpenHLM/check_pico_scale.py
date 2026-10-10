#!/usr/bin/env python3
"""PICO 身体位姿体检脚本 —— 定位 OpenHLM 遥操作机器人形态异常。

在 PC 工作站上、.venv_teleop 环境里运行：

    source .venv_teleop/bin/activate
    python check_pico_scale.py

它会：
  1. 等待 PICO 身体追踪就绪，采样若干帧
  2. 打印 24 个身体节点在 Unity 原始帧 / 机器人帧下的位置
  3. 检查关键量：骨盆高度、头顶高度、脚高度、左右对称性
  4. 给出单位 / 尺度是否与 GMR 假设一致的判断

只读，不发送任何控制指令，可以放心在真机旁边跑。
"""

import argparse
import time

import numpy as np

try:
    import xrobotoolkit_sdk as xrt
except ImportError:
    raise SystemExit(
        "找不到 xrobotoolkit_sdk。请确认已 source .venv_teleop/bin/activate，"
        "并且 XRoboToolkit SDK 已安装在当前环境。"
    )

# 与 GMR general_motion_retargeting/xrobot_utils.py 定义的 24 点布局完全一致
BODY_NAMES = [
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1",
    "Left_Knee", "Right_Knee", "Spine2",
    "Left_Ankle", "Right_Ankle", "Spine3",
    "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head",
    "Left_Shoulder", "Right_Shoulder",
    "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist",
    "Left_Hand", "Right_Hand",
]

# OpenHLM 只用这 14 个（pico_thread.py: PICO_XROBOT_INDEX）
USED_BY_OPENHLM = {
    "Pelvis", "Left_Hip", "Right_Hip", "Left_Knee", "Right_Knee",
    "Spine3", "Left_Foot", "Right_Foot",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist",
}

# 与 pico_thread.py:_unity_to_xrobot_pose 完全相同的坐标变换
ROT_MATRIX = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=np.float64
)

# GMR 的假设值（来自 ik_configs/xrobot_to_g1.json）
HUMAN_HEIGHT_ASSUMPTION = 1.8
OPENHLM_HARDCODED_HEIGHT = 1.6   # pico_thread.py:117
SCALE_LEGS = 0.9                 # human_scale_table 里躯干/髋/膝/足
SCALE_ARMS = 0.8                 # human_scale_table 里肩/肘/腕


def unity_to_robot(pos_unity):
    """Unity 帧 -> 机器人帧（只做轴向旋转，不含单位换算，与 OpenHLM 实现一致）。"""
    return np.asarray(pos_unity, dtype=np.float64) @ ROT_MATRIX.T


def gmr_expected_height(actual_height):
    return actual_height / HUMAN_HEIGHT_ASSUMPTION


def collect(samples, interval):
    poses = []
    for _ in range(samples):
        if not xrt.is_body_data_available():
            time.sleep(interval)
            continue
        raw = xrt.get_body_joints_pose()
        if raw is None:
            time.sleep(interval)
            continue
        arr = np.asarray(raw, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] < len(BODY_NAMES):
            time.sleep(interval)
            continue
        poses.append(arr)
        time.sleep(interval)
    return poses


def report(poses, my_height):
    if not poses:
        raise SystemExit("没采到任何有效的身体位姿，检查 XRoboToolkit 连接与 Full body 模式。")

    stack = np.stack(poses, axis=0)                    # (T, N, 7)
    mean = stack.mean(axis=0)                          # (N, 7)
    spread = stack.std(axis=0)[:, :3].max()            # 抖动幅度

    print("=" * 72)
    print(f"采样 {len(poses)} 帧，位置最大抖动 {spread * 1000:.2f} mm")
    if spread > 0.05:
        print("[警告] 抖动超过 5 cm，追踪不稳定，请先解决 tracker 视线/贴身衣裤问题")
    print("=" * 72)

    print("\n--- 24 个身体节点位置（机器人帧，单位沿用上游） ---")
    print(f"{'name':<16}{'x':>10}{'y':>10}{'z':>10}   {'OpenHLM 使用':<12}")
    for i, name in enumerate(BODY_NAMES):
        p = unity_to_robot(mean[i, :3])
        flag = "  <-- 用" if name in USED_BY_OPENHLM else ""
        print(f"{name:<16}{p[0]:>10.4f}{p[1]:>10.4f}{p[2]:>10.4f}   "
              f"{'YES' if name in USED_BY_OPENHLM else '-':<12}{flag}")

    def pos(name):
        return unity_to_robot(mean[BODY_NAMES.index(name), :3])

    pelvis = pos("Pelvis")
    head = pos("Head")
    foot_l = pos("Left_Foot")
    foot_r = pos("Right_Foot")

    print("\n--- 关键度量（按 GMR 的假设，机器人帧 z 应为「向上」） ---")
    print(f"骨盆高度 (z)        : {pelvis[2]:.4f}")
    print(f"头顶高度 (z)        : {head[2]:.4f}")
    print(f"左脚高度 (z)        : {foot_l[2]:.4f}")
    print(f"右脚高度 (z)        : {foot_r[2]:.4f}")
    print(f"身高估计 (头-脚 z 差): {head[2] - min(foot_l[2], foot_r[2]):.4f}")

    print("\n--- 单位判断 ---")
    est = head[2] - min(foot_l[2], foot_r[2])
    if 1.2 < est < 2.2:
        print("[OK] 量级像「米」，与 GMR 的 human_height_assumption=1.8 假设吻合")
    elif 120 < est < 220:
        print("[问题] 量级像「厘米」。GMR 不会做 /100 换算，必须在上游或")
        print("       pico_thread.py 的 _unity_to_xrobot_pose 里补上 * 0.01")
    elif est < 0.5:
        print("[问题] 量级过小，怀疑是归一化坐标或被除以了常数")
    else:
        print("[存疑] 量级不属于常见情形，请人工核对上面 24 点的数值")

    print("\n--- 左右对称性（用于确认坐标变换没搞反） ---")
    for a, b in [
        ("Left_Hip", "Right_Hip"),
        ("Left_Knee", "Right_Knee"),
        ("Left_Foot", "Right_Foot"),
        ("Left_Shoulder", "Right_Shoulder"),
        ("Left_Elbow", "Right_Elbow"),
        ("Left_Wrist", "Right_Wrist"),
    ]:
        d = pos(a) - pos(b)
        # 左右应在同一个水平轴上反号，且第三轴分量接近于 0
        print(f"{a:<16} - {b:<16} = ({d[0]:+.4f}, {d[1]:+.4f}, {d[2]:+.4f})")
    print("正常表现：某一水平轴分量明显非零且左右反号，上下轴(z)分量接近 0。")
    print("若 z 分量很大，说明左右被映射到了垂直方向，坐标变换有问题。")

    print("\n--- 缩放影响（OpenHLM 当前硬编码 vs 你的真实身高） ---")
    ratio_now = gmr_expected_height(OPENHLM_HARDCODED_HEIGHT)
    ratio_me = gmr_expected_height(my_height)
    print(f"OpenHLM 硬编码 1.6 m  -> ratio = 1.6/1.8 = {ratio_now:.4f}")
    print(f"你的真实身高 {my_height:.2f} m -> ratio = {my_height:.2f}/1.8 = {ratio_me:.4f}")
    print(f"  腿/躯干最终缩放：现在 {SCALE_LEGS * ratio_now:.3f}  ->  应为 {SCALE_LEGS * ratio_me:.3f}")
    print(f"  手臂最终缩放    ：现在 {SCALE_ARMS * ratio_now:.3f}  ->  应为 {SCALE_ARMS * ratio_me:.3f}")
    shrink_pct = (1.0 - ratio_now / ratio_me) * 100.0
    if abs(shrink_pct) > 3.0:
        print(f"[行动] 当前目标被额外压缩了约 {shrink_pct:.1f}%，"
              f"把 pico_thread.py:117 的 1.6 改成 {my_height:.2f}")
    else:
        print("[OK] 身高差异在 3% 以内，不是主要原因")

    print("\n--- 逐节点相对骨盆的距离（直接决定 IK 目标有多大） ---")
    print(f"{'name':<16}{'now(m)':>10}{'corrected(m)':>14}")
    for name in ["Left_Foot", "Right_Foot", "Left_Knee", "Right_Knee",
                 "Left_Wrist", "Right_Wrist", "Left_Elbow", "Right_Elbow"]:
        v = pos(name) - pelvis
        d_now = np.linalg.norm(v * (SCALE_LEGS if "Foot" in name or "Knee" in name else SCALE_ARMS) * ratio_now)
        d_ok = np.linalg.norm(v * (SCALE_LEGS if "Foot" in name or "Knee" in name else SCALE_ARMS) * ratio_me)
        print(f"{name:<16}{d_now:>10.4f}{d_ok:>14.4f}")

    print("\n提示：G1 的腿长约 0.62 m、手臂约 0.55 m（含手）。")
    print("      若上面 corrected 列仍明显小于这个量级，说明问题不只在身高，")
    print("      需要回到 offset_to_ground 与 QuatProcessor 标定上继续查。")


def main():
    parser = argparse.ArgumentParser(description="PICO 身体位姿体检")
    parser.add_argument("--samples", type=int, default=60, help="采样帧数")
    parser.add_argument("--interval", type=float, default=0.05, help="采样间隔秒")
    parser.add_argument("--my-height", type=float, required=True, help="你的真实身高（米）")
    parser.add_argument("--no-service", action="store_true",
                        help="跳过启动 roboticsservice（若已在运行）")
    args = parser.parse_args()

    if not args.no_service:
        import subprocess
        subprocess.Popen(["bash", "/opt/apps/roboticsservice/runService.sh"])
        time.sleep(1.0)

    xrt.init()
    print("等待 PICO 身体追踪数据……")
    waited = 0.0
    while not xrt.is_body_data_available():
        time.sleep(1.0)
        waited += 1.0
        if waited > 30:
            raise SystemExit("30 秒内没等到身体数据。检查 PICO 是否连上、"
                             "XRoboToolkit 是否勾选 Full body、Status 是否 WORKING。")
        print("  waiting for body data ...")

    print("身体追踪就绪，开始采样。请保持站立不动。\n")
    poses = collect(args.samples, args.interval)
    report(poses, args.my_height)


if __name__ == "__main__":
    main()
