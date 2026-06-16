#!/usr/bin/env python3
"""从 output 中的 g1_joint_data CSV 提取各关节峰值力矩与峰值速度。"""

import argparse
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output"
DATA_GLOB = "g1_joint_data_*.csv"

JOINT_NAMES = [
    "LeftHipPitch", "LeftHipRoll", "LeftHipYaw", "LeftKnee",
    "LeftAnklePitch", "LeftAnkleRoll",
    "RightHipPitch", "RightHipRoll", "RightHipYaw", "RightKnee",
    "RightAnklePitch", "RightAnkleRoll",
    "WaistYaw", "WaistRoll", "WaistPitch",
    "LeftShoulderPitch", "LeftShoulderRoll", "LeftShoulderYaw",
    "LeftElbow", "LeftWristRoll", "LeftWristPitch", "LeftWristYaw",
    "RightShoulderPitch", "RightShoulderRoll", "RightShoulderYaw",
    "RightElbow", "RightWristRoll", "RightWristPitch", "RightWristYaw",
]

JOINT_LABELS = [
    "left_hip_pitch_joint(左髋俯仰)",
    "left_hip_roll_joint(左髋侧摆)",
    "left_hip_yaw_joint(左髋偏航)",
    "left_knee_joint(左膝)",
    "left_ankle_pitch_joint(左踝俯仰)",
    "left_ankle_roll_joint(左踝侧摆)",
    "right_hip_pitch_joint(右髋俯仰)",
    "right_hip_roll_joint(右髋侧摆)",
    "right_hip_yaw_joint(右髋偏航)",
    "right_knee_joint(右膝)",
    "right_ankle_pitch_joint(右踝俯仰)",
    "right_ankle_roll_joint(右踝侧摆)",
    "waist_yaw_joint(腰部偏航)",
    "waist_roll_joint(腰部侧摆)",
    "waist_pitch_joint(腰部俯仰)",
    "left_shoulder_pitch_joint(左肩俯仰)",
    "left_shoulder_roll_joint(左肩侧摆)",
    "left_shoulder_yaw_joint(左肩偏航)",
    "left_elbow_joint(左肘)",
    "left_wrist_roll_joint(左腕滚转)",
    "left_wrist_pitch_joint(左腕俯仰)",
    "left_wrist_yaw_joint(左腕偏航)",
    "right_shoulder_pitch_joint(右肩俯仰)",
    "right_shoulder_roll_joint(右肩侧摆)",
    "right_shoulder_yaw_joint(右肩偏航)",
    "right_elbow_joint(右肘)",
    "right_wrist_roll_joint(右腕滚转)",
    "right_wrist_pitch_joint(右腕俯仰)",
    "right_wrist_yaw_joint(右腕偏航)",
]

CSV_HEADER = [
    "关节",
    "峰值力矩(N·m)",
    "峰值力矩时刻(s)",
    "峰值力矩时速度(rad/s)",
    "峰值速度(rad/s)",
    "峰值速度时刻(s)",
    "峰值速度时力矩(N·m)",
]


def _peak_index(values):
    return max(range(len(values)), key=lambda i: abs(values[i]))


def _output_path(input_path: Path) -> Path:
    name = input_path.name.replace("g1_joint_data", "joint_peaks", 1)
    return input_path.parent / name


def analyze_file(input_path: Path, output_path: Path | None = None) -> Path:
    with open(input_path, newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        raise ValueError(f"No data in {input_path}")

    times = [float(row["timestamp_s"]) for row in rows]
    result_rows = []

    for name, label in zip(JOINT_NAMES, JOINT_LABELS):
        dqs = [float(row[f"{name}_dq"]) for row in rows]
        taus = [float(row[f"{name}_tau"]) for row in rows]

        tau_idx = _peak_index(taus)
        dq_idx = _peak_index(dqs)

        result_rows.append([
            label,
            round(taus[tau_idx], 3),
            round(times[tau_idx], 3),
            round(dqs[tau_idx], 3),
            round(dqs[dq_idx], 3),
            round(times[dq_idx], 3),
            round(taus[dq_idx], 3),
        ])

    out = output_path or _output_path(input_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(result_rows)

    print(f"Input:  {input_path}")
    print(f"Output: {out}")
    return out


def _collect_inputs(paths: list[str]) -> list[Path]:
    if paths:
        return [Path(p).resolve() for p in paths]

    files = sorted(OUTPUT_DIR.glob(DATA_GLOB), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(
            f"No matching files in {OUTPUT_DIR} (pattern: {DATA_GLOB})"
        )
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Analyze g1 joint data CSV and export peak torque/velocity."
    )
    parser.add_argument(
        "input",
        nargs="*",
        help="Input CSV path(s). Default: all g1_joint_data_*.csv in output/",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output CSV path (only valid with a single input file)",
    )
    args = parser.parse_args()

    inputs = _collect_inputs(args.input)
    if args.output and len(inputs) != 1:
        parser.error("--output requires exactly one input file")

    for input_path in inputs:
        analyze_file(input_path, Path(args.output) if args.output else None)


if __name__ == "__main__":
    main()
