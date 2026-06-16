import csv
import time
import sys
import select
import termios
import tty
import threading
from datetime import datetime
from pathlib import Path

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import numpy as np

G1_NUM_MOTOR = 29

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

# 原始版本
Kp = [
    60, 60, 60, 100, 40, 40,      # legs
    60, 60, 60, 100, 40, 40,      # legs
    60, 40, 40,                   # waist
    40, 40, 40, 40,  40, 40, 40,  # arms
    40, 40, 40, 40,  40, 40, 40   # arms
]


Kd = [
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1, 2, 1, 1,     # legs
    1, 1, 1,              # waist
    1, 1, 1, 1, 1, 1, 1,  # arms
    1, 1, 1, 1, 1, 1, 1   # arms 
]



Kp = [i*10 for i in Kp]
Kd = [i*10 for i in Kd]


#  腿部进入阻尼模式
Kp[0:12] = [
    0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0,
]

# # 关闭所有关节的控制，调试代码阶段请不要注释掉下面两行，否则可能会有危险
# Kp = [i*0 for i in Kp]  # disable all arms
# Kd = [i*0 for i in Kd]  # disable all arms

# loop_delay = 0.1
loop_delay = 0.05

# 预备姿态角度（deg）
angles_ready_deg = [1.7, -8.1, -2.0, 29.7, -45.4, 41.2, 0.6, -1.8, 17.2, 41.2, 58.2, -58.6, -1.3, -4.7, -3.4, 24.1, -0.6, 0.3, -28.5, -4.4, -4.1, -12.8, 7.8, 3.0, -4.3, 75.0, -1.3, 12.8, -14.0]
# 出拳姿态角度（deg）
angles_fist_deg = [4.1, 8.8, 42.1, 37.8, -36.4, -6.5, -5.3, -6.8, -4.8, 45.1, -42.1, 14.3, -0.0, -0.4, -2.4, -68.4, 12.3, 25.2, 52.7, 0.1, 18.8, 10.1, 7.8, -0.4, -4.2, 76.2, -1.4, 12.8, -14.0]

# 预备

posture_angles_deg = [
    angles_ready_deg,  # 预备
    angles_fist_deg,   # 出拳
]

posture_angles_rad = []
for angles_deg in posture_angles_deg:
    angles_rad = [x * np.pi / 180.0 for x in angles_deg]
    posture_angles_rad.append(angles_rad)



class G1JointIndex:
    LeftHipPitch = 0
    LeftHipRoll = 1
    LeftHipYaw = 2
    LeftKnee = 3
    LeftAnklePitch = 4
    LeftAnkleB = 4
    LeftAnkleRoll = 5
    LeftAnkleA = 5
    RightHipPitch = 6
    RightHipRoll = 7
    RightHipYaw = 8
    RightKnee = 9
    RightAnklePitch = 10
    RightAnkleB = 10
    RightAnkleRoll = 11
    RightAnkleA = 11
    WaistYaw = 12
    WaistRoll = 13        # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistA = 13           # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistPitch = 14       # NOTE: INVALID for g1 23dof/29dof with waist locked
    WaistB = 14           # NOTE: INVALID for g1 23dof/29dof with waist locked
    LeftShoulderPitch = 15
    LeftShoulderRoll = 16
    LeftShoulderYaw = 17
    LeftElbow = 18
    LeftWristRoll = 19
    LeftWristPitch = 20   # NOTE: INVALID for g1 23dof
    LeftWristYaw = 21     # NOTE: INVALID for g1 23dof
    RightShoulderPitch = 22
    RightShoulderRoll = 23
    RightShoulderYaw = 24
    RightElbow = 25
    RightWristRoll = 26
    RightWristPitch = 27  # NOTE: INVALID for g1 23dof
    RightWristYaw = 28    # NOTE: INVALID for g1 23dof


class Mode:
    PR = 0  # Series Control for Pitch/Roll Joints
    AB = 1  # Parallel Control for A/B Joints

class Custom:
    def __init__(self):
        self.time_ = 0.0
        self.control_dt_ = 0.002  # [2ms]
        self.duration_ = 3.0    # [3 s]
        self.counter_ = 0
        self.mode_pr_ = Mode.PR
        self.mode_machine_ = 0
        self.low_cmd = unitree_hg_msg_dds__LowCmd_()  
        self.low_state = None 
        self.update_mode_machine_ = False
        self.crc = CRC()
        self.joint_angles_init = None
        self.posture_mode = None
        self.posture_mode_lock = threading.Lock()
        self.csv_file_ = None
        self.csv_writer_ = None
        self.csv_lock_ = threading.Lock()
        self.csv_path_ = None
        self.start_time_ = None

    def SetPostureMode(self, mode):
        with self.posture_mode_lock:
            self.posture_mode = mode

    def Init(self):
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        # create publisher #
        self.lowcmd_publisher_ = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.lowcmd_publisher_.Init()

        self._InitCsvLog()

        # create subscriber #
        self.lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.lowstate_subscriber.Init(self.LowStateHandler, 10)

    def _InitCsvLog(self):
        log_dir = Path(__file__).resolve().parent.parent / "output"
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path_ = log_dir / f"g1_joint_data_{loop_delay}_{ts}.csv"
        self.csv_file_ = open(self.csv_path_, "w", newline="")
        header = ["timestamp_s", "posture_mode"]
        for name in JOINT_NAMES:
            header.extend([f"{name}_dq", f"{name}_tau"])
        self.csv_writer_ = csv.writer(self.csv_file_)
        self.csv_writer_.writerow(header)
        self.csv_file_.flush()
        self.start_time_ = time.time()
        print(f"Logging joint velocity/torque to: {self.csv_path_}")

    def _WriteCsvRow(self, msg: LowState_):
        if self.csv_writer_ is None or self.start_time_ is None:
            return
        row = [time.time() - self.start_time_]
        with self.posture_mode_lock:
            row.append(
                self.posture_mode if self.posture_mode is not None else -1
            )
        for i in range(G1_NUM_MOTOR):
            ms = msg.motor_state[i]
            row.extend([ms.dq, ms.tau_est])
        with self.csv_lock_:
            self.csv_writer_.writerow(row)
            self.csv_file_.flush()

    def CloseCsv(self):
        if self.csv_file_ is not None:
            self.csv_file_.close()
            self.csv_file_ = None
            print(f"CSV log saved: {self.csv_path_}")

    def Start(self):
        self.lowCmdWriteThreadPtr = RecurrentThread(
            interval=self.control_dt_, target=self.LowCmdWrite, name="control"
        )
        while self.update_mode_machine_ == False:
            time.sleep(1)

        if self.update_mode_machine_ == True:
            self.lowCmdWriteThreadPtr.Start()

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg
        self._WriteCsvRow(msg)

        if self.update_mode_machine_ == False:
            self.mode_machine_ = self.low_state.mode_machine
            self.update_mode_machine_ = True
        
        self.counter_ +=1
        if (self.counter_ % 500 == 0) :
            self.counter_ = 0
            joint_angles = np.array(
                [self.low_state.motor_state[i].q for i in range(G1_NUM_MOTOR)]
            )
            # if self.joint_angles_init is None:
                # self.joint_angles_init = joint_angles
            # joint_angles_diff = joint_angles - self.joint_angles_init
            # joint_angles_diff_deg = joint_angles_diff * 180.0 / np.pi
            # formatted = [f"{x:.1f}" for x in joint_angles_diff_deg]

            joint_angles_deg = joint_angles * 180.0 / np.pi
            formatted = [f"{x:.1f}" for x in joint_angles_deg]
            print(f"Joint angles delta (deg): {formatted}")

    def LowCmdWrite(self):
        self.time_ += self.control_dt_

        if self.time_ < self.duration_ :
            # [Stage 1]: set robot to zero posture
            print(f"Moving to zero posture...")
            for i in range(G1_NUM_MOTOR):
                ratio = np.clip(self.time_ / self.duration_, 0.0, 1.0)
                self.low_cmd.mode_pr = Mode.PR
                self.low_cmd.mode_machine = self.mode_machine_
                self.low_cmd.motor_cmd[i].mode =  1 # 1:Enable, 0:Disable
                self.low_cmd.motor_cmd[i].tau = 0. 
                self.low_cmd.motor_cmd[i].q = (1.0 - ratio) * self.low_state.motor_state[i].q 
                self.low_cmd.motor_cmd[i].dq = 0. 
                self.low_cmd.motor_cmd[i].kp = Kp[i] 
                self.low_cmd.motor_cmd[i].kd = Kd[i]

        else:
            if self.posture_mode is None:
                return

            # 持续循环出拳（mode=11 是特殊模式，不是 posture_angles_rad 的下标）
            if self.posture_mode == 11:
                while True:
                    with self.posture_mode_lock:
                        if self.posture_mode != 11:
                            break
                    for model in [0, 1]:
                        angles = posture_angles_rad[model]
                        for i in range(G1_NUM_MOTOR):
                            self.low_cmd.motor_cmd[i].q = angles[i]
                        time.sleep(loop_delay)
                        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
                        self.lowcmd_publisher_.Write(self.low_cmd)
                return

            with self.posture_mode_lock:
                angles = posture_angles_rad[self.posture_mode]
                for i in range(G1_NUM_MOTOR):
                    self.low_cmd.motor_cmd[i].q = angles[i]

        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher_.Write(self.low_cmd)


def _read_key_nonblocking():
    if select.select([sys.stdin], [], [], 0)[0]:
        return sys.stdin.read(1)
    return None


if __name__ == '__main__':

    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv)>1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(0)

    custom = Custom()
    custom.Init()
    custom.Start()

    print("Keyboard: 'j'=ready posture, 'k'=fist posture, 'l'=continuous fist, 'o'=quit")
    old_tty = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())
    try:
        while True:
            key = _read_key_nonblocking()
            if key == 'o':
                print("Quit requested.")
                break
            elif key == 'j':
                custom.SetPostureMode(0)
                print("posture_mode=0 (ready)")
            elif key == 'k':
                custom.SetPostureMode(1)
                print("posture_mode=1 (fist)")
            elif key == 'l':
                custom.SetPostureMode(11)
                print("posture_mode=11 (continuous fist)")
            time.sleep(0.05)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_tty)
        custom.CloseCsv()