from filterpy.kalman import ExtendedKalmanFilter
import numpy as np
from enum import Enum
import math

T_ms = 0.0383


class MOTION_TYPE(Enum):
    STATIONARY = 1
    SMOOTH = 2
    NONLINEAR = 3
    RANDOM = 4
    TILTED = 5


# Constants
x_o = 30  # [cm]
x_f = 70
t_f = 10  # [sec]
motion = MOTION_TYPE.SMOOTH


def __gaussian_noise(mean=0, std_dev=0, num_samples=0):
    return np.random.normal(0, std_dev, num_samples)


# Smooth point-to-point motion simulator
def __smooth_p2p_motion(time: float):
    pos = (((x_f - x_o) / t_f) * time) + x_o + __gaussian_noise()
    return pos


def __nonlin_motion(time: float, amp: float, f: float):
    pos = __smooth_p2p_motion(time) + amp * math.sin(2 * math.pi * f * time / t_f)
    return pos


def __random_motion(time: float, n: int):
    As = np.random.rand(n)
    Bs = np.random.rand(n)
    pos = x_o
    for k in range(n):
        pos += As[k] * math.cos(2 * math.pi * k * time / t_f) + Bs[k] * math.sin(2 * math.pi * k * time / t_f)
    pos += __gaussian_noise()
    return pos


# Function to simulate actual position
def actual_pos(time: float):
    match motion:
        case MOTION_TYPE.SMOOTH:
            return __smooth_p2p_motion(time)
        case MOTION_TYPE.NONLINEAR:
            amp = 0.5
            f = 5
            return __nonlin_motion(time, amp, f)
        case MOTION_TYPE.RANDOM:
            n = 30
            return __random_motion(time, n)
        case _:
            return __smooth_p2p_motion(time)


# State transition matrix
def A() -> np.ndarray:
    return np.array([[1, T_ms, 0.5 * T_ms**2], [0, 1, T_ms], [0, 0, 1]])


# Measurement functions
def sensor_medium_d2v(d):
    0.244503113316274 * d - 1.842630707316104 + 30.377535557092287 / d


def sensor_long_d2v(d):
    -7.942430588138158 * d + 21.854787336053960 + 47.264760774084991 / d


def main():
    t = 0
    while t < t_f:
        t += T_ms


if __name__ == "__main__":
    main()
