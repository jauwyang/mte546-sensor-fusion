from filterpy.kalman import ExtendedKalmanFilter
import matplotlib.pyplot as plt
import numpy as np
from enum import Enum
import math

# cd Desktop\Projects\Programs\uw-courses\mte546-sensor-fusion\
# python labs\lab3\main.py


class MOTION_TYPE(Enum):
    STATIONARY = 1
    SMOOTH = 2
    NONLINEAR = 3
    RANDOM = 4
    TILTED = 5


# Constants
NUM_STATES = 3
NUM_MEASUREMENT_IN = 2

T_ms = 0.0383
x_o = 30  # [cm]
x_f = 70
t_f = 10  # [sec]
motion = MOTION_TYPE.SMOOTH

# 1.0e07
Q_mat_sim = 1.0e07 * np.array(
    [
        [0.000000292810336, 0.000005331742902, 0.000103885699918],
        [0.000005331742902, 0.000970437150569, 0.028065056461569],
        [0.000103885699918, 0.028065056461569, 1.462534098876549],
    ]
)
# Q_mat = 0.000001 * np.eye(3)

# Q_mat = 1.0e06 * np.array(
#     [
#         [0.000237653309536, 0, 0],
#         [0, 1.0e-06, 0],
#         [0, 0, 1.0e-06],
#     ]
# )

Q_mat = np.array(
    [
        [1.0e06 * 0.000237653309536, 0, 0],
        [0, 1, 0],
        [0, 0, 0.5],
    ]
)

R_mat = np.array([[0.000161625, 0], [0, 0.0011362222]])
# R_mat = np.eye(2)


R1_mat = np.array([[0.000161625]])
R2_mat = np.array([[0.0011362222]])


## Noise Helper Functions
def __gaussian_noise(mean=0, std_dev=0, num_samples=1):
    x = np.random.normal(mean, std_dev, num_samples)
    return x


def simulated_distance_noise(
    num_samples=1,
):  # is like simulating jittering/shaking of object
    return __gaussian_noise(0, Q_mat_sim[0, 0], num_samples)


def medium_range_sensor_noise(num_samples=1):
    return __gaussian_noise(0, R1_mat[0, 0], num_samples)


def long_range_sensor_noise(num_samples=1):
    return __gaussian_noise(0, R2_mat[0, 0], num_samples)


## Simulated Object Motion
def __smooth_p2p_motion(time: float):
    pos = (((x_f - x_o) / t_f) * time) + x_o + simulated_distance_noise()
    return pos


def __nonlin_motion(time: float, amp: float, f: float):
    pos = __smooth_p2p_motion(time) + amp * math.sin(2 * math.pi * f * time / t_f)
    return pos


def __random_motion(time: float, n: int):
    As = np.random.rand(n)
    Bs = np.random.rand(n)
    pos = x_o
    for k in range(n):
        pos += As[k] * math.cos(2 * math.pi * k * time / t_f) + Bs[k] * math.sin(
            2 * math.pi * k * time / t_f
        )
    pos += simulated_distance_noise()
    return pos


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


## Sensor Model Functions
def sensor_medium_d2v(d: float, has_noise: bool = True) -> float:
    sensed_voltage = -0.000670639395051 * d + 0.093032791646544 + 26.488033250235446 / d
    if has_noise:
        sensed_voltage += medium_range_sensor_noise()
    return sensed_voltage


def sensor_long_d2v(d: float, has_noise: bool = True) -> float:
    sensed_voltage = -0.007448158665767 * d + 0.897563547427417 + 36.878840141983289 / d
    if has_noise:
        sensed_voltage += long_range_sensor_noise()
    return sensed_voltage


## EKF Functions
def sensor_medium_v2d(v: float) -> float:
    estimated_distance = (
        0.244503113316274 * v - 1.842630707316104 + 30.377535557092287 / v
    )
    return estimated_distance


def sensor_long_v2d(v: float) -> float:
    estimated_distance = (
        -7.942430588138158 * v + 21.854787336053960 + 47.264760774084991 / v
    )
    return estimated_distance


def fuse_sensors(v_med: float, v_long: float):
    med_dist = sensor_medium_v2d(v_med)
    long_dist = sensor_long_v2d(v_long)
    # TODO: Kalman Filter
    pass


## Plot graphs
def plot_states_overtime(states: np.array, true_pos: np.array, t_f: float):
    time_range = np.arange(0, t_f + T_ms, T_ms)

    labels = [
        "True Position [cm]",
        "Est. Position [cm]",
        "Est. Velocity [cm/s]",
        "Est. Acceleration [cm/s^2]",
    ]

    # Stack the true state on top of the estimated states
    states = np.vstack((true_pos, states))
    plt.figure(figsize=(8, 5))
    # for state_i in range(states.shape[0]):
    for state_i in range(4):
        plt.plot(
            time_range[: len(time_range) - 1], states[state_i, :], label=labels[state_i]
        )

    plt.xlabel("Time [s]")
    plt.ylabel("State")
    plt.legend()
    plt.grid(True)
    plt.show()


def HJacobian(x, *args):  # [2, 3] is transposed in code (becomes [3,2])
    d = x[0]
    if type(d) == np.ndarray:
        d = d[0]
    return np.array(
        [
            [0.244503113316274 - 26.488033250235446 / d**2, 0, 0],
            [-7.942430588138158 - 47.264760774084991 / d**2, 0, 0],
        ]
    )


def Hx(x, *args):
    d = x[0]

    # Call the sensor models to get the expected measurements
    z1 = sensor_medium_d2v(d, False)
    z2 = sensor_long_d2v(d, False)

    # Return the vector of expected measurements
    return np.array([z1, z2]).reshape(2, 1)


def main():
    estimated_states = []
    true_pos = []

    # Q_mat = 1.0e07 * np.array(
    #     [
    #         [0.000000292810336, 0.000005331742902, 0.000103885699918],
    #         [0.000005331742902, 0.000970437150569, 0.028065056461569],
    #         [0.000103885699918, 0.028065056461569, 1.462534098876549],
    #     ]
    # )

    ekf = ExtendedKalmanFilter(NUM_STATES, NUM_MEASUREMENT_IN)
    ekf.x = np.array([x_o, 0, 0])  # initial states to 0 except for position
    ekf.P = Q_mat  # initialization choice
    ekf.R = R_mat
    ekf.Q = Q_mat
    ekf.F = np.array([[1, T_ms, 0.5 * T_ms**2], [0, 1, T_ms], [0, 0, 1]])
    ekf.H = None

    # Simulation time
    t = 0
    while t < t_f:
        curr_pos = actual_pos(t)
        print(t)
        z_1 = sensor_medium_d2v(curr_pos)
        z_2 = sensor_long_d2v(curr_pos)

        z = np.array([z_1, z_2])  # [2, 1]
        ekf.predict_update(z, HJacobian, Hx)
        posterior_state = ekf.x  # ????? idk if this is the correct function
        print(posterior_state)

        true_pos.append(curr_pos)
        estimated_states.append(posterior_state)
        t += T_ms

    # print(true_pos.shape)
    plot_states_overtime(
        np.hstack(estimated_states), np.array(true_pos).reshape(1, -1), t_f
    )


if __name__ == "__main__":
    main()


##### EKF.py
# def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
#         """ Performs the predict/update innovation of the extended Kalman
#         filter.

#         Parameters
#         ----------

#         z : np.array
#             measurement for this step.
#             If `None`, only predict step is perfomed.

#         HJacobian : function
#            function which computes the Jacobian of the H matrix (measurement
#            function). Takes state variable (self.x) as input, along with the
#            optional arguments in args, and returns H.

#         Hx : function
#             function which takes as input the state variable (self.x) along
#             with the optional arguments in hx_args, and returns the measurement
#             that would correspond to that state.

#         args : tuple, optional, default (,)
#             arguments to be passed into HJacobian after the required state
#             variable.

#         hx_args : tuple, optional, default (,)
#             arguments to be passed into Hx after the required state
#             variable.

#         u : np.array or scalar
#             optional control vector input to the filter.
#         """
#         #pylint: disable=too-many-locals

#         if not isinstance(args, tuple):
#             args = (args,)

#         if not isinstance(hx_args, tuple):
#             hx_args = (hx_args,)

#         if np.isscalar(z) and self.dim_z == 1:
#             z = np.asarray([z], float)
#         F = self.F
#         B = self.B
#         P = self.P
#         Q = self.Q
#         R = self.R
#         x = self.x

#         H = HJacobian(x, *args)

#         # predict step
#         x = dot(F, x) + dot(B, u)
#         P = dot(F, P).dot(F.T) + Q

#         # save prior
#         self.x_prior = np.copy(self.x)
#         self.P_prior = np.copy(self.P)

#         # update step
#         PHT = dot(P, H.T)

#         # print(f"PHT: {PHT.shape}")
#         self.S = dot(H, PHT) + R
#         self.SI = linalg.inv(self.S)
#         self.K = dot(PHT, self.SI)
#         # print(f"K: {self.K.shape}")

#         self.y = z - Hx(x, *hx_args)
#         # print(f"z_val: {z}")

#         # print(f"z: {z.shape}")
#         # print(f"Hx: {Hx(x, *hx_args).shape}")

#         # print(f"y: {self.y.shape}")
#         # print(f"Hx_val: {Hx(x, *hx_args)}")
#         # print(f"y_val: {self.y}")
#         # print(f"K_dot_y: {dot(self.K, self.y).shape}")
#         # print(f"x: {x.shape}")


#         self.x = x.reshape(3,1) + dot(self.K, self.y)  # HEEEREREERERE

#         I_KH = self._I - dot(self.K, H)
#         self.P = dot(I_KH, P).dot(I_KH.T) + dot(self.K, R).dot(self.K.T)

#         # save measurement and posterior state
#         self.z = deepcopy(z)
#         self.x_post = self.x.copy()
#         self.P_post = self.P.copy()

#         # set to None to force recompute
#         self._log_likelihood = None
#         self._likelihood = None
#         self._mahalanobis = None
