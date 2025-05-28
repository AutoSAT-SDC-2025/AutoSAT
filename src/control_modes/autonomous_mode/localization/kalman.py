import numpy as np
from pathlib import Path
import configparser


class KalmanFilter:
    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # adjust this number as needed
        config = configparser.ConfigParser()
        config.read(project_root/"config"/"config.ini")

        self.dt = 1
        # x, y, v, theta, omega
        self.x = np.array([[0], [0], [0], [0], [0]])  # state (location and velocity)
        self.A = np.array(
            [
                [
                    1,
                    0,
                    -self.dt * np.sin(self.x[3, 0]),
                    -self.x[2, 0] * self.dt * np.cos(self.x[3, 0]),
                    0,
                ],
                [
                    0,
                    1,
                    self.dt * np.cos(self.x[3, 0]),
                    -self.x[2, 0] * self.dt * np.sin(self.x[3, 0]),
                    0,
                ],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, self.dt],
                [0, 0, 0, 0, 1],
            ]
        )
        # We measure speed an orientation speed
        # z = [speed, angle_speed]
        self.H_A = np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1]])  # Measurement function
        self.H_B = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])  # Measurement function
        # Process noise covariance (Q)
        self.Q = np.diag([1e-4, 1e-4, 1e-2, 1e-2, 1e-3])

        # Measurement noise covariance (R)
        self.R_A = np.diag(
            [5**2, 0.1**2]
        )  # GPS error 5m, IMU error 0.1 rad, speed error 2 m/s
        self.R_A = np.diag(
            [float(config["Kalman"]["speed"])**2, float(config["Kalman"]["rotation_speed"])**2]
        )
        self.R_B = np.diag(
            [1**2, 1**2, 0.01**2]
        )
        self.R_B = np.diag(
            [float(config["Kalman"]["x"])**2, float(config["Kalman"]["y"])**2, float(config["Kalman"]["theta"])**2]
        )

        # Error covariance matrix (P)
        self.P = np.eye(5) * 1e3  # Large initial uncertainty

    def predict(self, z_a, z_b, score):
        self.A = np.array(
            [
                [
                    1,
                    0,
                    -self.dt * np.sin(self.x[3, 0]),
                    0,
                    0,
                ],
                [
                    0,
                    1,
                    self.dt * np.cos(self.x[3, 0]),
                    0,
                    0,
                ],

                [0, 0, 1, 0, 0],
                [0, 0, 0, 1, self.dt],
                [0, 0, 0, 0, 1],
            ]
        )

        x_p = self.A @ self.x  # add control here
        p_p = self.A @ self.P @ self.A.T + self.Q

        # Sensor A
        K = p_p @ self.H_A.T @ np.linalg.inv(self.H_A @ p_p @ self.H_A.T + self.R_A)
        x_p = x_p + K @ (z_a - self.H_A @ x_p)
        self.P = (np.eye(5) - K @ self.H_A) @ p_p

        # Sensor H_B
        R_B = np.diag(
            [self.R_B[0,0]*abs(np.cos(x_p[3,0]))+self.R_B[1,1]*abs(np.sin(x_p[3,0])), self.R_B[0,0]*abs(np.sin(x_p[3,0]))+self.R_B[1,1]*abs(np.cos(x_p[3,0])), self.R_B[2,2]]
        )
        # R_B = R_B*(np.exp((1-score)/(score+0.1)))
        # R_B = R_B* (1 + 99 * (1 - 1 / (1 + np.exp(-20 * (score - 0.5)))))
        K = p_p @ self.H_B.T @ np.linalg.inv(self.H_B @ p_p @ self.H_B.T + R_B)
        self.x = x_p + K @ (z_b - self.H_B @ x_p)
        self.P = (np.eye(5) - K @ self.H_B) @ p_p

        return self.x

if __name__ == "__main__":
    kalman = KalmanFilter()
    kalman.x = np.array([[0], [0], [50], [np.pi/2], [0]])  # state (location and velocity)

    for i in range(10):
        print(kalman.x)
        kalman.predict(np.array([[50], [0]]), np.array([[0],[0],[0]]), 1)
