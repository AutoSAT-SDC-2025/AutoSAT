import numpy as np
import matplotlib.pyplot as plt
from .comparer import Comparitor
from .mapper import Mapper
import cv2 as cv
from .kalman import KalmanFilter
import configparser
from pathlib import Path
def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ParticleFilter:
    def __init__(self) -> None:
        current_file = Path(__file__).resolve()
        project_root = current_file.parents[3]  # adjust this number as needed
        config = configparser.ConfigParser()
        config.read(project_root/"configs"/"configs.ini")
        std_x = float(config["ParticleFilter"]["std_x"])
        std_y = float(config["ParticleFilter"]["std_y"])
        std_theta = float(config["ParticleFilter"]["std_theta"])
        self.std = [std_x, std_y, std_theta]
        self.amount = int(config["ParticleFilter"]["particles"])
        self.generator = np.random.default_rng()
        self.mapper = Mapper()
        self.comparitor = Comparitor()
        self.groups = 1
        self.kalman = KalmanFilter()


    def update_particles(self, dx, dy, dtheta, std=None) -> None:
        if std is None:
            std = np.zeros(3)
            std[0] = np.abs(self.std[1]*np.sin(self.theta)) + np.abs(self.std[0]*np.cos(self.theta)) # x
            std[1] = np.abs(self.std[0]*np.sin(self.theta)) + np.abs(self.std[1]*np.cos(self.theta)) # y
            std[2] = self.std[2]
        noise = self.generator.normal(0, std, (self.amount, 3))
        difference = np.tile([dx, dy, dtheta], (self.amount, 1))
        self.particles = self.particles + noise + difference
        self.particles[:,2] = wrap_to_pi(self.particles[:,2])

    def spawn_new_particles(self, x, y, theta) -> None:
        self.particles = np.tile([x, y, theta], (self.amount, 1))
        self.x = x
        self.y = y
        self.theta = theta
        self.kalman.x = np.array([[x], [y], [0], [theta], [0]])  # state (location and velocity)

    def evolve_particles(self, particles) -> None:
        groups = []
        for particle in particles:
            x = particle[0]
            y = particle[1]
            theta = particle[2]
            groups.append(np.tile([x, y, theta], (self.amount//len(particles), 1)))

        self.particles = groups[0]
        for i in range(1, len(groups)):
            self.particles = np.vstack([self.particles, groups[i]])

    def update(self, lane, dx, dy, dtheta) -> None:
        trust_score = self.comparitor.trust_score(lane)
        v = np.sqrt(dx**2+dy**2)
        print("SPEED", v)
        self.update_particles(dx, dy, dtheta)
        particle, idx, score = self.find_location(lane)
        x = particle[0]
        y = particle[1]
        theta = particle[2]
        print("score", score)
        print("trust", trust_score)
        score = min(1, np.exp(-2*(score*trust_score-6)))
        self.kalman.predict(np.array([[v],[dtheta]]), np.array([[x],[y],[theta]]), score)
        self.x = self.kalman.x[0][0]
        self.y = self.kalman.x[1][0]
        self.theta = self.kalman.x[3][0]
        self.particles[idx[0]][0] = self.x
        self.particles[idx[0]][1] = self.y
        self.particles[idx[0]][2] = self.theta
        self.evolve_particles(self.particles[idx])

    def find_location(self, lane):
        lane = cv.resize(lane, (128, 64))
        scores = np.zeros(self.amount)
        features_lane = self.comparitor.get_hog_features(lane)
        for i, particle in enumerate(self.particles):
            pos = (particle[0], particle[1])
            angle = particle[2]
            map_lane = self.mapper.get_sight(pos, angle)
            map_features = self.comparitor.get_hog_features(map_lane)
            scores[i] = self.comparitor.get_distance_features(features_lane, map_features)
        idx = np.argsort(scores)
        return self.particles[idx[0]], idx[:self.groups], scores[idx[0]]

    def plot(self):
        plt.figure(figsize=(6, 6))
        plt.axis('equal')
        plt.grid(True)

        # Plot positions
        plt.scatter(self.particles[:, 0], self.particles[:, 1], color='blue', alpha=0.5, label='Noisy positions')

        # Add orientation arrows
        arrow_len = 0.2
        for i in range(self.amount):
            x_i, y_i, theta_i = self.particles[i]
            dx = arrow_len * np.sin(theta_i)
            dy = arrow_len * np.cos(theta_i)
            plt.arrow(x_i, y_i, dx, dy, head_width=0.05, head_length=0.05, fc='red', ec='red')

        plt.title("Noisy Pose Samples with Orientation")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.show()
