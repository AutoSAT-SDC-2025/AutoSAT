import numpy as np
import matplotlib.pyplot as plt
from comparer import Comparitor, KDTreeComparitor
from mapper import Mapper
import cv2 as cv

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ParticleFilter:
    def __init__(self, x, y, theta, x_std, y_std, theta_std, amount) -> None:
        self.x_std = x_std
        self.y_std = y_std
        self.theta_std = theta_std
        self.std = [x_std, y_std, theta_std]
        self.amount = amount
        self.generator = np.random.default_rng()
        self.particles = np.tile([x, y, theta], (amount, 1))
        map = cv.imread("../var/map.png", cv.IMREAD_GRAYSCALE)
        self.mapper = Mapper(scale=0.0483398, map=map)
        self.comparitor = Comparitor()
        self.groups = 1
        self.x = x
        self.y = y
        self.theta = theta


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
        if self.comparitor.trust_score(lane) < 0.4:
            print(self.comparitor.trust_score(lane))
            self.update_particles(dx, dy, dtheta, std=np.zeros(3))
            return
        self.update_particles(dx, dy, dtheta)
        particle, idx, score = self.find_location(lane)
        self.x = particle[0]
        self.y = particle[1]
        self.theta = particle[2]
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
