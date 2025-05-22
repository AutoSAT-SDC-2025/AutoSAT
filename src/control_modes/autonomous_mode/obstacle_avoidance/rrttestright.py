import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
import random

# Node class representing a state in the space
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.cost = 0

# RRT* algorithm
class RRTStar:
    def __init__(self, start, goal, num_obstacles, map_size, step_size=0.5, max_iter=500, obstacles=None):
        self.start = start if isinstance(start, Node) else Node(start[0], start[1])
        self.goal = goal if isinstance(goal, Node) else Node(goal[0], goal[1])
        self.map_size = map_size
        self.obstacles = obstacles if obstacles is not None else self.generate_random_obstacles(num_obstacles)
        self.step_size = step_size
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.goal_region_radius = 0.4
        self.search_radius = 5.0
        self.path = None
        self.goal_reached = False

        # Visualization setup
        self.fig, self.ax = plt.subplots()
        self.setup_visualization()

    def is_in_collision(self, node, obstacle_bbox):
        x1, y1, x2, y2 = obstacle_bbox
        return x1 <= node.x <= x2 and y1 <= node.y <= y2

    def generate_random_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            ox = -1.5
            oy = 5
            width = 3
            height = 4
            obstacles.append((ox, oy, width, height))  # Rectangle defined by bottom-left corner and size
        return obstacles

    def setup_visualization(self):
        self.ax.plot(self.start.x, self.start.y, 'bo', label='Start')
        self.ax.plot(self.goal.x, self.goal.y, 'ro', label='Goal')
        self.ax.set_xlim(-4, 4)
        #self.ax.set_xlim(-self.map_size[0], self.map_size[0])
        self.ax.set_ylim(0, self.map_size[1])
        self.ax.grid(True)

        # Draw the randomly generated obstacles
        self.draw_obstacles()

    def draw_obstacles(self):
        for (ox, oy, width, height) in self.obstacles:
            rect = plt.Rectangle((ox, oy), width, height, color='r')
            self.ax.add_patch(rect)

    def smooth_path(self, path, iterations=100):
        if not path:
            return path

        for _ in range(iterations):
            i = random.randint(0, len(path) - 2)
            j = random.randint(i + 1, len(path) - 1)

            p1, p2 = path[i], path[j]
            if self.is_straight_path_collision_free(p1, p2):
                # Replace the intermediate points with the shortcut
                path = path[:i + 1] + path[j:]
        return path

    def is_straight_path_collision_free(self, p1, p2):
        x1, y1 = (p1.x, p1.y) if hasattr(p1, 'x') else (p1[0], p1[1])
        x2, y2 = (p2.x, p2.y) if hasattr(p2, 'x') else (p2[0], p2[1])

        dx = x2 - x1
        dy = y2 - y1
        distance = math.hypot(dx, dy)
        steps = int(np.ceil(distance / 0.05))

        for i in range(steps + 1):
            t = i / steps
            x = x1 + t * dx
            y = y1 + t * dy

            for (ox, oy, width, height) in self.obstacles:
                if ox <= x <= ox + width and oy <= y <= oy + height:
                    return False
        return True

    def plan(self):
        """Main RRT* planning loop."""
        for i in range(self.max_iter):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if self.is_collision_free(new_node):
                neighbors = self.find_neighbors(new_node)
                new_node = self.choose_parent(neighbors, nearest_node, new_node)
                self.node_list.append(new_node)
                self.rewire(new_node, neighbors)

            if self.reached_goal(new_node):
                self.path = self.generate_final_path(new_node)
                self.goal_reached = True
                return

    def get_random_node(self):
        if random.random() < 0.1:
            return Node(self.goal.x, self.goal.y)

        sample_near_obstacle = random.random() < 0.1
        if sample_near_obstacle:
            rand_x = random.uniform(-4, -2)
            rand_y = random.uniform(-self.map_size[1]-0.5, self.map_size[1]+0.5)
        else:
            rand_x = random.uniform(-self.map_size[0], 0)
            rand_y = random.uniform(-self.map_size[1], self.map_size[1])

        return Node(rand_x, rand_y)

    def is_near_obstacle_and_x_greater_than_minus_one(self, node, threshold=1.0):
        if node.x <= -1.5 or node.x >= 4.0:
            return False  # x is within the allowed range

        for (ox, oy, width, height) in self.obstacles:
            # Check distance from node to obstacle bounding box
            closest_x = max(ox, min(node.x, ox + width))
            closest_y = max(oy, min(node.y, oy + height))
            dist = math.hypot(node.x - closest_x, node.y - closest_y)
            if dist < threshold:
                return True  # Too close to obstacle while x > -1

        return False

    def steer(self, from_node, to_node):
        theta = math.atan2(to_node.y - from_node.y, to_node.x - from_node.x)
        new_x = from_node.x + self.step_size * math.cos(theta)
        new_y = from_node.y + self.step_size * math.sin(theta)

        new_node = Node(new_x, new_y)
        new_node.cost = from_node.cost + self.step_size
        new_node.parent = from_node

        if self.is_near_obstacle_and_x_greater_than_minus_one(new_node):
            return None

        return new_node

    def is_collision_free(self, node):
        if node.parent is None:
            return True

        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        distance = math.hypot(dx, dy)
        steps = int(np.ceil(distance / (self.step_size / 2)))

        for i in range(steps + 1):
            t = i / steps
            x = node.parent.x + t * dx
            y = node.parent.y + t * dy

            for (ox, oy, width, height) in self.obstacles:
                if ox <= x <= ox + width and oy <= y <= oy + height:
                    return False

        return True

    def find_neighbors(self, new_node):
        """Find nearby nodes within the search radius."""
        return [node for node in self.node_list
                if np.linalg.norm([node.x - new_node.x, node.y - new_node.y]) < self.search_radius]

    def choose_parent(self, neighbors, nearest_node, new_node):
        """Choose the best parent for the new node based on cost."""
        min_cost = nearest_node.cost + np.linalg.norm([new_node.x - nearest_node.x, new_node.y - nearest_node.y])
        best_node = nearest_node

        for neighbor in neighbors:
            cost = neighbor.cost + math.hypot(new_node.x - neighbor.x, new_node.y - neighbor.y)
            if cost < neighbor.cost and self.is_collision_free(neighbor):
                best_node = neighbor
                min_cost = cost

        new_node.cost = min_cost
        new_node.parent = best_node
        return new_node

    def rewire(self, new_node, neighbors):
        """Rewire the tree by checking if any neighbor should adopt the new node as a parent."""
        for neighbor in neighbors:
            cost = neighbor.cost + math.hypot(new_node.x - neighbor.x, new_node.y - neighbor.y)
            if cost < neighbor.cost and self.is_collision_free(neighbor):
                neighbor.parent = new_node
                neighbor.cost = cost

    def reached_goal(self, node):
        """Check if the goal has been reached."""
        return np.linalg.norm([node.x - self.goal.x, node.y - self.goal.y]) < self.goal_region_radius

    def generate_final_path(self, goal_node):
        path = []
        node = goal_node
        while node is not None:
            path.append(node)
            node = node.parent
        return path[::-1]    # Reverse the path

    def get_nearest_node(self, node_list, rand_node):
        """Find the nearest node in the tree to the random node."""
        distances = [np.linalg.norm([node.x - rand_node.x, node.y - rand_node.y]) for node in node_list]
        nearest_node = node_list[np.argmin(distances)]
        return nearest_node

    def draw_tree(self, node):
        """Draw a tree edge from the current node to its parent."""
        if node.parent:
            self.ax.plot([node.x, node.parent.x], [node.y, node.parent.y], "-b")

    def draw_path(self):
        if self.path:
            self.ax.plot([node.x for node in self.path], [node.y for node in self.path], '-g', label='Path')

    def straighten_car(self):
        self.goal.x = 12
        if self.node_list:
            last_node = self.node_list[-1]
            if self.reached_goal(last_node):
                print("Goal reached")

    def search(self, max_iterations=500, obstacle_bbox=None):
        for _ in range(max_iterations):
            rand_node = self.get_random_node()
            nearest_node = self.get_nearest_node(self.node_list, rand_node)
            new_node = self.steer(nearest_node, rand_node)

            if new_node is None:
                continue

            if obstacle_bbox and self.is_in_collision(new_node, obstacle_bbox):
                continue

            neighbors = self.find_neighbors(new_node)
            new_node = self.choose_parent(neighbors, nearest_node, new_node)
            self.node_list.append(new_node)
            self.rewire(new_node, neighbors)

            if self.reached_goal(new_node):
                self.goal_reached = True
                self.path = self.generate_final_path(new_node)
                return self.path

        return None

# Animation function to run the RRT* iterations and plot in real-time
def animate(i):
    if i < rrt_star.max_iter and not rrt_star.goal_reached:
        rand_node = rrt_star.get_random_node()
        nearest_node = rrt_star.get_nearest_node(rrt_star.node_list, rand_node)
        new_node = rrt_star.steer(nearest_node, rand_node)

        if rrt_star.is_collision_free(new_node):
            neighbors = rrt_star.find_neighbors(new_node)
            new_node = rrt_star.choose_parent(neighbors, nearest_node, new_node)
            rrt_star.node_list.append(new_node)
            rrt_star.rewire(new_node, neighbors)
            rrt_star.draw_tree(new_node)

        if rrt_star.reached_goal(new_node):
            rrt_star.path = rrt_star.generate_final_path(new_node)
            rrt_star.draw_path()
            rrt_star.goal_reached = True  # Set the goal reached flag"""

# Main execution
if __name__ == "__main__":
    start = [0, 0]
    goal = [0, 15]
    num_obstacles = 1  # Number of random obstacles
    map_size = [10, 20]

    rrt_star = RRTStar(start, goal, num_obstacles, map_size)
    rrt_star.search()
    if rrt_star.path:
        rrt_star.path = rrt_star.smooth_path(rrt_star.path, iterations=200)
        rrt_star.draw_path()

    # Create animation
    ani = animation.FuncAnimation(rrt_star.fig, animate, frames=rrt_star.max_iter, interval=10, repeat=False)
    plt.show()