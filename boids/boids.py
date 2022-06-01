import random
import math

import arcade
import numpy as np


SCREEN_WIDTH = 900
SCREEN_HEIGHT = 900
SCREEN_TITLE = "Boids"
BOIDS_SCALE = 0.5
BOIDS_ACCELERATION = 100
BOIDS_MAX_SPEED = 100
N_BOIDS = 350

COHESION_WEIGHT = 0.05
SPACING_WEIGHT = 1
ALIGNMENT_WEIGHT = 0.2
INFLUENCE_RADIUS = 300
MIN_DISTANCE = 40
WORLD_BOUNDAIRES = "reflect"


class Simulation(arcade.Window):
    def __init__(self):

        super().__init__(SCREEN_WIDTH, SCREEN_HEIGHT, SCREEN_TITLE)

        self.boids_list = None
        self.total_weights = COHESION_WEIGHT + SPACING_WEIGHT + ALIGNMENT_WEIGHT
        self.cohesion_factor = COHESION_WEIGHT / self.total_weights
        self.spacing_factor = SPACING_WEIGHT / self.total_weights
        self.alignment_factor = ALIGNMENT_WEIGHT / self.total_weights
        self.boids_position = np.zeros((N_BOIDS, 2))
        self.boids_velocity = np.zeros((N_BOIDS, 2))

        arcade.set_background_color(arcade.color.DARK_IMPERIAL_BLUE)

    def setup(self):

        self.boids_list = arcade.SpriteList()

        for i in range(N_BOIDS):

            boid_sprite = Boid("./boid.png", scale=BOIDS_SCALE)

            radius = np.random.randint(0, SCREEN_HEIGHT / 2)
            theta = np.random.rand() * np.pi * 2

            boid_sprite.center_x = radius * np.cos(theta) + (SCREEN_WIDTH / 2)
            boid_sprite.center_y = radius * np.sin(theta) + (SCREEN_HEIGHT / 2)
            boid_sprite.angle = random.randint(0, 360)
            speed = random.randint(0.5 * BOIDS_MAX_SPEED, BOIDS_MAX_SPEED)
            boid_sprite.velocity[0] = speed * math.cos(math.radians(boid_sprite.angle))
            boid_sprite.velocity[1] = speed * math.sin(math.radians(boid_sprite.angle))
            boid_sprite.acceleration = BOIDS_ACCELERATION * np.random.rand(2)

            self.boids_list.append(boid_sprite)

            self.boids_position[i][0] = boid_sprite.center_x
            self.boids_position[i][1] = boid_sprite.center_y
            self.boids_velocity[i][0] = boid_sprite.velocity[0]
            self.boids_velocity[i][1] = boid_sprite.velocity[1]

    def on_mouse_drag(
        self, x: float, y: float, dx: float, dy: float, buttons: int, modifiers: int
    ):

        for boid in self.boids_list:

            new_angle = math.atan2(y - boid.center_y, x - boid.center_x)

            if buttons == 1:
                boid.angle = math.degrees(new_angle) + random.randint(0, 5)

            else:
                boid.angle = math.degrees(new_angle) + 180 + random.randint(0, 5)

            boid.velocity[0] = math.cos(math.radians(boid.angle)) * BOIDS_MAX_SPEED

            boid.velocity[1] = math.sin(math.radians(boid.angle)) * BOIDS_MAX_SPEED

    def on_mouse_release(self, x: float, y: float, button: int, modifiers: int):

        for boid in self.boids_list:

            boid.acceleration[0] = 0.0
            boid.acceleration[1] = 0.0

    def on_update(self, delta_time: float):

        for i, boid in enumerate(self.boids_list):

            acceleration_vector = np.zeros(2)

            distance_vector = self.boids_position - boid.position()
            distances = np.sqrt(distance_vector[:, 0] ** 2 + distance_vector[:, 1] ** 2)

            inside_influence_radius = distances <= INFLUENCE_RADIUS

            center_of_mass = np.mean(
                self.boids_position[inside_influence_radius], axis=0
            )
            velocity_average = np.mean(
                self.boids_velocity[inside_influence_radius], axis=0
            )
            cohesion_vector = center_of_mass - boid.position()

            if np.linalg.norm(cohesion_vector) != 0:

                acceleration_vector = acceleration_vector + (
                    self.cohesion_factor
                ) * cohesion_vector / np.linalg.norm(cohesion_vector)

                acceleration_vector = acceleration_vector + (
                    -self.spacing_factor * (1 / np.mean(distances[inside_influence_radius]))
                ) * cohesion_vector / np.linalg.norm(cohesion_vector)

            if np.linalg.norm(velocity_average) != 0:

                acceleration_vector = (
                    acceleration_vector
                    + self.alignment_factor
                    * velocity_average
                    / np.linalg.norm(velocity_average)
                )

            boid.acceleration = BOIDS_ACCELERATION * acceleration_vector

            if WORLD_BOUNDAIRES == "repel":

                if boid.center_x < 50:

                    boid.acceleration[0] = (
                        boid.acceleration[0] + 1 * BOIDS_ACCELERATION
                    )

                elif (SCREEN_WIDTH - boid.center_x) < 50:

                    boid.acceleration[0] = (
                        boid.acceleration[0] - 1 * BOIDS_ACCELERATION
                    )

                if boid.center_y < 50:

                    boid.acceleration[1] = (
                        boid.acceleration[1] + 1 * BOIDS_ACCELERATION
                    )

                elif (SCREEN_HEIGHT - boid.center_y) < 50:

                    boid.acceleration[1] = (
                        boid.acceleration[1] - 1 * BOIDS_ACCELERATION
                    )

            boid.velocity = boid.velocity + boid.acceleration * delta_time
            velocity_angle = math.atan2(boid.velocity[1], boid.velocity[0])

            if np.linalg.norm(boid.velocity) >= BOIDS_MAX_SPEED:

                boid.velocity[0] = math.cos(velocity_angle) * BOIDS_MAX_SPEED
                boid.velocity[1] = math.sin(velocity_angle) * BOIDS_MAX_SPEED

            position = boid.position() + boid.velocity * delta_time

            boid.center_x = position[0]
            boid.center_y = position[1]

            if WORLD_BOUNDAIRES == "reflect":

                if (boid.center_x <= 0) or (boid.center_x >= SCREEN_WIDTH):
                    boid.velocity[0] = -boid.velocity[0]

                if (boid.center_y <= 0) or (boid.center_y >= SCREEN_HEIGHT):
                    boid.velocity[1] = -boid.velocity[1]

            elif WORLD_BOUNDAIRES == "portal":

                if boid.center_x <= 0:
                    boid.center_x = SCREEN_WIDTH

                elif boid.center_x >= SCREEN_WIDTH:
                    boid.center_x = 0

                if boid.center_y <= 0:
                    boid.center_y = SCREEN_HEIGHT

                elif boid.center_y >= SCREEN_HEIGHT:
                    boid.center_y = 0

            boid.angle = math.degrees(velocity_angle)

            self.boids_position[i][0] = boid.center_x
            self.boids_position[i][1] = boid.center_y
            self.boids_velocity[i][0] = boid.velocity[0]
            self.boids_velocity[i][1] = boid.velocity[1]

    def on_draw(self):
        self.clear()
        self.boids_list.draw()


class Boid(arcade.Sprite):
    def __init__(self, image_path, scale):
        super().__init__(image_path, scale=scale)

        self.acceleration = np.zeros(2)
        self.velocity = np.zeros(2)

    def speed(self):
        return math.sqrt(self.velocity_x**2 + self.velocity_y**2)

    def position(self):
        return np.array([self.center_x, self.center_y])


def main():

    window = Simulation()
    window.setup()
    arcade.run()


if __name__ == "__main__":
    main()
