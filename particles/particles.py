from collections.abc import Callable
import random
import arcade
import numpy as np

from rich.progress import track

WORLD_TO_SCREEN_FACTOR = 1
SPRITE_SCALE = 0.2


class Particle:
    def __init__(self, position: np.array, velocity: np.array, mass: float):

        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.force = np.zeros(3)


class ForceGenerator:
    def __init__(self):
        pass

    def update_force(self, particle_list: list[Particle]):
        pass


class Wall:
    def __init__(self, point_a: np.array, point_b: np.array, restitution_coeff: float):

        self.point_a = point_a
        self.point_b = point_b
        self.restitution_coeff = restitution_coeff
        self.ab_vector = self.point_b - self.point_a
        self.length = np.linalg.norm(self.ab_vector)
        self.ab_vector_normalized = self.ab_vector / self.length
        self.normal_vector = np.array(
            [-self.ab_vector_normalized[1], self.ab_vector_normalized[0], 0.0]
        )

    def collision(self, particle_list: list[Particle]):

        for particle in particle_list:

            a_to_particle_vector = particle.position - self.point_a
            normal_dot_product = np.dot(a_to_particle_vector, self.normal_vector)
            tangential_dot_product = np.dot(
                a_to_particle_vector, self.ab_vector_normalized
            )

            if (normal_dot_product <= 0.0) and (
                0 <= tangential_dot_product <= self.length
            ):

                particle.velocity = particle.velocity * self.restitution_coeff

                normal_velocity = self.normal_vector * np.dot(
                    particle.velocity, self.normal_vector
                )
                tangential_velocity = particle.velocity - normal_velocity

                particle.velocity = tangential_velocity - normal_velocity

                particle.position = (
                    particle.position - normal_dot_product * self.normal_vector
                )


class ParticleSystem:
    def __init__(
        self,
        particle_list: list[Particle],
        force_generators: list[ForceGenerator] = None,
        walls: list[Wall] = None,
    ):

        self.particle_list = particle_list
        self.n_particles = len(particle_list)
        self.force_generators = force_generators
        self.walls = walls
        self.state_vector = np.zeros(self.n_particles * 6)

        for i, particle in enumerate(self.particle_list):
            j = i * 6
            self.state_vector[j : j + 3] = particle.position
            self.state_vector[j + 3 : j + 6] = particle.velocity

    def calculate_accelerations(self, initial_state_vector: np.array, time: float):

        # Clear the force acumulators
        for particle in self.particle_list:
            particle.force = np.zeros(3)

        # Call the force generators to update the forces
        for force_generator in self.force_generators:

            force_generator.update_force(self.particle_list)

        acceleration_vector = np.zeros(self.n_particles * 6)
        for i, particle in enumerate(self.particle_list):
            j = i * 6
            acceleration_vector[j : j + 3] = particle.velocity
            acceleration_vector[j + 3 : j + 6] = particle.force / particle.mass

        return acceleration_vector

    def update_particles(self, new_state_vector: float):

        for i, particle in enumerate(self.particle_list):

            j = i * 6
            particle.position = new_state_vector[j : j + 3]
            particle.velocity = new_state_vector[j + 3 : j + 6]

        for wall in self.walls:

            wall.collision(self.particle_list)

        for i, particle in enumerate(self.particle_list):
            j = i * 6
            self.state_vector[j : j + 3] = particle.position
            self.state_vector[j + 3 : j + 6] = particle.velocity

        return self.state_vector.copy()


class GravityField(ForceGenerator):
    def __init__(self, gravity_acceleration_m_s2: float, gravity_vector: np.array):

        self.gravity_acceleration_m_s2 = gravity_acceleration_m_s2
        self.gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)

    def update_force(self, particle_list: list[Particle]):

        for particle in particle_list:

            particle.force = (
                particle.force
                + particle.mass * self.gravity_acceleration_m_s2 * self.gravity_vector
            )


def integrator(
    initial_state_vector: np.array,
    derivative_function: Callable[[np.array], np.array],
    start_time: float,
    time_step: float,
):

    x0 = initial_state_vector
    f = derivative_function
    t0 = start_time
    h = time_step

    x1 = x0 + h * f(x0, t0)

    return x1


def run_particle_simulation(
    particle_system: ParticleSystem,
    start_time: float,
    end_time: float,
    time_step: float,
):

    t0 = start_time
    x0 = particle_system.state_vector
    h = time_step
    f = particle_system.calculate_accelerations

    history = []

    for t in track(
        np.linspace(start_time, end_time, int((end_time - start_time) / time_step)),
        description="Simulating...",
    ):

        history.append([t0, x0])

        x1 = integrator(x0, f, t0, h)
        t1 = t0 + h
        x0 = particle_system.update_particles(x1)
        t0 = t1

    return history


class Window(arcade.Window):
    def __init__(self, screen_width, screen_height, screen_title, center_window=False):

        super().__init__(
            screen_width, screen_height, screen_title, center_window=center_window
        )

        self.sprites_list = arcade.SpriteList()
        self.shapes_list = arcade.ShapeElementList()
        self.history = None
        self.time_record = 0
        self.time_label = arcade.Text(
            text="Time: 0.000s",
            start_x=0,
            start_y=screen_height - 20,
            width=screen_width,
            align="center",
        )

        arcade.set_background_color(arcade.color.DARK_IMPERIAL_BLUE)

    def setup(
        self,
        particle_system: ParticleSystem,
        history,
        time_step=float,
        playback_speed=float,
    ):

        for particle in particle_system.particle_list:

            particle_sprite = arcade.Sprite("particle.png", scale=SPRITE_SCALE)
            particle_sprite.center_x = particle.position[0] * WORLD_TO_SCREEN_FACTOR
            particle_sprite.center_y = particle.position[1] * WORLD_TO_SCREEN_FACTOR
            self.history = history
            self.time_step = time_step
            self.playback_speed = playback_speed

            self.sprites_list.append(particle_sprite)

        for wall in particle_system.walls:

            wall_line = arcade.create_line(
                wall.point_a[0],
                wall.point_a[1],
                wall.point_b[0],
                wall.point_b[1],
                color=arcade.color.WHITE,
                line_width=2,
            )
            self.shapes_list.append(wall_line)

    def on_update(self, delta_time: float):

        step = int(self.playback_speed * (delta_time / self.time_step))
        self.time_record = self.time_record + step

        if self.time_record > (len(self.history) - 1):
            self.time_record = 0

        instant, state = self.history[self.time_record]
        self.time_label.text = f"Time: {instant:.4f}s"

        for i, particle_sprite in enumerate(self.sprites_list):

            j = i * 6
            particle_sprite.center_x = state[j]
            particle_sprite.center_y = state[j + 1]

    def on_draw(self):

        self.clear()
        self.sprites_list.draw()
        self.shapes_list.draw()
        self.time_label.draw()


if __name__ == "__main__":

    screen_width = 900
    screen_height = 900
    screen_title = "Particles"

    n_particles = 100
    max_speed = 100

    start_time = 0.0
    end_time = 100
    time_step = 0.01
    playback_speed = 5
    particle_list = []

    print()
    print("PARTICLE SIMULATOR")
    print()

    for i in range(n_particles):

        position = np.array([screen_width / 2, screen_height * 0.7, 0.0])

        angle = random.random() * 2 * np.pi
        speed = random.random() * max_speed
        velocity = np.array([np.cos(angle) * speed, np.sin(angle) * speed, 0.0])
        mass = 1

        particle = Particle(position, velocity, mass)
        particle_list.append(particle)

    gravity = GravityField(9.81, np.array([0.0, -1.0, 0.0]))
    force_generators = [gravity]
    ground = Wall(
        np.array([screen_width / 3, 2.0, 0.0]),
        np.array([2 * screen_width / 3, 2.0, 0.0]),
        0.85,
    )

    right_ramp = Wall(
        np.array([2 * screen_width / 3, 2.0, 0.0]),
        np.array([screen_width - 2, screen_height / 3, 0.0]),
        1.0,
    )
    ceiling = Wall(
        np.array([screen_width, screen_height - 2, 0.0]),
        np.array([0.0, screen_height - 2, 0.0]),
        1.0,
    )
    right_wall = Wall(
        np.array([screen_width - 2, screen_height / 3, 0.0]),
        np.array([screen_width - 2, screen_height - 2, 0.0]),
        1.0,
    )
    left_wall = Wall(
        np.array([2.0, screen_height - 2, 0.0]),
        np.array([2.0, screen_height / 3, 0.0]),
        1.0,
    )
    left_ramp = Wall(
        np.array([2.0, screen_height / 3, 0.0]),
        np.array([screen_width / 3, 2.0, 0.0]),
        1.0,
    )
    walls = [ground, right_ramp, right_wall, ceiling, left_wall, left_ramp]

    my_particle_system = ParticleSystem(particle_list, force_generators, walls)

    history = run_particle_simulation(
        my_particle_system, start_time, end_time, time_step
    )

    with open("history.csv", "w") as outfile:

        for record in history:
            string_list = [f"{value:.4f}" for value in record[1]]
            string = f"{record[0]:.4f}," + ",".join(string_list) + "\n"
            outfile.write(string)

    particles_window = Window(
        screen_width, screen_height, screen_title, center_window=True
    )
    particles_window.setup(
        particle_system=my_particle_system,
        history=history,
        time_step=time_step,
        playback_speed=playback_speed,
    )

    print("Displaying...")
    print()
    arcade.run()
