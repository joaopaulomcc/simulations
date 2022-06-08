from collections.abc import Callable
import random
import arcade
import numpy as np

from rich.progress import track

WORLD_TO_SCREEN_FACTOR = 100
SPRITE_SCALE = 0.2


class Particle:
    def __init__(
        self, position: np.array, velocity: np.array, mass: float, category: str
    ):

        self.position = position
        self.velocity = velocity
        self.mass = mass
        self.category = category
        self.force = np.zeros(3)
        self.index = 0


class ForceGenerator:
    def __init__(self):
        pass

    def update_force(
        self,
        particle_list: list[Particle],
        state_vector: np.array,
        force_vector: np.array,
    ):
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
        particle_list: list[Particle] = None,
        force_generators: list[ForceGenerator] = None,
        walls: list[Wall] = None,
    ):

        self.particle_list = particle_list
        self.force_generators = force_generators
        self.walls = walls

        self.n_particles = len(particle_list)
        self.state_vector = np.zeros(self.n_particles * 6)
        self.force_vector = np.zeros((self.n_particles, 3))

        for i, particle in enumerate(self.particle_list):
            j = i * 6
            particle.index = i
            self.state_vector[j : j + 3] = particle.position
            self.state_vector[j + 3 : j + 6] = particle.velocity

    def calculate_accelerations(self, state_vector: np.array, time: float):

        # Clear the force acumulator
        self.force_vector = np.zeros((self.n_particles, 3))

        # Call the force generators to update the forces
        for force_generator in self.force_generators:

            self.force_vector = force_generator.update_force(
                self.particle_list, state_vector, self.force_vector
            )

        acceleration_vector = np.zeros(self.n_particles * 6)

        for particle in self.particle_list:

            if particle.category != "fixed":
                j = particle.index * 6
                acceleration_vector[j : j + 3] = state_vector[j + 3 : j + 6]
                acceleration_vector[j + 3 : j + 6] = (
                    self.force_vector[particle.index] / particle.mass
                )

        return acceleration_vector

    def update_particles(self, new_state_vector: float):

        for particle in self.particle_list:

            j = particle.index * 6
            particle.position = new_state_vector[j : j + 3]
            particle.velocity = new_state_vector[j + 3 : j + 6]

        for wall in self.walls:

            wall.collision(self.particle_list)

        for particle in self.particle_list:
            j = particle.index * 6
            self.state_vector[j : j + 3] = particle.position
            self.state_vector[j + 3 : j + 6] = particle.velocity

        return self.state_vector.copy()


class GravityField(ForceGenerator):
    def __init__(self, gravity_acceleration_m_s2: float, gravity_vector: np.array):

        self.gravity_acceleration_m_s2 = gravity_acceleration_m_s2
        self.gravity_vector = gravity_vector / np.linalg.norm(gravity_vector)

    def update_force(
        self,
        particle_list: list[Particle],
        state_vector: np.array,
        force_vector: np.array,
    ):

        for particle in particle_list:

            force_vector[particle.index] += (
                particle.mass * self.gravity_acceleration_m_s2 * self.gravity_vector
            )

        return force_vector


class Spring(ForceGenerator):
    def __init__(
        self,
        spring_constant: float,
        damping_coefficient: float,
        particle_a: Particle,
        particle_b: Particle,
    ):

        self.spring_constant = spring_constant
        self.damping_coefficient = damping_coefficient
        self.particle_a = particle_a
        self.particle_b = particle_b
        self.vector = particle_b.position - particle_a.position
        self.rest_length = np.linalg.norm(self.vector)

    def update_force(
        self,
        particle_list: list[Particle],
        state_vector: np.array,
        force_vector: np.array,
    ):

        particle_a_position = state_vector[
            self.particle_a.index * 6 : self.particle_a.index * 6 + 3
        ]
        particle_a_velocity = state_vector[
            self.particle_a.index * 6 + 3 : self.particle_a.index * 6 + 6
        ]

        particle_b_position = state_vector[
            self.particle_b.index * 6 : self.particle_b.index * 6 + 3
        ]
        particle_b_velocity = state_vector[
            self.particle_b.index * 6 + 3 : self.particle_b.index * 6 + 6
        ]

        new_vector = particle_b_position - particle_a_position
        new_length = np.linalg.norm(new_vector)

        direction_vector = new_vector / new_length

        extension = new_length - self.rest_length
        relative_velocity = particle_b_velocity - particle_a_velocity
        damping_velocity = np.dot(relative_velocity, direction_vector)

        spring_force = extension * self.spring_constant
        damping_force = damping_velocity * self.damping_coefficient

        force_vector[self.particle_a.index] += (
            spring_force * direction_vector + damping_force * direction_vector
        )

        force_vector[self.particle_b.index] += (
            -spring_force * direction_vector - damping_force * direction_vector
        )

        return force_vector


def integrator(
    initial_state_vector: np.array,
    derivative_function: Callable[[np.array], np.array],
    start_time: float,
    time_step: float,
    order: int = 1,
):

    x0 = initial_state_vector
    f = derivative_function
    t0 = start_time
    h = time_step

    k1 = h * f(x0, t0)

    if order == 1:
        return x0 + k1

    if order == 2:
        k2 = h * f(x0 + k1 / 2, t0 + h / 2)
        return x0 + k2

    if order == 4:
        k2 = h * f(x0 + k1 / 2, t0 + h / 2)
        k3 = h * f(x0 + k2 / 2, t0 + h / 2)
        k4 = h * f(x0 + k3, t0 + h)

        return x0 + (1 / 6) * k1 + (1 / 3) * k2 + (1 / 3) * k3 + (1 / 6) * k4

    if order not in [1, 2, 4]:
        raise ValueError


def run_particle_simulation(
    particle_system: ParticleSystem,
    start_time: float,
    end_time: float,
    time_step: float,
    integration_order: int = 1,
):

    t0 = start_time
    x0 = particle_system.state_vector
    h = time_step
    f = particle_system.calculate_accelerations

    history = []

    for _ in track(
        np.linspace(start_time, end_time, int((end_time - start_time) / time_step)),
        description="Simulating...",
    ):

        history.append([t0, x0])

        x1 = integrator(x0, f, t0, h, integration_order)
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
        self.springs_list = arcade.ShapeElementList()
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

        self.history = history
        self.time_step = time_step
        self.playback_speed = playback_speed
        self.particle_system = particle_system

        for particle in particle_system.particle_list:

            particle_sprite = arcade.Sprite("particle.png", scale=SPRITE_SCALE)
            particle_sprite.center_x = particle.position[0] * WORLD_TO_SCREEN_FACTOR
            particle_sprite.center_y = particle.position[1] * WORLD_TO_SCREEN_FACTOR
            self.sprites_list.append(particle_sprite)

        for wall in particle_system.walls:

            wall_line = arcade.create_line(
                wall.point_a[0] * WORLD_TO_SCREEN_FACTOR,
                wall.point_a[1] * WORLD_TO_SCREEN_FACTOR,
                wall.point_b[0] * WORLD_TO_SCREEN_FACTOR,
                wall.point_b[1] * WORLD_TO_SCREEN_FACTOR,
                color=arcade.color.WHITE,
                line_width=2,
            )
            self.shapes_list.append(wall_line)

    def on_update(self, delta_time: float):

        step = self.playback_speed * (delta_time / self.time_step)

        self.time_record = self.time_record + step

        if self.time_record > (len(self.history) - 1):
            self.time_record = 0

        instant, state = self.history[int(self.time_record)]
        self.time_label.text = f"Time: {instant:.4f}s"

        for i, particle_sprite in enumerate(self.sprites_list):

            j = i * 6
            particle_sprite.center_x = state[j] * WORLD_TO_SCREEN_FACTOR
            particle_sprite.center_y = state[j + 1] * WORLD_TO_SCREEN_FACTOR

        self.springs_list = arcade.ShapeElementList()

        for force_generator in self.particle_system.force_generators:

            if isinstance(force_generator, Spring):

                particle_a_index = force_generator.particle_a.index
                particle_b_index = force_generator.particle_b.index

                spring_line = arcade.create_line(
                    state[particle_a_index * 6] * WORLD_TO_SCREEN_FACTOR,
                    state[particle_a_index * 6 + 1] * WORLD_TO_SCREEN_FACTOR,
                    state[particle_b_index * 6] * WORLD_TO_SCREEN_FACTOR,
                    state[particle_b_index * 6 + 1] * WORLD_TO_SCREEN_FACTOR,
                    line_width=3,
                    color=arcade.color.RED_ORANGE,
                )

                self.springs_list.append(spring_line)

    def on_draw(self):

        self.clear()
        self.springs_list.draw()
        self.shapes_list.draw()
        self.sprites_list.draw()
        self.time_label.draw()


if __name__ == "__main__":

    screen_width = 900
    screen_height = 900
    screen_title = "Particles"

    n_particles = 100
    max_speed = 10

    start_time = 0.0
    end_time = 100
    time_step = 0.1
    integration_order = 4
    playback_speed = 1
    particle_list = []

    print()
    print("PARTICLE SIMULATOR")
    print()

    mass = 1

    for _ in range(n_particles):

        position = np.array([9 * 0.5, 9 * 0.7, 0.0])

        angle = random.random() * 2 * np.pi
        speed = random.random() * max_speed
        velocity = np.array([np.cos(angle) * speed, np.sin(angle) * speed, 0.0])

        particle = Particle(position, velocity, mass, "normal")
        particle_list.append(particle)

    particle_a = Particle(
        np.array([4.5, 4.5, 0.0]), velocity=np.array([-100.0, 0.0, 0.0]), mass=1, category="normal"
    )
    particle_b = Particle(
        np.array([5.5, 5.5, 0.0]), velocity=[0.0, -10.0, 0.0], mass=1, category="normal"
    )
    particle_c = Particle(
        np.array([4.5, 6.5, 0.0]), velocity=[10.0, 0.0, 0.0], mass=1, category="normal"
    )
    particle_d = Particle(
        np.array([3.5, 5.5, 0.0]), velocity=[0.0, 10.0, 0.0], mass=1, category="normal"
    )

    particle_list += [particle_a, particle_b, particle_c, particle_d]

    spring_b_c = Spring(100.0, 1.0, particle_b, particle_c)
    spring_c_d = Spring(100.0, 1.0, particle_c, particle_d)
    spring_d_a = Spring(100.0, 1.0, particle_d, particle_a)
    spring_a_c = Spring(100.0, 1.0, particle_a, particle_c)
    spring_a_b = Spring(100.0, 1.0, particle_a, particle_b)
    spring_b_d = Spring(100.0, 1.0, particle_b, particle_d)

    gravity = GravityField(9.81, np.array([0.0, -1.0, 0.0]))
    force_generators = [
        gravity,
        spring_b_c,
        spring_a_b,
        spring_c_d,
        spring_d_a,
        spring_a_c,
        spring_b_d,
    ]

    world_height = screen_height / WORLD_TO_SCREEN_FACTOR
    world_width = screen_width / WORLD_TO_SCREEN_FACTOR

    ground = Wall(
        np.array([world_width / 3, 0.0, 0.0]),
        np.array([2 * world_width / 3, 0.0, 0.0]),
        0.85,
    )

    right_ramp = Wall(
        np.array([2 * world_width / 3, 0.0, 0.0]),
        np.array([world_width, world_height / 3, 0.0]),
        1.0,
    )
    ceiling = Wall(
        np.array([world_width, world_height, 0.0]),
        np.array([0.0, world_width, 0.0]),
        1.0,
    )
    right_wall = Wall(
        np.array([world_width, world_height / 3, 0.0]),
        np.array([world_width, world_height, 0.0]),
        1.0,
    )
    left_wall = Wall(
        np.array([0.0, world_height, 0.0]),
        np.array([0.0, world_height / 3, 0.0]),
        1.0,
    )
    left_ramp = Wall(
        np.array([0.0, world_height / 3, 0.0]),
        np.array([world_width / 3, 0.0, 0.0]),
        1.0,
    )
    walls = [ground, right_ramp, right_wall, ceiling, left_wall, left_ramp]

    my_particle_system = ParticleSystem(particle_list, force_generators, walls)

    history = run_particle_simulation(
        my_particle_system, start_time, end_time, time_step, integration_order
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
