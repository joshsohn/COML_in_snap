from snapstack_msgs.msg import Wind
import jax
import jax.numpy as jnp

class WindSim():
    def __init__(self, key, num_traj, T, dt, wind_type):
        self.key = key
        self.num_traj = num_traj
        self.T = T
        self.dt = dt
        self.wind_type = wind_type
        self.t = jnp.arange(0, self.T + self.dt, self.dt)

        if self.wind_type is None:
            self.w_nominal_vectors = jnp.zeros((self.num_traj, self.t, 3))
        if self.wind_type is 'sine':
            keys = jax.random.split(self.key, 3)  # Split the key into parts for amplitude, period, and phase
        
            # Random amplitudes between 0.5 and 3.5
            amplitudes = 0.5 + jax.random.uniform(keys[0], shape=(self.num_traj, 3)) * 3.0
            
            # Random periods between 1.0 and 4.0
            periods = 1.0 + jax.random.uniform(keys[1], shape=(self.num_traj, 3)) * 3.0
            
            # Random phase shifts between 0 and 2*pi
            phases = jax.random.uniform(keys[2], shape=(self.num_traj, 3)) * 2 * jnp.pi
            
            # Calculate the sine wave for each trajectory and each dimension
            self.winds = jnp.array([amplitudes[i, :] * jnp.sin(2 * jnp.pi / periods[i, :] * self.t[:, None] + phases[i, :])
                            for i in range(self.num_traj)])
        else:
            # Sample wind velocities from the training distribution
            self.w_min = 0.  # minimum wind velocity
            self.w_max = 12.  # maximum wind velocity
            self.a = 5.      # shape parameter `a` for beta distribution
            self.b = 9.      # shape parameter `b` for beta distribution
            self.key, subkey = jax.random.split(self.key, 2)
            self.w = self.w_min + (self.w_max - self.w_min)*jax.random.beta(subkey, self.a, self.b, (self.num_traj,))

            # Randomize wind direction
            random_vectors = jax.random.normal(self.key, (self.num_traj, 3))
            unit_vectors = random_vectors/jnp.linalg.norm(random_vectors, axis=1, keepdims=True)
            self.w_nominal_vectors = self.w[:, jnp.newaxis]*unit_vectors*3

            self.winds = jnp.repeat(self.w_nominal_vectors[:, jnp.newaxis, :], len(self.t), axis=1)
    
    def create_wind(self, w_nominal_vector):
        wind = Wind()

        wind.w_nominal.x = w_nominal_vector[0]
        wind.w_nominal.y = w_nominal_vector[1]
        wind.w_nominal.z = w_nominal_vector[2]

        wind.w_gust.x = 0
        wind.w_gust.y = 0
        wind.w_gust.z = 0

        return wind

    def generate_all_winds(self):
        all_winds = []
        for i in range(self.num_traj):
            wind_i = []
            for wind_t in self.winds[i]:
                wind_i.append(self.create_wind(wind_t))
            all_winds.append(wind_i)
        
        return all_winds