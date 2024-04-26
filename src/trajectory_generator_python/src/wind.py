from snapstack_msgs.msg import Wind
import jax
import jax.numpy as jnp

class WindSim():
    def __init__(self, key, num_traj, off=False):
        self.key = key
        self.num_traj = num_traj

        if off:
            self.w = jnp.zeros((self.num_traj,))
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
            self.w_nominal_vectors = self.w[:, jnp.newaxis]*unit_vectors
    
    def create_wind(self, w_nominal_vector):
        wind = Wind()

        wind.w_nominal.x = w_nominal_vector[0]*3
        wind.w_nominal.y = w_nominal_vector[1]*3
        wind.w_nominal.z = w_nominal_vector[2]*3

        wind.w_gust.x = 0
        wind.w_gust.y = 0
        wind.w_gust.z = 0

        return wind

    def generate_all_winds(self):
        all_winds = []
        for i in range(self.num_traj):
            all_winds.append(self.create_wind(self.w_nominal_vectors[i]))
        
        return all_winds
