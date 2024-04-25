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
            self.w_min = 0.  # minimum wind velocity in inertial `x`-direction
            self.w_max = 6.  # maximum wind velocity in inertial `x`-direction
            self.a = 5.      # shape parameter `a` for beta distribution
            self.b = 9.      # shape parameter `b` for beta distribution
            self.key, subkey = jax.random.split(self.key, 2)
            self.w = self.w_min + (self.w_max - self.w_min)*jax.random.beta(subkey, self.a, self.b, (self.num_traj,))
    
    def create_wind(self, w):
        wind = Wind()
        # wind.w_nominal.x = w*5

        random_vector = jax.random.normal(self.key, (3,))
        unit_vector = random_vector/jnp.linalg.norm(random_vector)
        w_nominal_vector = w*unit_vector
        wind.w_nominal.x = w_nominal_vector[0]*5
        wind.w_nominal.y = w_nominal_vector[1]*5
        wind.w_nominal.z = w_nominal_vector[2]*5
        
        random_vector = jax.random.normal(self.key, (3,))
        unit_vector = random_vector/jnp.linalg.norm(random_vector)
        w_gust_vector = w*unit_vector
        wind.w_gust.x = w_gust_vector[0]
        wind.w_gust.y = w_gust_vector[1]
        wind.w_gust.z = w_gust_vector[2]

        return wind

    def generate_all_winds(self):
        all_winds = []
        for i in range(self.num_traj):
            all_winds.append(self.create_wind(self.w[i]))
        
        return all_winds
