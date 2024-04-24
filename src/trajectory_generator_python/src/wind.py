from snapstack_msgs.msg import Wind
import jax
import jax.numpy as jnp

def create_wind(w):
    wind = Wind()
    # random_vector = np.random.randn(3)
    # unit_vector = random_vector/np.linalg.norm(random_vector)
    # wind_vector = unit_vector*w
    # wind.w_nominal.x = 0
    wind.w_nominal.x = 5*w
    wind.w_nominal.y = 5*w
    wind.w_nominal.z = 5*w
    wind.w_gust.x = w
    wind.w_gust.y = w
    wind.w_gust.z = w

    return wind

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

    def generate_all_winds(self):
        all_winds = []
        for i in range(self.num_traj):
            all_winds.append(create_wind(self.w[i]))
        
        return all_winds
