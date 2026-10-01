import gymnasium as gym

class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale: float):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        return float(reward) * self.scale