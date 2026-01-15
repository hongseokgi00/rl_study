class InvertedDoublePendulumTermination:
    def __init__(self, min_tip_height: float):
        self.min_tip_height = min_tip_height

    def __call__(self, y_tip: float):
        return y_tip <= self.min_tip_height
