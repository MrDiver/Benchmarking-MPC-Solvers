class Algorithm:
    def __init__ (self):
        pass

    def step(self,t,q,qd):
        raise NotImplementedError

    def print_state(self):
        raise NotImplementedError