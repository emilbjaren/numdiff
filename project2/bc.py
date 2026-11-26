
class BC():
    def __init__ (self, value):
        self.value = value

class Dirichlet(BC):
    def __init__(self, value):
        super().__init__(value)

class Neumann(BC):
    def __init__(self, value):
        super().__init__(value)


