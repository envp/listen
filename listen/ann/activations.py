from enum import Enum


class Activations(Enum):
    SIGN = 'sign'
    SIGMOID = 'sigmoid'
    TANH = 'tanh'

    def func_name(self):
        return self.name

    def derivfunc_name(self):
        return "{}_derivative".format(self.name)
