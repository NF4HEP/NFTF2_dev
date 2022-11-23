#import NF4HEP
#from NF4HEP.inputs import data
#from NF4HEP import bijectors, inputs, utils, base
#NF4HEP.inputs

class foo:
    def __init__(self,
                 a = 2):
        self._a = 2

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self,
          new):
        new_a, new_b = new
        print("The value that is set is", new_a)
        self._a = new_a
        print("The second value is", new_b)

if __name__ == '__main__':
    p = foo(3)
    print(p._a)
    print(p.a)
    p.a = (5, 6)
    print(p._a)
    print(p.a)
        