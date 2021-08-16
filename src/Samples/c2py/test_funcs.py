import wrap
import time
import numpy


def test_add():
    assert(wrap.add(3, 4) == 7)

def add_py(a, b):
    for i in range(10000):
        a += b
    return a+b

if __name__ == '__main__':
    s_t = time.perf_counter()
    a = add_py(3,4)
    e_t = time.perf_counter()
    t1 = e_t-s_t
    print(t1)

    e_t = time.perf_counter()
    a = wrap.add(3, 4)
    t2 = time.perf_counter()- e_t
    print(t2)
    print(t1/t2)