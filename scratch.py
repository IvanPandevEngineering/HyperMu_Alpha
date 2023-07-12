import condition_data as cd
import matplotlib.pyplot as plt
from collections import namedtuple

vehicle_state = namedtuple('state', ['a_fr', 'a_d_fr', 'a_dd_fr'])

self_state = vehicle_state(1, 0.100, 0.000)

print(self_state.a_fr)
print(self_state.a_d_fr)
print(self_state.a_dd_fr)

self_state = vehicle_state(a_fr=100, a_d_fr= 0, a_dd_fr = [0]*2)

print(self_state.a_fr)
print(self_state.a_d_fr)
print(self_state.a_dd_fr)