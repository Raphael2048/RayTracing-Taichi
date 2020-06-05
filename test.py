import numpy as np
import math


nx = 800
ny = 400

FOV = 90

def normalize(n):
    s = 1.0 / math.sqrt(np.dot(n, n))
    return n * s

# lookfrom = np.array([-2.0, 2.0, 1.0], dtype=np.float32)
# lookat = np.array([0.0, 0.0, -1.0], dtype=np.float32)
# up = np.array([0.0, 1.0, 0.0], dtype=np.float32)

# lookfrom = np.array([-2.0, 2.0, 1.0], dtype=np.float32)
lookfrom = np.array([1.0, 0.0, 0.0], dtype=np.float32)
lookat = np.array([0.0, 0.0, -1.0], dtype=np.float32)
up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
w = normalize(lookfrom - lookat)
u = normalize(np.cross(up, w))
v = np.cross(w, u)

aspect =  ny / nx
fov_theta = FOV * math.pi / 180
half_width = math.tan(fov_theta / 2)
half_height = half_width * aspect


lower_left_corner =lookfrom - half_width * u - half_height * v - w
center = lookfrom - w
higher_right_cornet = lookfrom + half_width * u + half_height * v - w
horizontal =u * 2 * half_width
vertical =v * 2 * half_height
print(u)
print(v)
print(w)
print(center)
print(lower_left_corner)
print(higher_right_cornet)