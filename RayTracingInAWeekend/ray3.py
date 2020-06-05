# ray3.py

import taichi as ti

ti.init(arch=ti.gpu)

nx = 800
ny = 400
pixels = ti.Vector(3, dt=ti.f32, shape=(nx, ny))
lower_left_corner = ti.Vector([-2.0, -1.0, -1.0])
horizontal = ti.Vector([4.0, 0.0, 0.0])
vertical = ti.Vector([0.0, 2.0, 0.0])
origin = ti.Vector([0.0, 0.0, 0.0])

@ti.func
def unit_vector(v):
    k = 1 / (ti.sqrt(v.dot(v)))
    return k * v


@ti.func
def color(o, d):
    unit_d = unit_vector(d)
    t = 0.5 * (unit_d[1] + 1.0)
    return (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])


@ti.kernel
def paint():
    for i, j in pixels:
        u = float(i) / float(nx)
        v = float(j) / float(ny)
        direction = lower_left_corner + u * horizontal + v * vertical
        pixels[i, j] = color(origin, direction)


gui = ti.GUI("Ray3", (nx, ny))
for i in range(1000):
    paint()
    gui.set_image(pixels.to_numpy())
    gui.show()
