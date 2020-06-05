# ray1.py

import taichi as ti

ti.init(arch=ti.gpu)

nx = 200
ny = 100
pixels = ti.Vector(3, dt=ti.f32, shape=(nx, ny))
@ti.kernel
def paint():
  for i, j in pixels:
    r = float(i) / float(nx)
    g = float(j) / float(ny)
    b = 0.2
    pixels[i, j] = ti.Vector([r,g,b])

gui = ti.GUI("Ray1", (200, 100))

for i in range(1000):
  paint()
  gui.set_image(pixels.to_numpy())
  gui.show()
