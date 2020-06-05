# ray4.py

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
def hit_sphere(center, radius, rayo, rayd):
    l = center - rayo
    l2 = l.dot(l)
    r2 = radius * radius
    rst = False
    # 起始点在球形内部,必然相交
    if l2 < r2:
        rst = True
    else:
        # 计算投影长度
        s = l.dot(rayd)
        # 在射线方向上投影为负,说明射线方向是远离球体的,因此不会相交
        if s < 0:
            rst = True
        else:
            # 射线到球体中心的距离平方
            m2 = l2 - s * s
            rst = m2 < r2
    return rst

@ti.func
def color(o, d):
    unit_d = unit_vector(d)
    rst = ti.Vector([1.0, 0.0, 0.0])
    if hit_sphere(ti.Vector([0.0, 0.0, -1.0]), 0.5, o, unit_d):
        pass
    else:
        t = 0.5 * (unit_d[1] + 1.0)
        rst =  (1.0 - t) * ti.Vector([1.0, 1.0, 1.0]) + t * ti.Vector([0.5, 0.7, 1.0])
    return rst


@ti.kernel
def paint():
    for i, j in pixels:
        u = float(i) / float(nx)
        v = float(j) / float(ny)
        direction = lower_left_corner + u * horizontal + v * vertical
        pixels[i, j] = color(origin, direction)


gui = ti.GUI("Ray4", (nx, ny))
for i in range(1000):
    paint()
    gui.set_image(pixels.to_numpy())
    gui.show()
