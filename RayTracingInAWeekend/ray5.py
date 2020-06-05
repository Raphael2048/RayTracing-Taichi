import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)

nx = 800
ny = 400
pixels = ti.Vector(3, dt=ti.f32, shape=(nx, ny))
lower_left_corner = ti.Vector([-2.0, -1.0, -1.0])
horizontal = ti.Vector([4.0, 0.0, 0.0])
vertical = ti.Vector([0.0, 2.0, 0.0])
origin = ti.Vector([0.0, 0.0, 0.0])

#0-2代表球心,3代表半径
spheres = ti.Vector(4, dt=ti.f32, shape=2)
#两个球体
spheres[0] = np.array([0.0, 0.0, -1.0, 0.5], dtype=np.float32)
spheres[1] = np.array([0.0, -100.5, -1.0, 100.0], dtype = np.float32)

@ti.func
def unit_vector(v):
    k = 1 / (ti.sqrt(v.dot(v)))
    return k * v


# @ti.func
# def hit_sphere(center, radius, rayo, rayd):
#     l = center - rayo
#     l2 = l.dot(l)
#     r2 = radius * radius

#     rst = True
#     t = 0.0
#     # 计算投影长度
#     s = l.dot(rayd)
#     # 在射线方向上投影为负,说明射线方向是远离球体的,距离小于半径,说明起始点在球形外面
#     if s < 0 and l2 > r2:
#         rst = False
#     else:
#         # 射线到球体中心的距离平方
#         m2 = l2 - s * s
#         if m2 > r2:
#             rst = False
#         else:
#             q = ti.sqrt(r2 - m2)
#             if l2 > r2:
#                 t = s - q
#             else:
#                 t = s + q
#     return rst, t, rayo + t * rayd

@ti.func
def hit_sphere(center, radius, rayo, rayd):
    rst = False
    t = 0.0
    oc = rayo - center
    a = rayd.dot(rayd)
    b = 2.0 * oc.dot(rayd)
    c = oc.dot(oc) - radius * radius
    disc = b * b - 4.0 * a * c
    if disc >= 0:
        t = (-b - ti.sqrt(disc)) / (2.0 * a)
        if t > 0.001 and t < 1e10:
            rst = True
        else:
            t = (-b + ti.sqrt(disc)) / (2.0 * a)
            if t > 0.001:
                rst = True

    return rst, t, rayo + t * rayd

@ti.func
def hit_spheres(rayo, rayd):
    rst = False
    rt = 1e10
    normal = ti.Vector([0.0, 0.0, 0.0])
    for i in ti.static(range(2)):
        sc = ti.Vector([spheres[i][0], spheres[i][1], spheres[i][2]])
        r = spheres[i][3]
        hit, t, hitp = hit_sphere(sc, r, rayo, rayd)
        if hit and t < rt:
            rst = True
            rt = t
            normal = unit_vector(hitp - sc)
    return rst , normal


@ti.func
def color(o, d):
    unit_d = unit_vector(d)
    rst = ti.Vector([0.0, 0.0, 0.0])
    hit, n = hit_spheres(o, unit_d)
    if hit:
        rst = 0.5 * ti.Vector([n[0] + 1, n[1] + 1, n[2] + 1])
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


gui = ti.GUI("Ray5", (nx, ny))
for i in range(1000):
    paint()
    gui.set_image(pixels.to_numpy())
    gui.show()
