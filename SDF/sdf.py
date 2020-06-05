import taichi as ti

ti.init(arch=ti.gpu)

nx = 800
ny = 400
MAXSTEPS = 50
EPSILON = 0.01
pixels = ti.Vector(3, dt=ti.f32, shape=(nx, ny))
lower_left_corner =[-2.0, -1.0, 0.0]
horizontal = [4.0, 0.0, 0.0]
vertical = [0.0, 2.0, 0.0]
origin = [0.0, 0.0, 1.0]
light_pos = [1.0, 1.0, 2.0]
@ti.func
def length(v):
    return v.norm()

@ti.func
def reflect(v, n):
    return v - 2 * v.dot(n) * n

@ti.func
def sphereSDF(p, r):
    return length(p) - r

# https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
@ti.func
def boxSDF(p, b):
    q = ti.abs(p) - b
    return length(ti.max(q, 0.0)) + ti.min(q.max(), 0.0)


# CSG 交,两个值都是负的,说明点在相交区
@ti.func
def intersectSDF(distA, distB):
    return max(distA, distB)

# 并, 两个值其中一个为负的,在合并区
@ti.func
def unionSDF(distA, distB):
    return min(distA, distB)

#补, 在A的内部,B的外部
@ti.func
def differenceSDF(distA, distB):
    return max(distA, -distB)

# 根据梯度算法线
@ti.func
def estimateNormal(p, t):
    return ti.Vector([
        sceneSDF(ti.Vector([p[0] + EPSILON, p[1], p[2]]), t) - sceneSDF(ti.Vector([p[0] - EPSILON, p[1], p[2]]), t),
        sceneSDF(ti.Vector([p[0], p[1] + EPSILON, p[2]]), t) - sceneSDF(ti.Vector([p[0], p[1] - EPSILON, p[2]]), t),
        sceneSDF(ti.Vector([p[0], p[1], p[2] + EPSILON]), t) - sceneSDF(ti.Vector([p[0], p[1], p[2] - EPSILON]), t)
    ]).normalized()

@ti.func
def rotateY(thelta):
    c = ti.cos(thelta)
    s = ti.sin(thelta)
    return ti.Matrix([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c]
    ])

@ti.func
def sceneSDF(p, t):
    sphereD = sphereSDF(p, 0.3)
    # mat = rotateY(t).inverse()
    # applied_p = mat @ p
    cubeD = boxSDF(rotateY(-t) @ p , ti.Vector([0.2, 0.3, 0.3]))
    # return sphereD
    return intersectSDF(cubeD, sphereD)

@ti.func
def light(l, n, v, color):
    ambientStrength = 0.1
    # specularStrength = 100000
    lightColor = ti.Vector([1.0, 1.0, 1.0])

    #Ambient
    ambient = ambientStrength
    normalizel = l.normalized()
    # normalizev = v.normalized()
    #Diffuse
    diffuse = max(n.dot(normalizel), 0.0)
    # reflected = reflect(-normalizel, normalizev)
    # Specular
    # spec = ti.pow(max(normalizev.dot(reflected), 0.0), 32)
    # specular = specularStrength * spec
    return (ambient + diffuse) * lightColor * color

@ti.func
def color(o, d, t):
    rst = ti.Vector([0.0, 0.0, 0.0])
    depth = 0.0
    dist = 0.0
    for _ in range(MAXSTEPS):
        next_p = o + d * depth
        dist = sceneSDF(next_p, t)
        if dist < EPSILON:
            normal = estimateNormal(next_p, t)
            rst = light(ti.Vector(light_pos) - next_p, normal, ti.Vector(origin) - next_p, ti.Vector([0.5, 0.7, 0.9]))
            # rst = ti.Vector([0.0, 1.0, 1.0])
            break
        depth += dist
        if depth >= 100:
            break
    return rst


@ti.kernel
def paint(t: ti.f32):
    for i, j in pixels:
        u = float(i) / float(nx)
        v = float(j) / float(ny)
        direction = ( ti.Vector(lower_left_corner) + u *  ti.Vector(horizontal) + v *  ti.Vector(vertical) -  ti.Vector(origin)).normalized()
        pixels[i, j] = color( ti.Vector(origin), direction, t)


gui = ti.GUI("RayMarching", (nx, ny))
for i in range(1000):
    paint(i * 0.03)
    gui.set_image(pixels.to_numpy())
    gui.show()

