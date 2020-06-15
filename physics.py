import taichi as ti
import numpy as np
from enum import Enum

ti.init(debug=True)

MAX_NUM_ITEMS = 256

DT = 1e-1

NUM_ITEMS = ti.var(ti.i32, shape=())

CANVAS_SIZE = (512, 512)
# 位置
P = ti.Vector(2, dt=ti.f32, shape=MAX_NUM_ITEMS)
# 旋转角度 分别是角度, cos, sin值
R = ti.Vector(3, dt=ti.f32, shape=MAX_NUM_ITEMS)
# 速度
V = ti.Vector(2, dt=ti.f32, shape=MAX_NUM_ITEMS)
# 类型 0:圆, 1:长方形
TYPES = ti.var(dt=ti.u8, shape=MAX_NUM_ITEMS)
# 参数, 对于圆是半径,对于长方形是长宽
PARAMS = ti.Vector(2, dt=ti.u16, shape=MAX_NUM_ITEMS)

# 质量的倒数, 0表示Static
IM = ti.var(dt=ti.f32, shape=MAX_NUM_ITEMS)
# 转动惯量倒数
II = ti.var(dt=ti.f32, shape=MAX_NUM_ITEMS)

GRAVITY = ti.Vector(2, dt=ti.f32, shape=())


@ti.kernel
def substep():
    n = NUM_ITEMS[None]
    for i in range(n):
        if IM[i] != 0:
            V[i] = V[i] + GRAVITY * DT

    for i in range(n):
        P[i] = P[i] + V[i] * DT

@ti.kernel
def init1():
    GRAVITY[None] = ti.Vector([0, -10])
    NUM_ITEMS[None] = 2
    P[0] = ti.Vector([200, 512])
    IM[0] = 1
    TYPES[0] = 0
    PARAMS[0] = ti.Vector([20, 0])

    P[1] = ti.Vector([300, 400])
    IM[1] = 0
    TYPES[1] = 1
    PARAMS[1] = ti.Vector([50, 20])

gui = ti.GUI('Physics', res=CANVAS_SIZE)

init1()
params = PARAMS.to_numpy()
types = TYPES.to_numpy()

while True:
    substep()
    x = P.to_numpy()
    for i in range(NUM_ITEMS.to_numpy()):
        if types[i] == 0:
            gui.circle(x[i] / CANVAS_SIZE, color=0xff0000, radius=params[i][0])
        else:
            wh = params[i]
            center = x[i]
            p0 = (center - wh) / CANVAS_SIZE
            p3 = (center + wh) / CANVAS_SIZE
            p1 = (p0[0], p3[1])
            p2 = (p3[0], p0[1])
            gui.triangle(p0, p1, p2)
            gui.triangle(p3, p1, p2)
    gui.show()
