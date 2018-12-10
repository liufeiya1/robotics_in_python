"""
Object shape recognition with circle fitting
author: Atsushi Sakai (@Atsushi_twi)
"""

import matplotlib.pyplot as plt
import math
import random
import numpy as np

show_animation = True


def circle_fitting(x, y):

    sumx = sum(x) #∑xi
    sumy = sum(y) #∑yi
    sumx2 = sum([ix ** 2 for ix in x])#∑(xi)**2
    sumy2 = sum([iy ** 2 for iy in y])#∑(yi)**2
    sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])#∑(xi*yi)

    F = np.array([[sumx2, sumxy, sumx], #构造矩阵F
                  [sumxy, sumy2, sumy],
                  [sumx, sumy, len(x)]])

    G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],#列向量G

                  [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                  [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

    T = np.linalg.inv(F).dot(G) #F的逆矩阵×G矩阵  T向量中即 a,b,c

    cxe = float(T[0] / -2)
    cye = float(T[1] / -2)
    re = math.sqrt(cxe**2 + cye**2 - T[2])

    error = sum([np.hypot(cxe - ix, cye - iy) - re for (ix, iy) in zip(x, y)]) #即：函数f

    return (cxe, cye, re, error) 


def get_sample_points(cx, cy, cr, angle_reso):#取样点函数:会选择一个圆附近随机的样点。
    x, y, angle, r = [], [], [], []

    # points sampling
    for theta in np.arange(0.0, 2.0 * math.pi, angle_reso):
        nx = cx + cr * math.cos(theta)
        ny = cy + cr * math.sin(theta)
        nangle = math.atan2(ny, nx) #（nx,ny）与原点（0,0）连线和正x轴的夹角
        nr = math.hypot(nx, ny) * random.uniform(0.95, 1.05)

        x.append(nx)
        y.append(ny)
        angle.append(nangle)
        r.append(nr)

    # ray casting filter
    rx, ry = ray_casting_filter(x, y, angle, r, angle_reso)

    return rx, ry


def ray_casting_filter(xl, yl, thetal, rangel, angle_reso):#从一系列点过滤得到样点
    rx, ry = [], []
    rangedb = [float("inf") for _ in range(
        int(math.floor((math.pi * 2.0) / angle_reso)) + 1)]

    for i in range(len(thetal)):
        angleid = math.floor(thetal[i] / angle_reso)

        if rangedb[angleid] > rangel[i]:
            rangedb[angleid] = rangel[i]

    for i in range(len(rangedb)):
        t = i * angle_reso
        if rangedb[i] != float("inf"):
            rx.append(rangedb[i] * math.cos(t))
            ry.append(rangedb[i] * math.sin(t))

    return rx, ry


def plot_circle(x, y, size, color="-b"):
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + size * math.cos(np.deg2rad(d)) for d in deg]
    yl = [y + size * math.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)


def main():

    simtime = 15.0  # 做15+1张样图
    dt = 1.0  # 时间间隔

    cx = -2.0  # 初始实际圆横坐标
    cy = -8.0  #初始实际圆横坐标
    cr = 1.0  # 初始圆半径
    theta = np.deg2rad(30.0)  # 移动方向
    angle_reso = np.deg2rad(3.0)  

    time = 0.0
    while time <= simtime:
        time += dt

        cx += math.cos(theta)
        cy += math.cos(theta)

        x, y = get_sample_points(cx, cy, cr, angle_reso)#取得样点

        ex, ey, er, error = circle_fitting(x, y)#圆拟合
        print("Error:", error)

        if show_animation:
            plt.cla()
            plt.axis("equal")
            plt.plot(0.0, 0.0, "*r") #原点（0,0）用红色的‘*’表示
            plot_circle(cx, cy, cr)  #画出蓝色圆
            plt.plot(x, y, "xr")     #画出散点 ，并用'x'标记各点
            plot_circle(ex, ey, er, "-r")#画出拟合圆 默认颜色为-b
            plt.pause(dt)

    print("Done")


if __name__ == '__main__':
    main()
