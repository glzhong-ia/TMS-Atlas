# -*- coding: utf-8 -*-\
import simnibs
import numpy as np
import math
from simnibs import sim_struct
from nelder_mead import NelderMead
from pathlib import Path

# 目标场点
target_point = [-37.0, -7.5, 36.7]

# 历史值
best_history = {}

# 保存历史记录最大值
max_history = 10

# 迭代次数记录
iter_count = 1


# 计算目标场点出的电场Ex, Ey, Ez
# coord:目标场点坐标
def get_E(msh, coord):
    cha = msh.nodes.node_coord - np.array(coord)
    E = []
    dis = cha[:, 0] * cha[:, 0] \
          + cha[:, 1] * cha[:, 1] \
          + cha[:, 2] * cha[:, 2]
    nr = msh.nodes.node_number[np.argmin(dis)]
    elm_lst = msh.elm.find_all_elements_with_node(nr)
    node_lst = []
    rows = elm_lst.shape
    for i in range(rows[0] - 1):
        node_index = msh.elm.node_number_list[int(elm_lst[i] - 1), :]
        elm_type = msh.elm.elm_type[int(elm_lst[i] - 1)]
        if elm_type == 2:  # 三角形
            node_lst.append([msh.nodes.node_coord[node_index[0] - 1],
                             msh.nodes.node_coord[node_index[1] - 1],
                             msh.nodes.node_coord[node_index[2] - 1],
                             [-1, -1, -1]])
        elif elm_type == 4:  # 四面体
            tetrahedra = [msh.nodes.node_coord[node_index[0] - 1],
                          msh.nodes.node_coord[node_index[1] - 1],
                          msh.nodes.node_coord[node_index[2] - 1],
                          msh.nodes.node_coord[node_index[3] - 1]]
            D0 = np.array([[tetrahedra[0][0], tetrahedra[0][1], tetrahedra[0][2], 1],
                           [tetrahedra[1][0], tetrahedra[1][1], tetrahedra[1][2], 1],
                           [tetrahedra[2][0], tetrahedra[2][1], tetrahedra[2][2], 1],
                           [tetrahedra[3][0], tetrahedra[3][1], tetrahedra[3][2], 1]])
            D1 = np.array([[coord[0], coord[1], coord[2], 1],
                           [tetrahedra[1][0], tetrahedra[1][1], tetrahedra[1][2], 1],
                           [tetrahedra[2][0], tetrahedra[2][1], tetrahedra[2][2], 1],
                           [tetrahedra[3][0], tetrahedra[3][1], tetrahedra[3][2], 1]])
            D2 = np.array([[tetrahedra[0][0], tetrahedra[0][1], tetrahedra[0][2], 1],
                           [coord[0], coord[1], coord[2], 1],
                           [tetrahedra[2][0], tetrahedra[2][1], tetrahedra[2][2], 1],
                           [tetrahedra[3][0], tetrahedra[3][1], tetrahedra[3][2], 1]])
            D3 = np.array([[tetrahedra[0][0], tetrahedra[0][1], tetrahedra[0][2], 1],
                           [tetrahedra[1][0], tetrahedra[1][1], tetrahedra[1][2], 1],
                           [coord[0], coord[1], coord[2], 1],
                           [tetrahedra[3][0], tetrahedra[3][1], tetrahedra[3][2], 1]])
            D4 = np.array([[tetrahedra[0][0], tetrahedra[0][1], tetrahedra[0][2], 1],
                           [tetrahedra[1][0], tetrahedra[1][1], tetrahedra[1][2], 1],
                           [tetrahedra[2][0], tetrahedra[2][1], tetrahedra[2][2], 1],
                           [coord[0], coord[1], coord[2], 1]])
            det0 = np.linalg.det(D0)
            det1 = np.linalg.det(D1)
            det2 = np.linalg.det(D2)
            det3 = np.linalg.det(D3)
            det4 = np.linalg.det(D4)
            if (int(np.sign(det0)) == 1 and int(np.sign(det1)) == 1
                and int(np.sign(det2)) == 1 and int(np.sign(det3)) == 1
                and int(np.sign(det4)) == 1) or (int(np.sign(det0)) == -1 and int(np.sign(det1)) == -1
                                                 and int(np.sign(det2)) == -1 and int(np.sign(det3)) == -1
                                                 and int(np.sign(det4)) == -1):
                node_lst.append([msh.nodes.node_coord[node_index[0] - 1],
                                 msh.nodes.node_coord[node_index[1] - 1],
                                 msh.nodes.node_coord[node_index[2] - 1],
                                 msh.nodes.node_coord[node_index[3] - 1]])
                E = msh.elmdata[0][node_index[1] - 1]
    return E


# coil_center：线圈中心坐标
# coil_direction:线圈的y轴是把手的延伸的坐标，这里规定coil_direction必须在以coil_center为中心的半径10mm的球形表面
# dir_name: 保存路径
# cpus: 使用几个cpu进行计算
def createSim(coil_center, coil_direction, dir_name, cpus=1):
    # General Infoarmation
    S = sim_struct.SESSION()
    abPath = '/home/wq/下载/simnibs_examples/ernie/'
    S.fnamehead = abPath + 'ernie.msh'  # head mesh
    S.pathfem = dir_name  # Directory for the simulation
    # 定义线圈类型
    tms = S.add_tmslist()
    tms.fnamecoil = 'Magstim_70mm_Fig8.nii.gz'  # Choose a coil from the ccd-files folder

    # 定义线圈位置
    pos = tms.add_position()
    # Place coil over the hand knob
    # Here, the hand knob is defined in MNI coordinates (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2034289/)
    # And transformed to subject coordinates
    # We can use positions in the cortex. SimNIBS will automatically project them to the skin surface
    # and add the specified distance
    pos.centre = simnibs.mni2subject_coords(coil_center, abPath + 'm2m_ernie')
    # Point the coil handle posteriorly, we just add 10 mm to the original M1 y coordinate
    pos.pos_ydir = simnibs.mni2subject_coords(coil_direction, abPath + 'm2m_ernie')
    pos.distance = 4  # 4 mm distance from coil surface to head surface

    # Run Simulation
    out = S.run(cpus, False, True)
    re = simnibs.read_msh(S.pathfem + '/ernie_TMS_1-0001_Magstim_70mm_Fig8_nii_scalar.msh')
    return re


# 为了便于简化寻优约束条件，规定线圈的方向可以在3个自由度(球坐标系r=10mm, theata(0-360), fai(-180-180))内+-180度旋转
# 但是由于仿真程序的限制，需要将线圈方向转换为一个坐标值，在本程序中，以线圈中心为坐标原点，x,y,z轴的方向与世界坐标系完全相同建立球坐标系
# 本函数的作用为将球坐标系值转换为直角坐标系，得到的结果与世界坐标系下的线圈中心坐标相加，即可得到世界坐标系下的线圈y轴延伸方向坐标值
# direction: 球坐标系值(角度值)
def direction2coord(direction):
    x = direction[0] * math.sin(direction[1] * math.pi / 180.0) * math.cos(direction[2] * math.pi / 180.0)
    y = direction[0] * math.sin(direction[1] * math.pi / 180.0) * math.sin(direction[2] * math.pi / 180.0)
    z = direction[0] * math.cos(direction[1] * math.pi / 180.0)
    return [x, y, z]


# 目标函数， E的模值的倒数
# x[0:2] 线圈中心坐标x,y,z
# x[3:4] 线圈方向球坐标theata, fai
def func(x):
    global iter_count
    global best_history
    global target_point
    coil_center = x[0:3]
    coil_direction = direction2coord([10.0, x[3], x[4]])
    coil_direction[0] = coil_direction[0] + coil_center[0]
    coil_direction[1] = coil_direction[1] + coil_center[1]
    coil_direction[2] = coil_direction[2] + coil_center[2]
    dir_name = 'tms_hand_' + str(iter_count)
    # 删除文件夹
    dname = Path('./' + dir_name)
    if dname.is_dir():
        dname.rmdir()
    msh = createSim(coil_center, coil_direction, dir_name, 1)
    e = get_E(msh, target_point)
    result = 1.0 / math.sqrt(e[0] * e[0] + e[1] * e[1] + e[2] * e[2])
    if len(best_history) < max_history:
        best_history[dir_name] = result
    else:
        temp = sorted(best_history.items(), key=lambda item: item[1])
        if result < temp[-1][1]:
            best_history.pop(temp[-1][0])
            best_history[dir_name] = result
            dname = Path('./' + temp[-1][0])
            if dname.is_dir():
                dname.rmdir()

    iter_count = iter_count + 1
    return result


# 主函数
def main():
    # 优化目标
    # cx cy cz为线圈中心在世界坐标系下的坐标，后面为起至范围
    # ct cf为以线圈中心点为原点，建立与世界坐标系平行的球坐标系下线圈手柄延伸方向的坐标(10, ct, cf)
    params = {
        "cx": ["real", (-37, 37)],
        "cy": ["real", (-37, 37)],
        "cz": ["real", (-37, 37)],
        "ct": ["real", (0, 360)],
        "cf": ["real", (-180, 180)],
    }

    nm = NelderMead(func, params)
    # 最大迭代次数
    nm.minimize(n_iter=30)
    for key in best_history.keys():
        print('dir: ' + key + ' targetValue: ' + best_history[key])

if __name__ == "__main__":
    main()
