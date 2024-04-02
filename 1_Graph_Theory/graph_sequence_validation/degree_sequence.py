# -*- coding: utf-8 -*-
# This module is a simple code for the "Graph Theory Assignment - Detection and Generation of Simple Graphs".
# Please do not use this module in production code.
# Created by XiangDong Yang on May 30th, 2023
# Copyright (c) 2023 yangxiangdong.cs@aliyun.com

import tkinter as tk
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def havel_hakimi(degrees):
    """
    根据 Havel-Hakimi 定理，判断给定的度序列是否可图
    :param degrees: 指定的度序列，使用一个 int 数组传入，度序列可以无序
    :return: 如果度序列可图，返回 True，否则返回 False
    """
    # 1. 对度序列进行降序排序
    degrees.sort(reverse=True)
    # 2. 循环处理度序列，直至达到算法结束标志
    while degrees:
        # 2.1. 如果度序列的第一个元素为 0，则其余元素也必须为 0
        if degrees[0] == 0:
            return True
        # 2.2. 移除并保存度序列的第一个元素
        d = degrees[0]
        degrees = degrees[1:]
        # 2.3. 检查度序列的长度是否小于首元素的值，不足则非可图序列
        if len(degrees) < d:
            return False
        # 2.4. 对剩余度序列的前 d 个元素执行减一操作
        degrees[:d] = [x - 1 for x in degrees[:d]]
        # 2.5. 度序列重排序
        degrees.sort(reverse=True)
        degrees = [x for x in degrees if x > 0]  # 移除末尾小于等于零的度
    # 3. 其余情况均返回可图序列
    return True


def rest_degree_sequence(degrees_total, degrees_used):
    """
    给定总的节点度序列，和已经使用掉的度序列，计算节点的剩余度序列，
    并将节点序号按照剩余度序列降序顺序排序并返回
    :param degrees_total: 每个节点总的度序列
    :param degrees_used: 每个节点已经使用掉的度序列，每增加 1 条边后，关联节点使用掉的度数都会增加 1
    :return: 节点序列，按照剩余度序列降序的顺序排序
    """
    degrees_rest = []
    for i in range(len(degrees_total)):
        degrees_rest.append(degrees_total[i] - degrees_used[i])
    indices = sorted(range(len(degrees_rest)), key=lambda idx: degrees_rest[idx], reverse=True)
    return indices


def draw_graph(degrees):
    """
    根据给定的度序列绘制简单图
    :param degrees: 指定的度序列，使用一个 int 数组传入，度序列可以无序
    :return: 如果度序列可图，则返回一个 networkx.Graph ，否则返回 None
    """
    # 0. 判断度序列是否可图
    if not havel_hakimi(degrees):
        print(f"The degree sequence {degrees} cannot be a valid graph")
        return None
    # 1. 创建一个空图
    graph = nx.Graph()
    # 2. 添加节点并设置节点的度数
    for i, deg in enumerate(degrees):
        graph.add_node(i, degree=deg)
    # 3. 遍历节点，根据度数创建边（算法核心）
    for node_idx in graph.nodes:
        # 3.1. 对剩余度数进行排序，优先使用剩余度数最多的节点进行边创建
        node_deg = graph.nodes[node_idx]['degree']
        degrees_order = rest_degree_sequence(degrees, graph.degree)
        # 3.2. 依次遍历其它节点，尝试与当前节点建立边
        for other_node_idx in degrees_order:
            other_node_deg = graph.nodes[other_node_idx]['degree']
            # 3.2.1. 当前 node 已经满足度数要求，终止
            if graph.degree[node_idx] == node_deg:
                break
            # 3.2.2. 当前 other_node 与 node 为同一个节点，跳过
            # 3.2.3. 当前 other_node 与 node 已经存在关联边，跳过
            if node_idx == other_node_idx or graph.has_edge(node_idx, other_node_idx):
                continue
            # 3.2.4. 当前 other_node 还有剩余度数，为 other_node 与 node 建立关联边
            if graph.degree[other_node_idx] < other_node_deg:
                graph.add_edge(node_idx, other_node_idx)
    # 4. 返回创建完成的图
    return graph


def print_result(param, tool):
    """
    打印程序运行结果到 GUI 组件中
    :param param: GUI 传入的参数，也就是度序列
    :param tool: GUI 组件，包含根节点和显示组件，其中显示组件用于显示错误信息或者生成的简单图
    :return: None
    """
    degrees = [int(i) for i in param.split()]
    graph = draw_graph(degrees)
    if tool.widget is not None:
        tool.widget.destroy()
    # 1. 可图判断
    if graph is None:
        tool.widget = tk.Label(tool.root, text=f"度序列{degrees}不是可图序列")
        tool.widget.pack()
        return None
    # 2. 绘图
    # 2.1. 创建一个 Matplotlib 图形
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_subplot(111)
    ax.axis('off')
    # 2.2. 绘制 NetworkX 图形到 Matplotlib
    pos = nx.spring_layout(graph)  # 选择布局算法
    nx.draw_networkx(graph, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray')
    # 2.3. 将 Matplotlib 嵌入到 Tkinter 窗口中
    canvas = FigureCanvasTkAgg(fig, master=tool.root)
    canvas.draw()
    tool.widget = canvas.get_tk_widget()
    tool.widget.pack()


class TkTool:
    def __init__(self, tk_root, tk_widget):
        self.root = tk_root
        self.widget = tk_widget


if __name__ == '__main__':
    root = tk.Tk()
    root.title("度序列可图判断/绘制")

    # 创建标签和输入框
    label1 = tk.Label(root, text="请输入度序列（空格分隔的整数，形如：2 2 2 2）=> ")
    label1.pack()
    entry1 = tk.Entry(root)
    entry1.pack()

    # 创建标签和按钮
    tk_tool = TkTool(root, None)
    button = tk.Button(root, text="绘制简单图", command=lambda: print_result(entry1.get(), tk_tool))
    button.pack()

    # 启动主循环
    root.mainloop()
