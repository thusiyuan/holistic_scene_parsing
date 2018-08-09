"""
Created on Aug 27, 2017

@author: Siyuan Huang

visualize code for SUNRGBD
"""
import matplotlib.pyplot as plt
from random import random as rand
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

sunrgbd_cate_all_changed = ['__background__', 'laptop', 'paper', 'oven', 'keyboard', 'ottoman', 'chair', 'monitor', 'cup', 'tv',
'bench', 'board', 'stove', 'plate', 'fridge', 'desk', 'coffee_table', 'vanity', 'towel', 'sofa',
'bag', 'tray', 'rack', 'bulletin_board', 'picture', 'night_stand', 'computer', 'mirror',
'container', 'clock', 'stool', 'microwave', 'mug', 'back_pack', 'cubby', 'electric_fan', 'cart',
'sink', 'box', 'island', 'whiteboard', 'desktop', 'pillow', 'pot', 'urinal', 'dining_table',
'tv_stand', 'projector', 'curtain', 'door', 'shelf', 'sofa_chair', 'table', 'cabinet', 'telephone',
'bookshelf', 'blinds', 'thermos', 'stack_of_chairs', 'bed', 'books', 'bathtub', 'toilet', 'scanner',
'recycle_bin', 'endtable', 'glass', 'drawer', 'tissue', 'organizer', 'mouse', 'bowl', 'machine',
'lamp', 'book', 'speaker', 'poster', 'suits_case', 'blanket', 'dresser', 'plant', 'printer',
'garbage_bin', 'podium', 'blackboard', 'cloth', 'dresser_mirror', 'counter', 'flower_vase',
'person', 'switch', 'bottle', 'basket', 'painting', 'cpu']


def show_2dboxes(im, bdbs, scale=1.0):
    plt.cla()
    plt.axis('off')
    plt.imshow(im)
    for bdb in bdbs:
        bbox = np.array([bdb['x1'], bdb['y1'], bdb['x2'], bdb['y2']]) * scale
        color = (rand(), rand(), rand())
        rect = plt.Rectangle((bbox[0], bbox[1]), bbox[2] - bbox[0], bbox[3] - bbox[1], fill=False, edgecolor=color, linewidth=2.5)
        plt.gca().add_patch(rect)
        plt.gca().text(bbox[0], bbox[1], '{:s}'.format(bdb['classname']), bbox=dict(facecolor=color, alpha=0.5), fontsize=9, color='white')
    plt.show()
    return im


def show_3dpointcloud(im_rgb, im_depth, K, sparsity=40):
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    fig = plt.figure()
    invalid = (im_depth == 0)
    ax = Axes3D(fig)
    im_rgb = im_rgb.astype('double')
    m, n = im_depth.shape
    x, y = np.meshgrid(range(n), range(m))
    x = x[::sparsity, ::sparsity]
    y = y[::sparsity, ::sparsity]
    im_depth = im_depth[::sparsity, ::sparsity]
    x3 = (x + 1 - cx) * im_depth / fx
    y3 = (y + 1 - cy) * im_depth / fy
    z3 = im_depth
    # point3d = np.hstack((x3, z3, -y3))
    # point3d[invalid, :] = None
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            # print im_rgb[y[i, j], x[i, j], :]
            ax.scatter(x3[i, j], y3[i, j], z3[i, j], c='black', marker='o', s=3)
    # ax.scatter(x3, z3, -y3, c='r', marker='o')
    plt.show(block=False)


def show_3dpointcloud_aligned(im_rgb, im_depth, K, R_tilt, sparsity=40):
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    fig = plt.figure()
    invalid = (im_depth == 0)
    ax = Axes3D(fig)
    im_rgb = im_rgb.astype('double')
    m, n = im_depth.shape
    x, y = np.meshgrid(range(n), range(m))
    x = x[::sparsity, ::sparsity]
    y = y[::sparsity, ::sparsity]
    im_depth = im_depth[::sparsity, ::sparsity]
    x3 = (x + 1 - cx) * im_depth / fx
    y3 = (y + 1 - cy) * im_depth / fy
    z3 = im_depth
    # point3d = np.hstack((x3, z3, -y3))
    # point3d[invalid, :] = None
    for i in range(x3.shape[0]):
        for j in range(x3.shape[1]):
            # print im_rgb[y[i, j], x[i, j], :]
            new_coor = R_tilt.dot(np.array([x3[i, j], z3[i, j], -y3[i, j]]))
            ax.scatter(new_coor[0], new_coor[1], new_coor[2], c='black', marker='o', s=3)
            # ax.scatter(x3[i, j], z3[i, j], -y3[i, j], c=cm.ScalarMappable().to_rgba(im_rgb[y[i, j], x[i, j], :]), marker='o', s=5)
    # ax.scatter(x3, z3, -y3, c='r', marker='o')
    plt.show(block=False)


def show_2dcorner(im_rgb, gt3dcorners, K, R_ex, R_tilt, img_only=1):
    # gt_3dcorners_temp is the camera coor
    if gt3dcorners is None:
        return
    gt_3dcorners_temp = R_ex.T.dot(gt3dcorners)
    gt_3dcorners_temp = np.vstack((gt_3dcorners_temp[0, :], gt_3dcorners_temp[2, :], -gt_3dcorners_temp[1, :]))
    fb_index = (gt_3dcorners_temp[1, :] < 0) * -2 + 1
    gt2dcorners = K.dot(R_ex.T).dot(gt3dcorners)   # note the gt3dcorner
    gt2dcorners = gt2dcorners[:2, :] / gt2dcorners[2, :] / fb_index
    num_corner = gt2dcorners.shape[1] / 2
    plt.figure()
    plt.imshow(im_rgb)
    plt.plot(np.hstack((gt2dcorners[0, :num_corner], gt2dcorners[0, 0])), np.hstack((gt2dcorners[1, :num_corner], gt2dcorners[1, 0])), 'r')
    plt.plot(np.hstack((gt2dcorners[0, num_corner:], gt2dcorners[0, num_corner])), np.hstack((gt2dcorners[1, num_corner:], gt2dcorners[1, num_corner])), 'b')
    for i in range(num_corner):
        if fb_index[i] == -1:
            continue
        plt.plot(gt2dcorners[0, [i, i + num_corner]], gt2dcorners[1, [i, i+num_corner]], 'y')
    m, n = im_rgb.shape[:2]
    # plt.grid(True)
    # plt.xlim((1, n))
    # plt.ylim((1, m))
    axes = plt.gca()
    if img_only == 1:
        axes.set_xlim([1, n])
        axes.set_ylim([m, 1])
    plt.show()
    return plt
    # print gt2dcorners


def plot_world_point(ax, p1, p2, color='r-'):
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color)


def plot_cuboid(ax, p1, p2, p3, p4, p5, p6, p7, p8, color='r-'):
    plot_world_point(ax, p1, p2, color)
    plot_world_point(ax, p2, p3, color)
    plot_world_point(ax, p3, p4, color)
    plot_world_point(ax, p4, p1, color)
    plot_world_point(ax, p5, p6, color)
    plot_world_point(ax, p6, p7, color)
    plot_world_point(ax, p7, p8, color)
    plot_world_point(ax, p8, p5, color)
    plot_world_point(ax, p1, p5, color)
    plot_world_point(ax, p2, p6, color)
    plot_world_point(ax, p3, p7, color)
    plot_world_point(ax, p4, p8, color)
    return p1, p2, p3, p4, p5, p6, p7, p8


def plot_skeleton(ax, skeleton, color='r-'):
    index = [[3, 2], [2, 4], [2, 8], [2, 1], [1, 0], [0, 12], [0, 16], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
             [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]]
    for ind in index:
        ax.plot([skeleton[ind[0], 0], skeleton[ind[1], 0]], [skeleton[ind[0], 1], skeleton[ind[1], 1]], [skeleton[ind[0], 2], skeleton[ind[1], 2]], color)


def plot_skeleton_2d(ax, skeleton, color='r-'):
    index = [[3, 2], [2, 4], [2, 8], [2, 1], [1, 0], [0, 12], [0, 16], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10], [10, 11],
             [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]]
    for ind in index:
        ax.plot([skeleton[ind[0], 0], skeleton[ind[1], 0]], [skeleton[ind[0], 1], skeleton[ind[1], 1]], color)


def object_color(obj_id):
    obj_color = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
     "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f", "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
     "#8dd3c7", "#bebada", "#fb8072", "#80b1d3", "#fdb462", "#b3de69", "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5",
     "#ffed6f", "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00", "#a65628", "#f781bf", "#999999", "#621e15",
     "#e59076", "#128dcd", "#083c52", "#64c5f2", "#61afaf", "#0f7369", "#9c9da1", "#365e96", "#983334", "#77973d",
     "#5d437c", "#36869f", "#d1702f", "#8197c5", "#c47f80", "#acc484", "#9887b0", "#2d588a", "#58954c", "#e9a044",
     "#c12f32", "#723e77", "#7d807f", "#9c9ede", "#7375b5", "#4a5584", "#cedb9c", "#b5cf6b", "#8ca252", "#637939",
     "#e7cb94", "#e7ba52", "#bd9e39", "#8c6d31", "#e7969c", "#d6616b", "#ad494a", "#843c39", "#de9ed6", "#ce6dbd",
     "#a55194", "#7b4173", "#000000", "#0000FF"]
    return obj_color[obj_id]


def sunrgbd_object_color(obj_type):
    return object_color(sunrgbd_cate_all_changed.index(obj_type) - 2)







