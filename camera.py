"""
Created on Aug 26, 2017

@author: Siyuan Huang

scripts about camera
"""

import numpy as np
from numpy.linalg import inv
import copy
from shapely.geometry.polygon import Polygon


#  counter-clockwise rotation about the z-axis:
def rotation_matrix_3d_z(angle):
    R = np.zeros((3, 3))
    R[0, 0] = np.cos(angle)
    R[1, 1] = np.cos(angle)
    R[0, 1] = -np.sin(angle)
    R[1, 0] = np.sin(angle)
    R[2, 2] = 1
    return R


def rotation_matrix_3d_x(angle):
    R = np.zeros((3, 3))
    R[1, 1] = np.cos(angle)
    R[2, 2] = np.cos(angle)
    R[1, 2] = -np.sin(angle)
    R[2, 1] = np.sin(angle)
    R[0, 0] = 1
    return R


# estimate the camera parameters
# vp-vanishing points
# h,w-height and width of the image
# K,R-Camera intrinsic and Rotation matrix
# check dhoiem.cs.illinois.edu/courses/vision_spring10/lectures/lecture3_projectivegeomtery.pdf for this algorithm
def calibrate_cam(vp, h, w):
    infchk = np.logical_and(vp[:, 0] > 50*w, vp[:, 1] > 50*h)
    if np.sum(infchk) == 0:
        v1 = vp[0, :]
        v2 = vp[1, :]
        v3 = vp[2, :]
        m_11 = v1[0] + v2[0]
        m_12 = v1[1] + v2[1]
        m_13 = v1[0]*v2[0] + v1[1]*v2[1]
        m_21 = v1[0] + v3[0]
        m_22 = v1[1] + v3[1]
        m_23 = v1[0]*v3[0] + v1[1]*v3[1]
        m_31 = v3[0] + v2[0]
        m_32 = v3[1] + v2[1]
        m_33 = v3[0]*v2[0] + v3[1]*v2[1]
        a_11 = m_11 - m_21
        a_12 = m_12 - m_22
        a_21 = m_11 - m_31
        a_22 = m_12 - m_32
        b_1 = m_13 - m_23
        b_2 = m_13 - m_33
        det_a = a_11*a_22 - a_12*a_21
        u0 = (a_22*b_1-a_12*b_2)/det_a
        v0 = (a_11*b_2-a_21*b_1)/det_a
        temp = m_11*u0 + m_12*v0 - m_13 - u0*u0 - v0*v0
        f = temp ** 0.5
    if np.sum(infchk) == 1:
        ii = np.nonzero(infchk == 0)
        v1 = vp[ii[0], :]
        v2 = vp[ii[1], :]
        r = ((w/2 - v1[0])*(v2[0] - v1[0])+(h/2 - v1[1])*(v2[1]-v1[1])) / ((v2[0] - v1[0]) ** 2 + (v2[1] - v1[1]) ** 2)
        u0 = v1[0] + r*(v2[0] - v1[0])
        v0 = v1[1] + r*(v2[1] - v1[1])
        temp = u0 * (v1[0] + v2[0]) + v0*(v2[1] + v1[1]) - (v1[0]*v2[0] + v2[1]*v1[1] + u0**2 + v0**2)
        f = temp ** 0.5
    if 'f' in locals() and 'u0' in locals() and 'v0' in locals():
        k = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
        vecx = inv(k).dot(np.hstack((vp[1, :], 1)).T)
        vecx /= np.linalg.norm(vecx)
        if vp[1, 0] < u0:
            vecx = -vecx
        vecz = inv(k).dot(np.hstack((vp[2, :], 1)).T)
        vecz = -vecz/np.linalg.norm(vecz)
        vecy = np.cross(vecz, vecx)
        r = np.vstack((vecx, vecy, vecz))
    if sum(infchk) == 2:
        if infchk[1] == 1:
            vp[1, :] = vp[1, :]/np.linalg.norm(vp[1, :])
        if infchk[0] == 1:
            vp[0, :] = vp[0, :]/np.linalg.norm(vp[0, :])
        if infchk[2] == 1:
            vp[2, :] = vp[2, :]/np.linalg.norm(vp[2, :])
        u0 = w/2
        v0 = h/2
        if infchk[1] == 1:
            vecx = np.hstack((vp[1, :], 0)).T
            vecx /= np.linalg.norm(vecx)
            if vp[1, 0] < u0:
                vecx = -vecx
        if infchk[0] == 1:
            vecy = np.hstack((vp[0, :], 0)).T
            vecy /= np.linalg.norm(vecy)
            if vp[0, 1] > v0:
                vecy = -vecy
        if infchk[2] == 1:
            vecz = np.hstack((vp[2, :], 0)).T
            vecz = -vecz/np.linalg.norm(vecz)
        if 'vecx' in locals() and 'vecy' in locals():
            vecz = np.cross(vecx, vecy)
        elif 'vecy' in locals() and 'vecz' in locals():
            vecx = np.cross(vecy, vecz)
        else:
            vecy = np.cross(vecz, vecx)
        r = np.vstack((vecx, vecy, vecz))
        f = 544
        k = np.array([[f, 0, u0], [0, f, v0], [0, 0, 1]])
    return k, r


# calculate the tilt according to R
def inverse_camera(R):
    # print R
    beta = np.arcsin(R[2, 1])
    return beta / np.pi * 180


# [X, Y, Z]^T = C + \lambda R^-1 K^-1 p
# get the world coordinates from the rgb coordinates given height
# C is the center of camera, p -> [u, v, 1]
def single_image_to_world(C, p, K, R, h):
    temp_foot = inv(R).dot(inv(K)).dot(p)
    temp_foot = temp_foot[[0, 2, 1]]
    temp_foot[2] *= -1
    scale = (h - C[2]) / temp_foot[2]
    x_foot = scale * temp_foot[0]
    y_foot = scale * temp_foot[1]
    z_foot = h
    return np.array([x_foot, y_foot, z_foot])


# get the world coordinates from the image pixel position and the depth
def rgbd_to_world(p, depth, K, R_tilt, R=None):
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]
    x = p[0]
    y = p[1]
    x3 = (x + 1 - cx) * depth / fx
    y3 = (y + 1 - cy) * depth / fy
    z3 = depth
    new_coor = R_tilt.T.dot(np.array([x3, z3, -y3]))   # camera to world so it's R_tilt
    # R.T.dot(np.array([x3, y3, z3]))[0, 2, -1]   # this is equal to the last sentence
    return new_coor


def fix_camera_focal_length(K):  # fix the wrong prediction of focal length
    if K[0, 0] > 694 or K[0, 0] < 500:  # fix the wrong prediction of focal length
        k_origin = K[0, 0]
        if K[0, 0] > 694:
            K[0, 0] = 694
            K[1, 1] = 694
        elif K[0, 0] < 510:
            K[0, 0] = 530
            K[1, 1] = 530
        print 'fix focal length from %d to %d' % (k_origin, K[0, 0])
    return K


def recover_rotate_angle(R):
    print 'hello'


# for leaning the sunrgbd data
def get_corners_of_bb3d(basis, coeffs, centroid):
    corners = np.zeros((8, 3))
    # order the basis
    index = np.argsort(np.abs(basis[:, 0]))[::-1]
    # the case that two same value appear the same time
    if index[2] != 2:
        index[1:] = index[1:][::-1]
    basis = basis[index, :]
    coeffs = coeffs[index]
    # Now, we know the basis vectors are orders X, Y, Z. Next, flip the basis vectors towards the viewer
    basis = flip_towards_viewer(basis, centroid)
    coeffs = np.abs(coeffs)
    corners[0, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[1, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[2, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]
    corners[3, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + basis[2, :] * coeffs[2]

    corners[4, :] = -basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[5, :] = basis[0, :] * coeffs[0] + basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[6, :] = basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners[7, :] = -basis[0, :] * coeffs[0] + -basis[1, :] * coeffs[1] + -basis[2, :] * coeffs[2]
    corners = corners + np.tile(centroid, (8, 1))
    return corners


def flip_towards_viewer(normals, points):
    points = points / np.linalg.norm(points)
    proj = points.dot(normals[:2, :].T)
    flip = np.where(proj > 0)
    normals[flip, :] = -normals[flip, :]
    return normals


# project 3d point to left hand coordinates
def project3dPtsToOpengl(point, R_z, T):
    # back to the original coordinates
    point = R_z.T.dot(point)
    point = point - T
    x = point[0]
    y = -point[2]
    z = point[1]
    # transfer to OPENGL coordinates
    R_x = rotation_matrix_3d_x(np.pi)
    return R_x.dot(np.array([x, y, z]))


def multi_project3dPtsToOpengl(point, R_z, T):
    point = R_z.T.dot(point.T).T
    point = point - np.tile(T, (point.shape[0], 1))
    point_temp = copy.copy(point)
    point[:, 1] = -point_temp[:, 2]
    point[:, 2] = point_temp[:, 1]
    # transform to OPENGL coordinates
    R_x = rotation_matrix_3d_x(np.pi)
    # print R.dot(point.T).sT
    return R_x.dot(point.T).T


def project3dvectorToOpengl(vector, R_z, R):
    vector = R_z.T.dot(vector)
    x = vector[0]
    y = -vector[2]
    z = vector[1]
    return R.dot(np.array([x, y, z]))


def rectangle_shrink(p1, p2, p3, p4, scale):
    center = (p1 + p3) / 2
    return (p1 - center) * scale + center, (p2 - center) * scale + center, (p3 - center) * scale + center, (p4 - center) * scale + center


# cu -> 8x3 numpy arrray
def intersection_cuboid(cu1, cu2):
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersection_2d = polygon_1.intersection(polygon_2).area
    if min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]) > 0:
        return (min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2])) * intersection_2d
    else:
        return 0

# return the intersection area of two cuboid in x-y coordinates / area of cuboid #1
def intersection_2d_ratio(cu1, cu2):
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersection_ratio = polygon_1.intersection(polygon_2).area / polygon_1.area
    return intersection_ratio

# cu1 -> cuboid of object cu2 -> cuboid of layout
def intersection_over_layout(cu1, cu2):
    polygon_1 = Polygon([(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])
    polygon_2 = Polygon([(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])
    intersection_2d = polygon_1.intersection(polygon_2).area
    if min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]) > 0:
        # return (polygon_1.area - intersection_2d) * (min(cu1[0][2], cu2[0][2]) - max(cu1[4][2], cu2[4][2]))
        return (polygon_1.area - intersection_2d) * 0.1
    else:
        # return polygon_1.area * (cu1[0][2] - cu1[4][2])
        return polygon_1.area * 0.1


# transfer the center-based cuboid to corner-based cuboid
def center_to_corners(center, size, angle):
    obj_angle = angle / 180 * np.pi
    basis_1 = np.array([np.cos(obj_angle), np.sin(obj_angle)])
    basis_2 = np.array([np.cos(obj_angle + np.pi / 2), np.sin(obj_angle + np.pi / 2)])
    basis = np.zeros((3, 3))
    basis[2] = np.array([0, 0, 1])
    basis[0, :2] = basis_1
    basis[1, :2] = basis_2
    p = get_corners_of_bb3d(basis, size / 2, center)
    return p


def main():
    # test code
    # vp = 1000 * np.array([[-0.4825, 4.0977], [-4.0542, -0.7856], [0.4458, 0.2102]])
    # calibrate_cam(vp, 530, 730)
    R = np.array([[0.99768, 0.067791, 0.0063], [-0.067791, 0.980573, 0.184069], [0.0063, -0.184069, 0.9882893]])
    # print R.T.dot(R)
    inverse_camera(R)

if __name__ == '__main__':
    main()