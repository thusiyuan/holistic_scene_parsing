"""
Created on Oct 10, 2017

@author: Siyuan Huang

Inference process for generating parse graph

"""

import config
import os
import shutil
import copy
from camera import multi_project3dPtsToOpengl, project3dPtsToOpengl, intersection_cuboid, center_to_corners, intersection_over_layout, intersection_2d_ratio, rotation_matrix_3d_z
import numpy as np
import sys
from osmesa.render_scene import render_scene, read_metadata, alignment_check
from sklearn.metrics import mean_squared_error
import random
import pickle
from mcmc import metropolis_hasting
from object_loader import OBJ_SAVER_SIMPLIFIED
import scipy.io
import time
import json
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from camera import rectangle_shrink
from sample_human import HumanSample

paths = config.Paths()
metadata_root = paths.metadata_root
proposal_root = paths.proposal_root
inference_root = os.path.join(proposal_root, 'inference')
stats_root = os.path.join(metadata_root, 'sunrgbd_stats')
save_root = 'result'


class Inference(object):
    def __init__(self, pg, est_depth, est_seg, est_normal, lambda_depth, lambda_seg, lambda_normal, save_path):
        self.pg = pg
        self.est_depth = est_depth
        self.est_seg = est_seg
        self.est_normal = est_normal
        self.lambda_depth = lambda_depth
        self.lambda_seg = lambda_seg
        self.lambda_normal = lambda_normal
        self.save_path = save_path
        self.energy_landscape = list()
        if pg.objects == None:
            self.num_object = 0
            pg._objects = list()
        else:
            self.num_object = len(pg.objects)
        self.obj_info = list()
        self.dataset_path = 'data'
        self.model_path = os.path.join(self.dataset_path, 'models_scaled')
        self.scale = 1.0
        self.normal_scale = 1.0
        self.record = list()
        self.inference_step = 0
        self.count_prior = False
        self.bool_intersection = np.zeros((self.num_object, 1))
        with open(os.path.join(stats_root, 'size', 'size_sampler.pickle'), 'rb') as f:
            self.size_sampler = pickle.load(f)
        f.close()
        with open(os.path.join(stats_root, 'size', 'size_mean_cov.pickle'), 'rb') as f:
            self.size_mean_cov = pickle.load(f)
        # init the object size of terminal nodes

    # Transfer the information in all the terminal nodes to OpenGL coordinates
    @staticmethod
    def pg_to_opengl(pg):
        pg = copy.deepcopy(pg)
        K = pg.camera.K
        for obj_index, object in enumerate(pg.objects):
            if object.terminal is None:
                continue
            new_center = project3dPtsToOpengl(object.terminal.obj_center, pg.layouts.R_C, pg.layouts.T_C).T
            object.terminal.set_center(new_center)
            # change size to opengl coordinates
            object.terminal.set_size(object.terminal.obj_size[np.array([0, 2, 1])])
        floor = multi_project3dPtsToOpengl(np.array(pg.layouts.floor), pg.layouts.R_C, pg.layouts.T_C)
        pg.layouts.set_floor(floor)
        ceiling = multi_project3dPtsToOpengl(np.array(pg.layouts.ceiling), pg.layouts.R_C, pg.layouts.T_C)
        pg.layouts.set_ceiling(ceiling)
        mwall = multi_project3dPtsToOpengl(np.array(pg.layouts.mwall), pg.layouts.R_C, pg.layouts.T_C)
        pg.layouts.set_mwall(mwall)
        lwall = multi_project3dPtsToOpengl(np.array(pg.layouts.lwall), pg.layouts.R_C, pg.layouts.T_C)
        pg.layouts.set_lwall(lwall)
        rwall = multi_project3dPtsToOpengl(np.array(pg.layouts.rwall), pg.layouts.R_C, pg.layouts.T_C)
        pg.layouts.set_rwall(rwall)
        return pg

    def load_obj(self, pg):
        pg = copy.deepcopy(pg)
        print 'loading CAD model'
        for i in range(self.num_object):
            for j in range(len(pg.objects[i].proposals)):
                pg.objects[i].set_terminal(pg.objects[i].proposals[j])
                obj_id = pg.objects[i].terminal.obj_id
                row = read_metadata(obj_id)
                if row is None:
                    continue
                up_string = row[4]
                front_string = row[5]
                unit = row[6]
                if unit == "":
                    print 'no unit'
                    continue
                if not alignment_check(up_string, front_string):
                    continue
                print obj_id
                filename = os.path.join(self.model_path, obj_id + '.obj')
                info = OBJ_SAVER_SIMPLIFIED(filename)
                self.obj_info.append({'info': info})
                print '{} exists and align success'.format(pg.objects[i].obj_type)
                break
        print 'CAD model loaded successfully'
        layout_type = 'mwall'
        filename = os.path.join(self.model_path, 'suncgwall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'floor'
        filename = os.path.join(self.model_path, 'suncgfloor.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'ceiling'
        filename = os.path.join(self.model_path, 'suncgfloor.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'lwall'
        filename = os.path.join(self.model_path, 'suncgsidewall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'rwacll'
        filename = os.path.join(self.model_path, 'suncgsidewall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})
        return pg

    def load_result(self, pg):
        pg = copy.deepcopy(pg)
        print 'loading CAD model'
        for i in range(self.num_object):
            obj_id = pg.objects[i].terminal.obj_id
            filename = os.path.join(self.model_path, obj_id + '.obj')
            info = OBJ_SAVER_SIMPLIFIED(filename)
            self.obj_info.append({'info': info})
            print '{} exists and align success'.format(pg.objects[i].obj_type)
        print 'loading CAD model successfully'
        layout_type = 'mwall'
        filename = os.path.join(self.model_path, 'suncgwall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'floor'
        filename = os.path.join(self.model_path, 'suncgfloor.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'ceiling'
        filename = os.path.join(self.model_path, 'suncgfloor.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'lwall'
        filename = os.path.join(self.model_path, 'suncgsidewall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})

        layout_type = 'rwacll'
        filename = os.path.join(self.model_path, 'suncgsidewall.obj')
        info = OBJ_SAVER_SIMPLIFIED(filename, layout_type=layout_type)
        self.obj_info.append({'info': info})
        return pg

    def one_step_support_initial(self, pg_ori, e_total_ori, obj_index):
        pg = copy.deepcopy(pg_ori)
        with open(os.path.join(stats_root, 'support', 'support_relation.json'), 'r') as f:
            support_relation = json.load(f)
        f.close()
        obj_type = pg.objects[obj_index].obj_type
        if obj_type == 'vanity':
            supporting = support_relation['bathroomvanity']
        else:
            supporting = support_relation[obj_type]
        num_total = 0
        for _, value in supporting.iteritems():
            num_total += value
        prob_max = 0
        prob_max_type = None
        for key, value in supporting.iteritems():
            supporting[key] = value / float(num_total)
            if supporting[key] > prob_max:
                prob_max = supporting[key]
                prob_max_type = key
        distance_floor = pg.objects[obj_index].terminal.obj_center[2] - pg.layouts.floor[0][2]
        if prob_max_type == 'wall' or distance_floor > 2.0:
            print '{} is supported by wall'.format(obj_type)
            pg.objects[obj_index]._supported_obj = 'wall'
            return pg, e_total_ori
        support_energy = list()
        for j in range(self.num_object):
            if obj_type == pg.objects[j].obj_type:
                support_energy.append(0)
            elif pg.objects[j].terminal.obj_center[2] > pg.objects[obj_index].terminal.obj_center[2]:
                support_energy.append(0)
            else:
                if pg.objects[j].obj_type not in supporting.keys() or supporting[pg.objects[j].obj_type] < 1 / 10.0:
                    prior_energy = 1 / 10.0
                else:
                    prior_energy = supporting[pg.objects[j].obj_type]
                cu1 = center_to_corners(pg.objects[j].terminal.obj_center, pg.objects[j].terminal.obj_size, pg.objects[j].terminal.angle)
                cu2 = center_to_corners(pg.objects[obj_index].terminal.obj_center, pg.objects[obj_index].terminal.obj_size, pg.objects[obj_index].terminal.angle)
                area_intersect_ratio = 1 - intersection_2d_ratio(cu1, cu2)
                height_distance = np.abs(pg.objects[j].terminal.obj_center[2] + pg.objects[j].terminal.obj_size[2] / 2 - (pg.objects[obj_index].terminal.obj_center[2] - pg.objects[obj_index].terminal.obj_size[2] / 2))
                likelihood_energy = np.exp(-height_distance*3-area_intersect_ratio)
                print pg.objects[obj_index].obj_type, pg.objects[j].obj_type, height_distance, area_intersect_ratio
                support_energy.append(prior_energy * likelihood_energy)

        # compute the supporting relation with the floor
        prior_energy = supporting['floor']
        likelihood_energy = np.exp(-distance_floor*3)
        support_energy.append(prior_energy * likelihood_energy)

        # compute the supporting relation with the wall
        supported_index = support_energy.index(max(support_energy))
        if supported_index != len(support_energy) - 1:
            pg.objects[obj_index]._supported_obj = pg.objects[supported_index].obj_type
            pg.objects[supported_index]._supporting_obj = pg.objects[obj_index].obj_type
            height_distance = pg.objects[obj_index].terminal.obj_center[2] - pg.objects[supported_index].terminal.obj_center[2]
            move_dis = height_distance - (pg.objects[obj_index].terminal.obj_size[2] +
                                          pg.objects[supported_index].terminal.obj_size[2]) / 2
            print move_dis
            pg.objects[obj_index].terminal.move_downface(move_dis / 2)
            pg.objects[supported_index].terminal.move_upface(move_dis / 2)
            print '{} is supported by {}'.format(obj_type, pg.objects[supported_index].obj_type)
        else:
            pg.objects[obj_index]._supported_obj = 'floor'
            print '{} is supported by floor'.format(obj_type)
            print pg.objects[obj_index].terminal.obj_center[2], pg.layouts.floor[0][2], pg.objects[obj_index].terminal.obj_size[2] / 2
            height_distance = pg.objects[obj_index].terminal.obj_center[2] - pg.layouts.floor[0][2]
            move_dis = height_distance - pg.objects[obj_index].terminal.obj_size[2] / 2
            print move_dis
            pg.objects[obj_index].terminal.move_downface(move_dis)
        e_total = self.compute_total_likelihood(pg, if_vis=True) + self.compute_prior(pg)
        return pg, e_total

    def support_initial(self, pg_ori, e_total_ori):
        for i in range(self.num_object):
            pg_ori, e_total_ori = self.one_step_support_initial(pg_ori, e_total_ori, i)
        return pg_ori, e_total_ori

    def record_accept(self):
        accepted_times = len(self.energy_landscape)
        shutil.copy(os.path.join(self.save_path, 'normal.png'), os.path.join(self.save_path, 'normal', 'normal_accepted_' + str(accepted_times) + '.png'))
        shutil.copy(os.path.join(self.save_path, 'depth.png'), os.path.join(self.save_path, 'depth', 'depth_accepted_' + str(accepted_times) + '.png'))
        shutil.copy(os.path.join(self.save_path, 'segmentation.png'), os.path.join(self.save_path, 'seg', 'seg_accepted_' + str(accepted_times) + '.png'))

    # compute total likelihood, return energy
    def compute_total_likelihood(self, pg, show_energy=False, if_vis=True):
        self.inference_step += 1
        render_depth = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, if_vis=if_vis, render_type='depth', obj_info=self.obj_info)
        render_seg = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, if_vis=if_vis, render_type='segmentation', obj_info=self.obj_info)
        render_normal = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, if_vis=if_vis, render_type='normal', obj_info=self.obj_info)
        render_normal = render_normal / 255.0 * 2 - 1
        e_normal = self.compute_normal_error(render_normal)
        e_depth = self.compute_depth_error(render_depth)
        e_seg = self.compute_seg_error(render_seg)
        e_total = e_depth + e_seg + e_normal
        if show_energy:
            print 'e_total is :{}. e_depth is :{}. e_seg is :{}. e_normal is :{}'.format(e_total, e_depth, e_seg, e_normal)
        return e_total

    def compute_depth_error(self, depth):
        depth_error_map = mean_squared_error(self.est_depth, depth)
        fig = plt.figure()
        ii = plt.imshow(np.abs(self.est_depth - depth), interpolation='nearest')
        fig.colorbar(ii)
        plt.savefig(os.path.join(self.save_path, 'depth_error.png'))
        plt.close()
        return self.lambda_depth * depth_error_map

    def compute_seg_error(self, seg):
        m, n = seg.shape[:2]
        seg_error_map = self.est_seg != seg
        scipy.io.savemat(os.path.join(self.save_path, 'seg_error.mat'), mdict={'seg': seg_error_map})
        return self.lambda_seg * np.sum(seg_error_map) / float(m * n)

    def compute_normal_error(self, normal):
        m, n = normal.shape[:2]
        normal_error_map = self.est_normal - normal
        scipy.io.savemat(os.path.join(self.save_path, 'normal_error.mat'), mdict={'normal': normal_error_map})
        return self.lambda_normal * np.sum(normal_error_map ** 2) / (m * n)

    # compute total prior, return energy
    def compute_prior(self, pg):
        if self.count_prior:
            # prior for the size of the object
            pg = copy.deepcopy(pg)
            size_prior = 0
            area_conf = 0
            for index in range(self.num_object):
                size = copy.deepcopy(pg.objects[index].terminal.obj_size) / 2
                obj_type = copy.deepcopy(pg.objects[index].obj_type)
                mean, cov = self.size_mean_cov[obj_type]
                for i in range(3):
                    energy = np.abs(size[i] - mean[i]) / mean[i] * 0.1
                    size_prior += energy
            self.bool_intersection = np.zeros((self.num_object, 1))
            for i in range(self.num_object - 1):
                for j in range(i + 1, self.num_object):
                    obj1 = pg.objects[i].terminal
                    obj2 = pg.objects[j].terminal
                    cu1 = center_to_corners(obj1.obj_center, obj1.obj_size, obj1.angle)
                    cu2 = center_to_corners(obj2.obj_center, obj2.obj_size, obj2.angle)
                    area_intersect = intersection_cuboid(cu1, cu2)
                    if area_intersect > 0:
                        print 'area of intersection between {} and {} is: {}'.format(pg.objects[i].obj_type,
                                                                                     pg.objects[j].obj_type,
                                                                                     area_intersect)
                        if area_intersect > 1e-6:
                            self.bool_intersection[i] = 1
                            self.bool_intersection[j] = 1
                        area_conf += area_intersect
            for i in range(self.num_object):
                cu1 = center_to_corners(pg.objects[i].terminal.obj_center, pg.objects[i].terminal.obj_size,
                                        pg.objects[i].terminal.angle)
                cu2 = np.array(pg.layouts.corners)[[4, 5, 6, 7, 0, 1, 2, 3], :]
                area_intersect_over_layout = intersection_over_layout(cu1, cu2)
                if area_intersect_over_layout > 0:
                    print '{} out of the layout with {}'.format(pg.objects[i].obj_type, area_intersect_over_layout)
                    area_conf += area_intersect_over_layout
                    if area_intersect_over_layout > 1e-6:
                        self.bool_intersection[i] = 1
            return area_conf + size_prior
        else:
            return 0

    # compute depth posterior according to the parse graph
    def depth_likelihood(self, pg):
        depth = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, render_type='depth', obj_info=self.obj_info)
        depth_like = self.compute_depth_error(depth)
        return depth_like

    # compute seg posterior according to the parse graph
    def seg_likelihood(self, pg):
        seg = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, render_type='segmentation', obj_info=self.obj_info)
        seg_like = self.compute_seg_error(seg)
        return seg_like

    # compute normal posterior according to the parse graph
    def normal_likelihood(self, pg):
        normal = render_scene(pg=self.pg_to_opengl(pg), save_path=self.save_path, render_type='normal', obj_info=self.obj_info)
        normal = normal / 255.0 * 2 - 1
        normal_like = self.compute_normal_error(normal)
        return normal_like

    # propose the moving method
    def q_moving_proposal(self):
        r = random.random()
        if r < 0.95:
            return 0, 0.95   # propose gradient descent algorithm
        if 0.95 <= r <= 1:
            return 1, 0.05   # propose gradient ascent algorithm

    def one_step_layout_moving(self, pg_ori, e_total_ori, T, move_type):
        print 'adjust {} with T = {} and scale = {}'.format(move_type, T, self.scale)
        corner_ori = pg_ori.layouts.corners
        corner_des = copy.deepcopy(corner_ori)
        delta = 0.05 * self.scale
        rotate_angle = 5.625 / 180 * np.pi * self.scale
        # in our normal coordinates,
        if move_type == 'mwall':    # move mwall
            for i in [0, 1, 5, 4]:
                corner_des[i] += [0, delta, 0]
        elif move_type == 'lwall':
            for i in [1, 2, 6, 5]:
                corner_des[i] += [delta, 0, 0]
        elif move_type == 'rwall':
            for i in [0, 3, 7, 4]:
                corner_des[i] += [delta, 0, 0]
        elif move_type == 'floor':
            for i in [0, 1, 2, 3]:
                corner_des[i] += [0, 0, delta]
        elif move_type == 'ceiling':
            for i in [4, 5, 6, 7]:
                corner_des[i] += [0, 0, delta]
        elif move_type == 'rotate':
            layout_center = 0.5 * (corner_des[0] + corner_des[2])
            layout_center[2] = 0
            corner_des = rotation_matrix_3d_z(rotate_angle).dot(np.array(corner_des - layout_center).T).T + layout_center
        pg_des = copy.deepcopy(pg_ori)
        pg_des.layouts.set_corners(corner_des)
        e_total_des = self.compute_total_likelihood(pg_des) + self.compute_prior(pg_des)
        gradient = (e_total_ori - e_total_des) / np.abs(delta)
        gradient_type, move_prob = self.q_moving_proposal()
        # generate pg_new
        if move_type == 'ceiling':
            move_scale = 0.5 * self.scale
        elif move_type == 'floor':
            move_scale = 0.1 * self.scale
        else:
            move_scale = 0.3 * self.scale
        if gradient_type == 0:  # do gradient descent
            move_dis = gradient * move_scale
            rotate_angle *= np.sign(gradient)
        else:
            move_dis = - gradient * move_scale
            rotate_angle *= -np.sign(gradient)
        # avoid too large gradient
        if np.abs(move_dis) > 0.5:
            move_dis = 0.2
        corner_new = copy.deepcopy(pg_ori.layouts.corners)
        if move_type == 'mwall':    # move mwall
            for i in [0, 1, 5, 4]:
                corner_new[i] += [0, move_dis, 0]
        elif move_type == 'lwall':
            for i in [1, 2, 6, 5]:
                corner_new[i] += [move_dis, 0, 0]
        elif move_type == 'rwall':
            for i in [0, 3, 7, 4]:
                corner_new[i] += [move_dis, 0, 0]
        elif move_type == 'floor':
            for i in [0, 1, 2, 3]:
                corner_new[i] += [0, 0, move_dis]
        elif move_type == 'ceiling':
            for i in [4, 5, 6, 7]:
                corner_new[i] += [0, 0, move_dis]
        elif move_type == 'rotate':
            layout_center = 0.5 * (corner_new[0] + corner_new[2])
            layout_center[2] = 0
            corner_new = rotation_matrix_3d_z(rotate_angle).dot(np.array(corner_new - layout_center).T).T + layout_center
        pg_new = copy.deepcopy(pg_ori)
        pg_new.layouts.set_corners(corner_new)
        e_total_new = self.compute_total_likelihood(pg_new, show_energy=False) + self.compute_prior(pg_new)
        accept = metropolis_hasting(e_total_ori, e_total_new, move_prob, 1 - move_prob, T)
        if accept:
            self.record_accept()
            self.energy_landscape.append(e_total_new)
            pg_ori = copy.deepcopy(pg_new)
            e_total_ori = e_total_new
            if move_type == 'mwall':
                if move_dis > 0:
                    print 'move the middle wall to the front, gradient is:{}, move distance is :{}'.format(gradient,
                                                                                                           move_dis)
                else:
                    print 'move the middle wall to the back, gradient is:{}, move distance is :{}'.format(gradient,
                                                                                                          move_dis)
            if move_type == 'lwall' or move_type == 'rwall':
                if move_dis > 0:
                    print 'move the {} to the left, gradient is:{}, move distance is :{}'.format(move_type, gradient,
                                                                                                           move_dis)
                else:
                    print 'move the {} to the right, gradient is:{}, move distance is :{}'.format(move_type, gradient,
                                                                                           move_dis)
            if move_type == 'floor' or move_type == 'ceiling':
                if move_dis > 0:
                    print 'move the {} to the up, gradient is:{}, move distance is :{}'.format(move_type, gradient,
                                                                                                           move_dis)
                else:
                    print 'move the {} to the down, gradient is:{}, move distance is :{}'.format(move_type, gradient,
                                                                                                          move_dis)
            if move_type == 'rotate':
                    print 'rotate the layout with {}'.format(rotate_angle / np.pi * 180)
            self.record.append(1)
        else:
            self.record.append(0)
        return pg_ori, e_total_ori

    # adjust layout
    def layout_adjust(self, pg_ori, e_total_ori, T):
        r = random.random()
        if 0 < r < 0.2:
            move_type = 'rotate'
        elif 0.2 < r <= 0.4:
            move_type = 'mwall'
        elif 0.4 < r <= 0.6:
            move_type = 'lwall'
        elif 0.6 < r <= 0.8:
            move_type = 'rwall'
        elif 0.8 < r <= 0.9:
            move_type = 'ceiling'
        elif 0.9 < r <= 1:
            move_type = 'floor'
        # move_type = 'rotate'
        return self.one_step_layout_moving(pg_ori, e_total_ori, T, move_type)

    def one_step_object_adjust(self, pg_ori, e_total_ori, T, move_object, move_type, move_face=None):
        print 'adjust {} with {} for object {}, T = {}, scale ={}'.format(move_type, move_face, pg_ori.objects[move_object].obj_type, T, self.scale)
        # translate the object to avoid occlusion between objects
        if move_type == 'translate':
            size_ori = pg_ori.objects[move_object].terminal.obj_size
            if move_face == 'x':
                delta = 0.2 * size_ori[0] * self.scale
            elif move_face == 'y':
                delta = 0.2 * size_ori[1] * self.scale
            elif move_face == 'z':
                delta = 0.2 * size_ori[2] * self.scale
                if pg_ori.objects[move_object]._supported_obj == 'floor':
                    print 'not moving'
                    return pg_ori, e_total_ori
            pg_des = copy.deepcopy(pg_ori)
            if move_face == 'x':
                pg_des.objects[move_object].terminal.move_x(delta)
            elif move_face == 'y':
                pg_des.objects[move_object].terminal.move_y(delta)
            elif move_face == 'z':
                pg_des.objects[move_object].terminal.move_z(delta)
            e_total_des = self.compute_total_likelihood(pg_des, if_vis=False) + self.compute_prior(pg_des)
            gradient = (e_total_ori - e_total_des) / np.abs(delta)
            if gradient == 0:
                return pg_ori, e_total_ori
            gradient_type, move_prob = self.q_moving_proposal()
            if move_face == 'x':
                move_dis = 0.2 * size_ori[0] * self.scale
            elif move_face == 'y':
                move_dis = 0.2 * size_ori[1] * self.scale
            elif move_face == 'z':
                move_dis = 0.2 * size_ori[2] * self.scale
            move_dis *= np.sign(gradient)
            if gradient_type == 1:
                move_dis *= -1
            pg_new = copy.deepcopy(pg_ori)
            if move_face == 'x':
                pg_new.objects[move_object].terminal.move_x(move_dis)
            elif move_face == 'y':
                pg_new.objects[move_object].terminal.move_y(move_dis)
            elif move_face == 'z':
                pg_new.objects[move_object].terminal.move_z(move_dis)
            e_total_new = self.compute_total_likelihood(pg_new, show_energy=True) + self.compute_prior(pg_new)
            accept = metropolis_hasting(e_total_ori, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.record_accept()
                self.energy_landscape.append(e_total_new)
                pg_ori = copy.deepcopy(pg_new)
                e_total_ori = e_total_new
                self.record.append(1)
                print 'new energy is: {}, gradient is :{}, move_distance is :{}'.format(e_total_new, gradient, move_dis)

        elif move_type == 'position':
            size_ori = pg_ori.objects[move_object].terminal.obj_size
            if move_face in ['rface', 'lface']:
                delta = 0.2 * size_ori[0] * self.scale
            elif move_face in ['upface', 'downface']:
                delta = 0.2 * size_ori[2] * self.scale
            elif move_face in ['frontface', 'backface']:
                delta = 0.2 * size_ori[1] * self.scale
            pg_des = copy.deepcopy(pg_ori)
            if move_face == 'rface':
                pg_des.objects[move_object].terminal.move_rface(delta)
            elif move_face == 'lface':
                pg_des.objects[move_object].terminal.move_lface(delta)
            elif move_face == 'upface':
                pg_des.objects[move_object].terminal.move_upface(delta)
            elif move_face == 'downface':
                if pg_des.objects[move_object]._supported_obj == 'floor':
                    print 'not moving'
                    return pg_ori, e_total_ori
                pg_des.objects[move_object].terminal.move_downface(delta)
            elif move_face == 'frontface':
                pg_des.objects[move_object].terminal.move_frontface(delta)
            elif move_face == 'backface':
                pg_des.objects[move_object].terminal.move_backface(delta)
            e_total_des = self.compute_total_likelihood(pg_des, if_vis=False) + self.compute_prior(pg_des)
            gradient = (e_total_ori - e_total_des) / np.abs(delta)
            if gradient == 0:
                return pg_ori, e_total_ori
            gradient_type, move_prob = self.q_moving_proposal()
            move_scale = 0.2 * self.scale
            if move_face in ['rface', 'lface']:
                move_dis = move_scale * size_ori[0]
            elif move_face in ['upface', 'downface']:
                move_dis = move_scale * size_ori[2]
            elif move_face in ['frontface', 'backface']:
                move_dis = move_scale * size_ori[1]
            move_dis *= np.sign(gradient)
            if gradient_type == 1:
                move_dis *= -1
            pg_new = copy.deepcopy(pg_ori)
            if move_face == 'rface':
                pg_new.objects[move_object].terminal.move_rface(move_dis)
            elif move_face == 'lface':
                pg_new.objects[move_object].terminal.move_lface(move_dis)
            elif move_face == 'upface':
                pg_new.objects[move_object].terminal.move_upface(move_dis)
            elif move_face == 'downface':
                pg_new.objects[move_object].terminal.move_downface(move_dis)
            elif move_face == 'frontface':
                pg_new.objects[move_object].terminal.move_frontface(move_dis)
            elif move_face == 'backface':
                pg_new.objects[move_object].terminal.move_backface(move_dis)
            e_total_new = self.compute_total_likelihood(pg_new, show_energy=True) + self.compute_prior(pg_new)
            accept = metropolis_hasting(e_total_ori, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.record_accept()
                self.energy_landscape.append(e_total_new)
                pg_ori = copy.deepcopy(pg_new)
                e_total_ori = e_total_new
                self.record.append(1)
                print 'new energy is: {}, gradient is :{}, move_distance is :{}'.format(e_total_new, gradient, move_dis)
            else:
                self.record.append(0)
        elif move_type == 'normal':
            angle_ori = pg_ori.objects[move_object].terminal.angle
            angle_des = copy.deepcopy(angle_ori)
            delta = 11.25 * self.scale * self.normal_scale
            #
            angle_des += delta
            pg_des = copy.deepcopy(pg_ori)
            pg_des.objects[move_object].terminal.set_angle(angle_des)
            e_total_des = self.compute_total_likelihood(pg_des, if_vis=True) + self.compute_prior(pg_des)
            gradient_pos = (e_total_ori - e_total_des) / np.abs(delta)

            # compute the energy for the negative case
            pg_des_neg = copy.deepcopy(pg_ori)
            angle_des_neg = copy.deepcopy(angle_ori)
            angle_des_neg -= delta
            pg_des_neg.objects[move_object].terminal.set_angle(angle_des_neg)
            e_total_des_neg = self.compute_total_likelihood(pg_des_neg, if_vis=True) + self.compute_prior(pg_des_neg)
            gradient_neg = (e_total_ori - e_total_des_neg) / np.abs(delta)
            gradient_type, move_prob = self.q_moving_proposal()
            if gradient_neg > gradient_pos:
                gradient = gradient_neg
                move_scale = -11.25 * self.scale * self.normal_scale
            else:
                gradient = gradient_pos
                move_scale = 11.25 * self.scale * self.normal_scale
            if gradient == 0:
                return pg_ori, e_total_ori
            if gradient_type == 0:
                move_dis = np.sign(gradient) * move_scale
            else:
                move_dis = - np.sign(gradient) * move_scale
            angle_new = copy.deepcopy(angle_ori)
            angle_new += move_dis
            pg_new = copy.deepcopy(pg_ori)
            pg_new.objects[move_object].terminal.set_angle(angle_new)
            e_total_new = self.compute_total_likelihood(pg_new, show_energy=True) + self.compute_prior(pg_new)
            accept = metropolis_hasting(e_total_ori, e_total_new, move_prob, 1 - move_prob, T)
            if accept:
                self.record_accept()
                self.energy_landscape.append(e_total_new)
                pg_ori = copy.deepcopy(pg_new)
                e_total_ori = e_total_new
                self.record.append(1)
                print 'new energy is {}, gradient is :{}, gradient_pos is :{}, gradient_neg is :{}, move_distance is :{}'.format(e_total_new, gradient, gradient_pos, gradient_neg, move_dis)
            else:
                self.record.append(0)
        return pg_ori, e_total_ori

    def multi_step_position_adjust(self, pg_ori, T, index, move_type):
        print 'adjust lambda segmentation and re-compute the energy'
        self.lambda_seg *= 2.0
        self.lambda_depth *= 2.0
        self.lambda_normal = 0.2
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type,
                                                          move_face='frontface')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type,
                                                          move_face='backface')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type, move_face='rface')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type, move_face='lface')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type, move_face='upface')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type,
                                                          move_face='downface')
        self.lambda_depth /= 2.0
        self.lambda_seg /= 2.0
        self.lambda_normal = 1.0
        # compute energy with 1:1:1 setting
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        return pg_ori, e_total_ori

    def multi_step_normal_adjust(self, pg_ori, T, index, move_type):
        print 'adjust lambda normal and re-compute the energy'
        self.lambda_normal *= 2.0
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type)
        self.lambda_normal /= 2.0
        # compute energy with 1:1:1 setting
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        return pg_ori, e_total_ori

    def multi_step_translate(self, pg_ori, T, index, move_type):
        print 'adjust lambda normal and re-compute the energy'
        self.lambda_seg *= 2.0
        self.lambda_normal = 0.2
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type, move_face='x')
        pg_ori, e_total_ori = self.one_step_object_adjust(pg_ori, e_total_ori, T, index, move_type, move_face='y')
        # compute energy with 1:1:1 setting
        self.lambda_seg /= 2.0
        self.lambda_normal = 1.0

        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        return pg_ori, e_total_ori

    def object_adjust(self, pg_ori, e_total_ori, T, if_random=False, if_initial=False, change_proposal=False):
        if if_random:
            if self.num_object == 0:
                return pg_ori, e_total_ori
            index = random.randint(1, self.num_object) - 1
            print 'adjust the {}'.format(pg_ori.objects[index].obj_type)
            r = random.random()
            if 0 < r <= 0.5:
                move_type = 'position'
                pg_ori, e_total_ori = self.multi_step_position_adjust(pg_ori, T, index, move_type)
            elif 0.5 < r <= 1.0:
                move_type = 'normal'
                pg_ori, e_total_ori = self.multi_step_normal_adjust(pg_ori, T, index, move_type)
        elif if_initial:
            if change_proposal:
                for index in range(self.num_object):
                    print 'change proposals'
                    pg_list = list()
                    energy_list = list()
                    for i in [0, 1]:  # try two proposals for each objects
                        pg_new = copy.deepcopy(pg_ori)
                        pg_new.objects[index].set_terminal(pg_new.objects[index].proposals[i])
                        for _ in range(1):  # adjust the position and normal in three rounds
                            for j in range(2):
                                move_type = 'position'
                                pg_new, e_total_new = self.multi_step_position_adjust(pg_new, T, index, move_type)

                            for j in range(2):
                                move_type = 'normal'
                                pg_new, e_total_new = self.multi_step_normal_adjust(pg_new, T, index, move_type)
                        pg_list.append(pg_new)
                        energy_list.append(e_total_new)
                    max_index = energy_list.index(max(energy_list))
                    print 'best proposal for {} is {}'.format(pg_new.objects[index].obj_type, max_index)
                    pg_ori = copy.deepcopy(pg_list[max_index])
                    e_total_ori = energy_list[max_index]
            else:
                for _ in range(1):
                    for index in range(self.num_object):
                        # print index
                        # if index != 4:
                        #     continue
                        print 'initial the {}'.format(pg_ori.objects[index].obj_type)
                        if self.bool_intersection[index] == 1:
                            print 'translate object {}'.format(pg_ori.objects[index].obj_type)
                            for i in range(2):
                                move_type = 'translate'
                                pg_ori, e_total_ori = self.multi_step_translate(pg_ori, T, index, move_type)
                        # let the normal scale be large at first to jump out of the local mode
                        if pg_ori.objects[index].obj_type in ['chair', 'sofa', 'desk', 'table', 'dresser',
                                                              'night_stand', 'sofa_chair', 'coffee_table']:
                            normal_scale = 3
                        else:
                            normal_scale = 1
                        for i in range(normal_scale, 0, -1):
                            self.scale *= (2 ** i)
                            move_type = 'normal'
                            pg_ori, e_total_ori = self.multi_step_normal_adjust(pg_ori, T, index, move_type)
                            self.scale /= (2 ** i)
                        for j in range(2):
                            # for the occluded object, we translate the object to avoid occlusion
                            print 'translate object {}'.format(pg_ori.objects[index].obj_type)
                            for i in range(2):
                                move_type = 'translate'
                                pg_ori, e_total_ori = self.multi_step_translate(pg_ori, T, index, move_type)
                            for i in range(2):
                                move_type = 'position'
                                pg_ori, e_total_ori = self.multi_step_position_adjust(pg_ori, T, index, move_type)
                            for i in range(2):
                                move_type = 'normal'
                                pg_ori, e_total_ori = self.multi_step_normal_adjust(pg_ori, T, index, move_type)

        else:
            for index in range(self.num_object):
                move_type = 'translate'
                pg_ori, e_total_ori = self.multi_step_translate(pg_ori, T, index, move_type)
                move_type = 'position'
                pg_ori, e_total_ori = self.multi_step_position_adjust(pg_ori, T, index, move_type)
                move_type = 'normal'
                pg_ori, e_total_ori = self.multi_step_normal_adjust(pg_ori, T, index, move_type)
        return pg_ori, e_total_ori

    # determine which type to adjust
    def adjust_type(self, type_prob):
        r = random.random()
        if 0 < r <= sum(type_prob[:1]):
            adjust_type = 'layout'
        elif sum(type_prob[:1]) < r <= sum(type_prob[:2]):
            adjust_type = 'proposal'
        elif sum(type_prob[:2]) < r <= sum(type_prob[:3]):
            adjust_type = 'property'
        elif sum(type_prob[:3]) < r <= sum(type_prob[:4]):
            adjust_type = 'support'
        return adjust_type

    # push the objects covered by the layouts to the front
    def push_to_front(self, pg_ori):
        pg_new = copy.deepcopy(pg_ori)
        for index in range(self.num_object):
            if index == 0:
                print pg_new.objects[index].terminal.obj_center_proposals
            for step in range(10):
                center = pg_new.objects[index].terminal.obj_center
                corners = copy.deepcopy(pg_new.layouts.corners)
                p1, p2, p3, p4 = rectangle_shrink(corners[0], corners[1], corners[2], corners[3], 0.8)
                point = Point(center[0], center[1])
                polygon = Polygon([(p1[0], p1[1]), (p2[0], p2[1]), (p3[0], p3[1]), (p4[0], p4[1])])
                if not polygon.contains(point) or pg_new.objects[index].terminal.obj_center[2] - pg_new.objects[index].terminal.obj_size[2] / 4 < pg_new.layouts.floor[0][2]:
                    pg_new.objects[index].terminal.set_center(copy.deepcopy(pg_new.objects[index].terminal.obj_center_proposals[step]))
                else:
                    break
        e_total_new = self.compute_total_likelihood(pg_new, show_energy=True) + self.compute_prior(pg_new)
        return pg_new, e_total_new
    # infer the best parse graph with lowest energy

    def joint_infer(self, pg_cur):
        pg_ori = self.load_result(pg_cur)
        # adjust the object according to sampled human
        skeleton_path = os.path.join(stats_root, 'skeleton', 'hoi_relation.pickle')
        with open(skeleton_path, 'r') as f:
            skeleton_stats = pickle.load(f)
        f.close()
        T = 0.00001
        self.scale = 1
        for _ in range(2):
            sample = HumanSample(pg_ori, skeleton_stats, self.save_path)
            pg_new, adjust_index = sample.sample()
            if len(adjust_index) == 0:
                break
            pg_ori = copy.deepcopy(pg_new)
            for i in range(2):
                for index in adjust_index:
                    move_type = 'translate'
                    pg_ori, e_total_ori = self.multi_step_translate(pg_ori, T, index, move_type)
                    move_type = 'position'
                    pg_ori, e_total_ori = self.multi_step_position_adjust(pg_ori, T, index, move_type)
                    move_type = 'normal'
                    pg_ori, e_total_ori = self.multi_step_normal_adjust(pg_ori, T, index, move_type)
        with open(os.path.join(self.save_path, 'human.pickle'), 'w') as f:
            pickle.dump(pg_ori, f)
        f.close()
        print 'finish joint sampling'

    def infer_pg(self):
        start = time.time()
        pg_ori = self.load_obj(self.pg)
        # Before initialize the layout, the prior may hurt the performance.
        # We don't count the prior until we finish adjusting the layout
        self.count_prior = False
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        self.energy_landscape.append(e_total_ori)
        if not os.path.exists(os.path.join(self.save_path, 'normal')):
            os.mkdir(os.path.join(self.save_path, 'normal'))
        if not os.path.exists(os.path.join(self.save_path, 'depth')):
            os.mkdir(os.path.join(self.save_path, 'depth'))
        if not os.path.exists(os.path.join(self.save_path, 'seg')):
            os.mkdir(os.path.join(self.save_path, 'seg'))
        # adjust the layout
        print 'optimize the layout'
        for _ in range(30):
            adjust_type = self.adjust_type([1, 0.0, 0.0, 0.0])
            if adjust_type == 'layout':
                pg_ori, e_total_ori = self.layout_adjust(pg_ori, e_total_ori, 0.001)
        # push to the front
        with open(os.path.join(self.save_path, 'layout.pickle'), 'w') as f:
            pickle.dump(pg_ori, f)
        f.close()
        # push the object to the front to avoid local maximum
        print 'push to the front'
        pg_ori, e_total_ori = self.push_to_front(pg_ori)
        # initialize the support relation
        print 'support init'
        pg_ori, e_total_ori = self.support_initial(pg_ori, e_total_ori)
        # adjust the object
        print 'optimize the object'
        self.count_prior = True
        e_total_ori = self.compute_total_likelihood(pg_ori, show_energy=True) + self.compute_prior(pg_ori)
        # annealing
        for T in [0.01, 0.001, 0.0001]:
            pg_ori, e_total_ori = self.object_adjust(pg_ori, e_total_ori, T, if_initial=True, change_proposal=False)
            for _ in range(20):
                pg_ori, e_total_ori = self.layout_adjust(pg_ori, e_total_ori, 0.001)
        self.scale /= 2.0
        for T in [0.00001]:
            pg_ori, e_total_ori = self.object_adjust(pg_ori, e_total_ori, T, if_initial=True, change_proposal=False)
            for _ in range(20):
                pg_ori, e_total_ori = self.layout_adjust(pg_ori, e_total_ori, 0.001)
            for _ in range(self.num_object * 3):
                pg_ori, e_total_ori = self.object_adjust(pg_ori, e_total_ori, T, if_random=True)
        print 'optimize the object and layout together'
        with open(os.path.join(self.save_path, 'joint.pickle'), 'w') as f:
            pickle.dump(pg_ori, f)
        f.close()
        # check the support relations
        pg_ori, e_total_ori = self.support_initial(pg_ori, e_total_ori)
        with open(os.path.join(self.save_path, 'support.pickle'), 'w') as f:
            pickle.dump(pg_ori, f)
        f.close()
        end = time.time()
        print self.energy_landscape
        print self.record
        print self.inference_step
        print 'total inference time is :{}'.format(end - start)


# infer layout and objects iteratively
def inference(index):
    print 'infer 3D room layout and 3D object for sample {}'.format(index)
    pg_root = os.path.join(proposal_root, 'pg', str(index) + '.pickle')
    if not os.path.exists(pg_root):
        print 'pg not existed'
    with open(pg_root, 'r') as f:
        pg = pickle.load(f)
    f.close()
    est_depth = np.load(os.path.join(proposal_root, 'depth', str(index) + '.npy'))
    est_seg = np.load(os.path.join(proposal_root, 'segmentation', str(index) + '.npy')).T
    est_normal = np.load(os.path.join(proposal_root, 'surface_normal', str(index) + '.npy'))
    m, n = est_seg.shape[:2]
    for k in range(m):
        for l in range(n):
            if est_seg[k, l] == 18:
                est_seg[k, l] = 4
    if m != pg.camera.K[1, 2] * 2 or n != pg.camera.K[0, 2] * 2:
        pg.camera._K[1, 2] = m / 2.0
        pg.camera._K[0, 2] = n / 2.0
    m, n = est_normal.shape[:2]
    if m != pg.camera.K[1, 2]*2 or n != pg.camera.K[0, 2]*2:
        est_normal = est_normal[int(pg.camera.K[1, 2]*2), int(pg.camera.K[0, 2]*2), :]
    save_path = os.path.join(save_root, str(index))
    inference = Inference(pg, est_depth, est_seg, est_normal, 1, 1, 1, save_path)
    inference.infer_pg()


def inference_human(index):
    print 'infer 3D room layout, objects and human context for sample {}'.format(index)
    pg_root = os.path.join('result', str(index),  'support.pickle')
    if not os.path.exists(pg_root):
        print 'result not existed, run inference first'
    with open(pg_root, 'r') as f:
        pg = pickle.load(f)
    f.close()
    est_depth = np.load(os.path.join(proposal_root, 'depth', str(index) + '.npy'))
    est_seg = np.load(os.path.join(proposal_root, 'segmentation', str(index) + '.npy')).T
    est_normal = np.load(os.path.join(proposal_root, 'surface_normal', str(index) + '.npy'))
    m, n = est_seg.shape[:2]
    for k in range(m):
        for l in range(n):
            if est_seg[k, l] == 18:
                est_seg[k, l] = 4
    if m != pg.camera.K[1, 2] * 2 or n != pg.camera.K[0, 2] * 2:
        pg.camera._K[1, 2] = m / 2.0
        pg.camera._K[0, 2] = n / 2.0
    m, n = est_normal.shape[:2]
    if m != pg.camera.K[1, 2] * 2 or n != pg.camera.K[0, 2] * 2:
        est_normal = est_normal[int(pg.camera.K[1, 2] * 2), int(pg.camera.K[0, 2] * 2), :]
    save_path = os.path.join(save_root, str(index))
    inference = Inference(pg, est_depth, est_seg, est_normal, 1, 1, 1, save_path)
    inference.joint_infer(pg)


def main():
    opt = sys.argv[1]
    index = int(sys.argv[2])
    if opt == '-lo':
        inference(index)
    elif opt == '-human':
        inference_human(index)


if __name__ == '__main__':
    main()
