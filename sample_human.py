"""
Created on Nov 4, 2017

@author: Siyuan Huang

Sample Human from the Scene

"""
import config
import os
import pickle
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import visualize
import copy
from camera import rotation_matrix_3d_z, center_to_corners
import random
import sys
from PIL import Image
paths = config.Paths()
metadata_root = paths.metadata_root
proposal_root = paths.proposal_root
evaluation_root = os.path.join(metadata_root, 'evaluation')
stats_root = os.path.join(metadata_root, 'sunrgbd_stats')
office_label = ['reading', 'put-down-item', 'take-item', 'play-computer']


class HumanSample(object):
    def __init__(self, pg, skeleton_stats, save_root='.', if_vis=True):
        self.pg = copy.deepcopy(pg)
        self.skeleton_stats = skeleton_stats
        self.learning_rate = 1
        self.pg_output = copy.deepcopy(pg)
        self.skeleton = list()
        self.adjust_index = list()
        self.accept_record = list()
        self.if_vis = if_vis
        self.save_root = save_root
        self.mean_pose_root = os.path.join(stats_root, 'skeleton', 'aligned_skeleton.mat')

    def get_major_object(self, action_label):
        obj_list = list()
        obj_num = list()
        for key, value in self.skeleton_stats[action_label].iteritems():
            if not value:
                continue
            action_len = value['number']
            obj_list.append(key)
            obj_num.append(action_len)
        obj_num = np.array(obj_num)
        index = np.argsort(obj_num)[::-1]
        return [obj_list[i] for i in index]

    def get_mean_pose(self, action_label):
        mean_pose = loadmat(self.mean_pose_root)['alignedSkeleton'][0]
        mean_pose = mean_pose[office_label.index(action_label)]
        return mean_pose

    def compute_skeleton(self, initial_pose, rotate_angle, T, scale):
        rotate_angle = rotate_angle / 180.0 * np.pi
        initial_pose = copy.copy(initial_pose)
        initial_pose *= scale
        pos = rotation_matrix_3d_z(rotate_angle).dot(initial_pose.T).T + np.tile(T, (25, 1))
        return pos

    # transform the coordinates from corner-at-origin to camera-at-origin
    def pg_to_real(self, pg):
        pg_new = copy.copy(pg)
        R_C = pg.layouts.R_C
        T_C = pg.layouts.T_C
        for obj in pg_new.objects:
            obj.terminal.set_center(R_C.T.dot(np.array(obj.terminal.obj_center).T).T - np.array(T_C))
            obj.terminal._obj_center[1] += obj.terminal.obj_size[1] / 2
            if obj.obj_type == 'desk':
                obj._obj_type = 'table'
        return pg_new

    def vis_skeleton(self, pg, cur_skeleton, involved_index):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        # ax.scatter(p_camera[0], p_camera[1], p_camera[2], c='r', marker='o')
        for obj_index, obj in enumerate(pg.objects):
            p = center_to_corners(pg.objects[obj_index].terminal.obj_center, pg.objects[obj_index].terminal.obj_size, pg.objects[obj_index].terminal.angle)
            if obj_index not in involved_index:
                visualize.plot_cuboid(ax, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 'g-')
            else:
                visualize.plot_cuboid(ax, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], 'b-')
        visualize.plot_skeleton(ax, cur_skeleton, 'r-')
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
        plt.show()
        plt.close()

    def vis_skeleton_2d_all(self, pg):
        img = np.array(Image.open(os.path.join(metadata_root, 'images', '%06d' % pg.sequence_id + '.jpg')))
        plt.figure()
        plt.imshow(img)
        for obj in self.pg.objects:
            if obj.action_group is not None:
                r = random.random()
                num_actitivity = len(obj.action_group)
                if 0 < r < 0.5:
                    vis_index = 0
                else:
                    vis_index = np.random.permutation(num_actitivity)[0]
                cur_skeleton = obj.action_group[vis_index]['skeleton']
                cur_skeleton = np.vstack((-cur_skeleton[:, 0].T, cur_skeleton[:, 2].T, -cur_skeleton[:, 1].T))
                skeleton_2d = pg.camera.K.dot(pg.camera.R_tilt).dot(cur_skeleton)  # note the gt3dcorner
                skeleton_2d = skeleton_2d[:2, :] / skeleton_2d[2, :]
                skeleton_2d = skeleton_2d.T
                index = [[3, 2], [2, 4], [2, 8], [2, 1], [1, 0], [0, 12], [0, 16], [4, 5], [5, 6], [6, 7], [8, 9], [9, 10],
                         [10, 11],
                         [12, 13], [13, 14], [14, 15], [16, 17], [17, 18], [18, 19]]
                for ind in index:
                    plt.plot([skeleton_2d[ind[0], 0], skeleton_2d[ind[1], 0]], [skeleton_2d[ind[0], 1], skeleton_2d[ind[1], 1]], 'r-', linewidth=3.0)
        plt.axis('off')
        plt.show()

    def vis_skeleton_3d_all(self, pg):
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        for obj_index, obj in enumerate(pg.objects):
            p = center_to_corners(pg.objects[obj_index].terminal.obj_center, pg.objects[obj_index].terminal.obj_size,
                                  pg.objects[obj_index].terminal.angle*2)
            visualize.plot_cuboid(ax, p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], visualize.sunrgbd_object_color(pg.objects[obj_index].obj_type))
        for obj in self.pg.objects:
            if obj.action_group is not None:
                r = random.random()
                num_actitivity = len(obj.action_group)
                if 0 < r < 0.5:
                    vis_index = 0
                else:
                    vis_index = np.random.permutation(num_actitivity)[0]
                cur_skeleton = obj.action_group[vis_index]['skeleton']
                visualize.plot_skeleton(ax, cur_skeleton, 'r-')
        ax.set_xlim([-1, 2])
        ax.set_ylim([-1, 2])
        ax.set_zlim([-1, 2])
        ax.grid(False)
        ax.axis('off')
        plt.show()

    def compute_likelihood(self, action_label, skeleton, involved_index):
        li = 1
        for obj_index in involved_index:
            obj_type = self.pg.objects[obj_index].obj_type
            if obj_type == 'desk' or obj_type == 'table':
                distance = skeleton - (self.pg.objects[obj_index].terminal.obj_center + np.array([0, 0, self.pg.objects[obj_index].terminal.obj_size[2] / 2]))
            else:
                distance = skeleton - self.pg.objects[obj_index].terminal.obj_center
            distance = np.linalg.norm(distance, axis=1)
            prob = np.exp(-np.linalg.norm(distance) / len(involved_index))
            li *= prob
        return li

    def sample_around_object(self, action_label, mean_pose, involved_index):
        center = self.pg.objects[involved_index[0]].terminal.obj_center
        angle = self.pg.objects[involved_index[0]].terminal.angle
        T = center + np.array([0, 0, 0])
        scale = 1.0
        angle_skeleton = angle
        for _ in range(100):
            r = random.random()
            if 0 < r < 0.5:
                angle_skeleton, T, scale = self.one_step_translate(action_label, involved_index, mean_pose, angle_skeleton,
                                                               T, scale, 2)
            elif 0.5 < r < 0.7:
                angle_skeleton, T, scale = self.one_step_rotate(action_label, involved_index, mean_pose, angle_skeleton, T, scale)
            else:
                angle_skeleton, T, scale = self.one_step_scale(action_label, involved_index, mean_pose, angle_skeleton, T, scale)
        cur_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale)
        cur_li = self.compute_likelihood(action_label, cur_skeleton, involved_index)
        # change the output pg
        if len(involved_index) > 1:
            self.pg_output.objects[involved_index[0]].terminal.set_angle(angle_skeleton)
            for index in involved_index:
                if index not in self.adjust_index:
                    self.adjust_index.append(index)
        self.skeleton.append(cur_skeleton)
        if self.pg.objects[involved_index[0]]._action_group is None:
            self.pg.objects[involved_index[0]]._action_group = list()
        self.pg.objects[involved_index[0]]._action_group.append({'skeleton': cur_skeleton, 'energy': cur_li, 'activity': action_label})


    def one_step_translate(self, action_label, involved_index, mean_pose, angle_skeleton, T, scale, axe):
        delta = 0.1
        cur_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale)
        cur_li = self.compute_likelihood(action_label, cur_skeleton, involved_index)
        T_des = copy.copy(T)
        T_des[axe] += delta
        des_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T_des, scale)
        des_li = self.compute_likelihood(action_label, des_skeleton, involved_index)
        gradient = (des_li - cur_li) / delta
        move_dis = gradient * self.learning_rate
        # print move_dis
        T_new = copy.copy(T)
        T_new[axe] += move_dis
        new_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T_new, scale)
        new_li = self.compute_likelihood(action_label, new_skeleton, involved_index)
        if new_li > cur_li:
            print 'translate skeleton with {}, new energy is {}, old energy is {}'.format(move_dis, new_li, cur_li)
            self.accept_record.append(1)
            return angle_skeleton, T_new, scale
        else:
            self.accept_record.append(0)
            return angle_skeleton, T, scale

    def one_step_random(self, action_label, involved_index, mean_pose, angle_skeleton, T, scale):
        delta = random.random()
        axe = random.randint(0, 2)
        cur_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale)
        cur_li = self.compute_likelihood(action_label, cur_skeleton, involved_index)
        T_des = copy.copy(T)
        T_des[axe] += delta
        des_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T_des, scale)
        des_li = self.compute_likelihood(action_label, des_skeleton, involved_index)
        gradient = (des_li - cur_li) / delta
        move_dis = gradient * self.learning_rate
        # print move_dis
        T_new = copy.copy(T)
        T_new[axe] += move_dis
        new_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T_new, scale)
        new_li = self.compute_likelihood(action_label, new_skeleton, involved_index)
        if new_li > cur_li:
            print 'translate skeleton with {}, new energy is {}, old energy is {}'.format(move_dis, new_li, cur_li)
            self.accept_record.append(1)
            return angle_skeleton, T_new, scale
        else:
            self.accept_record.append(0)
            return angle_skeleton, T, scale

    def one_step_scale(self, action_label, involved_index, mean_pose, angle_skeleton, T, scale):
        delta = 1.05
        cur_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale)
        cur_li = self.compute_likelihood(action_label, cur_skeleton, involved_index)
        scale_des = copy.copy(scale)
        scale_des *= delta
        des_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale_des)
        des_li = self.compute_likelihood(action_label, des_skeleton, involved_index)
        gradient = (des_li - cur_li) / delta
        if gradient > 0:
            scale_new = scale * delta
        else:
            return angle_skeleton, T, scale
        # print move_dis
        new_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale_new)
        new_li = self.compute_likelihood(action_label, new_skeleton, involved_index)
        if new_li > cur_li:
            print 'scale skeleton, new energy is {}'.format(new_li)
            self.accept_record.append(1)
            return angle_skeleton, T, scale_new
        else:
            self.accept_record.append(0)
            return angle_skeleton, T, scale

    def one_step_rotate(self, action_label, involved_index, mean_pose, angle_skeleton, T, scale):
        delta = 11.25
        cur_skeleton = self.compute_skeleton(mean_pose, angle_skeleton, T, scale)
        cur_li = self.compute_likelihood(action_label, cur_skeleton, involved_index)
        angle_left = copy.copy(angle_skeleton)
        angle_left += delta
        left_skeleton = self.compute_skeleton(mean_pose, angle_left, T, scale)
        left_li = self.compute_likelihood(action_label, left_skeleton, involved_index)
        angle_right = copy.copy(angle_skeleton)
        angle_right -= delta
        right_skeleton = self.compute_skeleton(mean_pose, angle_right, T, scale)
        right_li = self.compute_likelihood(action_label, right_skeleton, involved_index)
        if left_li < right_li:
            move_dis = - delta
        else:
            move_dis = delta
        angle_new = copy.copy(angle_skeleton)
        angle_new += move_dis
        new_skeleton = self.compute_skeleton(mean_pose, angle_new, T, scale)
        new_li = self.compute_likelihood(action_label, new_skeleton, involved_index)
        if new_li > cur_li:
            print 'rotate skeleton with {}, new energy is {}, old energy is {}'.format(move_dis, new_li, cur_li)
            self.accept_record.append(1)
            return angle_new, T, scale
        else:
            self.accept_record.append(0)
            return angle_skeleton, T, scale

    def sample(self):
        self.pg = self.pg_to_real(self.pg)
        for action_label in office_label:
            max_obj = self.get_major_object(action_label)
            mean_pose = self.get_mean_pose(action_label)
            for obj_index, obj in enumerate(self.pg.objects):
                if obj.obj_type == max_obj[0]:
                    sample_group = list()
                    sample_group.append(obj_index)
                    for second_index, second_obj in enumerate(self.pg.objects):
                        if (second_obj.obj_type == max_obj[1] or second_obj.obj_type == max_obj[2]) and np.linalg.norm(self.pg.objects[obj_index].terminal.obj_center - self.pg.objects[second_index].terminal.obj_center) < 2:
                            sample_group.append(second_index)
                    self.sample_around_object(action_label, mean_pose, sample_group)
        if len(self.adjust_index) == 0:
            print 'Not able to infer hidden human context in this scene due to the limited data in 3D human object interaction'
            return self.pg_output, self.adjust_index
        self.vis_skeleton_2d_all(self.pg)
        self.vis_skeleton_3d_all(self.pg)
        return self.pg_output, self.adjust_index


def main():
    scene_id = int(sys.argv[1])
    pg_path = os.path.join('result', str(scene_id), 'support.pickle')
    with open(pg_path, 'r') as f:
        pg = pickle.load(f)
    f.close()
    skeleton_path = os.path.join(stats_root, 'skeleton', 'hoi_relation.pickle')
    with open(skeleton_path, 'r') as f:
        skeleton_stats = pickle.load(f)
    f.close()
    sample = HumanSample(pg, skeleton_stats)
    sample.sample()


if __name__ == '__main__':
    main()