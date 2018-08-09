"""
Created on Sep 19, 2017

@author: Siyuan Huang

code for getting a parse graph for a total scene
"""
import copy


class PG(object):
    def __init__(self, camera, layouts, objects, sequence_name, sequence_id):
        self._scene_type = None
        self._layouts = layouts
        self._camera = camera
        self._activities = None
        self._sequence_name = sequence_name
        self._sequence_id = sequence_id
        self._objects = objects

    def __str__(self):
        return 'sequence_name: {}, sequence_id: {}'.format(self._sequence_name, self._sequence_id)

    def __repr__(self):
        return self.__str__()

    @property
    def sequence_name(self):
        return self._sequence_name

    @property
    def sequence_id(self):
        return self._sequence_id

    @property
    def scene_type(self):
        return self._scene_type

    @property
    def layouts(self):
        return self._layouts

    @property
    def camera(self):
        return self._camera

    @property
    def activities(self):
        return self._activities

    @property
    def objects(self):
        return self._objects

    def set_scene_type(self, scene_type):
        self._scene_type = scene_type


class Layout(object):
    def __init__(self, layout_info):
        self._R_C = layout_info['R_C']
        self._T_C = layout_info['T_C']
        self._P_camera = layout_info['P_camera']
        self._ceiling = layout_info['ceiling']
        self._floor = layout_info['floor']
        self._mwall = layout_info['mwall']
        self._lwall = layout_info['lwall']
        self._rwall = layout_info['rwall']
        self._corners = layout_info['corners']
        self._usable = layout_info['usable']
        self._wall_color = None
        self._floor_color = None
        self._ceiling_color = None

    def __str__(self):
        return 'usable: {}'.format(self._usable)

    def __repr__(self):
        return self.__str__()

    @property
    def R_C(self):
        return self._R_C

    @property
    def T_C(self):
        return self._T_C

    @property
    def P_camera(self):
        return self._P_camera

    @property
    def ceiling(self):
        return self._ceiling

    @property
    def floor(self):
        return self._floor

    @property
    def mwall(self):
        return self._mwall

    @property
    def lwall(self):
        return self._lwall

    @property
    def rwall(self):
        return self._rwall

    @property
    def corners(self):
        return self._corners

    @property
    def usable(self):
        return self._usable

    @property
    def wall_color(self):
        return self._wall_color

    @property
    def floor_color(self):
        return self._floor_color

    @property
    def ceiling_color(self):
        return self._ceiling_color

    def set_floor(self, floor):
        self._floor = floor

    def set_ceiling(self, ceiling):
        self._ceiling = ceiling

    def set_mwall(self, mwall):
        self._mwall = mwall

    def set_lwall(self, lwall):
        self._lwall = lwall

    def set_rwall(self, rwall):
        self._rwall = rwall

    def set_corners(self, corners):
        self._corners = corners
        self.set_ceiling([corners[i] for i in range(4, 8)])
        self.set_floor([corners[i] for i in range(4)])
        self.set_mwall([corners[i] for i in [0, 1, 5, 4]])
        self.set_lwall([corners[i] for i in [1, 2, 6, 5]])
        self.set_rwall([corners[i] for i in [0, 3, 7, 4]])


# the functional object node which contains the group relation and support relation
class Object(object):
    def __init__(self, obj_type, type_prob):
        self._obj_type = obj_type
        self._type_prob = type_prob
        self._action_group = None
        self._group_prob = None
        self._supported_obj = None
        self._supporting_obj = None
        self._proposals = list()
        self._terminal = None

    def __str__(self):
        return 'object type: {}'.format(self._obj_type)

    def __repr__(self):
        return self.__str__()

    @property
    def obj_type(self):
        return self._obj_type

    @property
    def type_prob(self):
        return self._type_prob

    @property
    def action_group(self):
        return self._action_group

    @property
    def group_prob(self):
        return self._group_prob

    @property
    def supported_obj(self):
        return self._supported_obj

    @property
    def supporting_obj(self):
        return self._supporting_obj

    @property
    def proposals(self):
        return self._proposals

    @property
    def terminal(self):
        return self._terminal

    def append_proposal(self, proposal_3d):
        self._proposals.append(proposal_3d)

    def set_terminal(self, terminal_node):
        self._terminal = copy.deepcopy(terminal_node)


# terminal node with attributes for the parse graph, which is also the node in geometric space
class ObjProposal(object):
    def __init__(self, obj_id, id_prob, obj_size, size_prob, obj_center, obj_center_proposals,  center_prob, obj_rotate, rotate_prob, angle, obj_color=None):
        self._obj_id = obj_id
        self._id_prob = id_prob
        self._obj_size = obj_size
        self._size_prob = size_prob
        self._obj_center = obj_center
        self._obj_center_proposals = obj_center_proposals
        self._center_prob = center_prob
        self._obj_rotate = obj_rotate
        self._rotate_prob = rotate_prob
        self._angle = angle
        self._obj_color = obj_color

    def __str__(self):
        return 'object id: {}'.format(self._obj_id)

    def __repr__(self):
        return self.__str__()

    @property
    def obj_id(self):
        return self._obj_id

    @property
    def id_prob(self):
        return self._id_prob

    @property
    def obj_size(self):
        return self._obj_size

    @property
    def size_prob(self):
        return self._size_prob

    @property
    def obj_center(self):
        return self._obj_center

    @property
    def obj_center_proposals(self):
        return self._obj_center_proposals

    @property
    def center_prob(self):
        return self._center_prob

    @property
    def obj_rotate(self):
        return self._obj_rotate

    @property
    def rotate_prob(self):
        return self._rotate_prob

    @property
    def angle(self):
        return self._angle

    @property
    def obj_color(self):
        return self._obj_color

    def set_size(self, size):
        self._obj_size = size

    def set_center(self, center):
        self._obj_center = center

    def set_rotate(self, rotate):
        self._obj_rotate = rotate

    def set_size_prob(self, size_prob):
        self._size_prob = size_prob

    def set_angle(self, angle):
        self._angle = angle

    def move_rface(self, dis):
        if self._obj_size[0] + dis > 0:
            self._obj_center[0] -= dis / 2
            self._obj_size[0] += dis

    def move_lface(self, dis):
        if self._obj_size[0] + dis > 0:
            self._obj_center[0] += dis / 2
            self._obj_size[0] += dis

    def move_upface(self, dis):
        if self._obj_size[2] + dis > 0:
            self._obj_center[2] += dis / 2
            self._obj_size[2] += dis

    def move_downface(self, dis):
        if self._obj_size[2] + dis > 0:
            self._obj_center[2] -= dis / 2
            self._obj_size[2] += dis

    def move_frontface(self, dis):
        if self._obj_size[1] + dis > 0:
            self._obj_center[1] += dis / 2
            self._obj_size[1] += dis

    def move_backface(self, dis):
        if self._obj_size[1] + dis > 0:
            self._obj_center[1] -= dis / 2
            self._obj_size[1] += dis

    def move_x(self, dis):
        self._obj_center[0] += dis

    def move_y(self, dis):
        self._obj_center[1] += dis

    def move_z(self, dis):
        self._obj_center[2] += dis


class Camera(object):
    def __init__(self, K, R, R_tilt, alpha, beta, gamma, height):
        self._K = K
        self._R = R
        self._R_tilt = R_tilt
        self._alpha = alpha  # tilt angle along the z axis in OPENGL coordinates
        self._beta = beta    # tilt angle along the x axis in OPENGL coordinates
        self._gamma = gamma  # tilt angle along the y axis in OPENGL coordinates
        self._height = height
        self._focal_length = self._K[0, 0]

    def __str__(self):
        return 'focal length: {}, height: {}'.format(self._focal_length, self._height)

    def __repr__(self):
        return self.__str__()

    @property
    def K(self):
        return self._K

    @property
    def R(self):
        return self._R

    @property
    def R_tilt(self):
        return self._R_tilt

    @property
    def alpha(self):
        return self._alpha

    @property
    def beta(self):
        return self._beta

    @property
    def gamma(self):
        return self._gamma

    @property
    def focal_length(self):
        return self._focal_length


class ActivityGroup(object):
    def __init__(self, group_type, prob_group, human_skeleton=None):
        self._group_type = group_type
        self._prob_group = prob_group
        self._human_skeleton = human_skeleton
        self._objects = list()

    def __str__(self):
        return 'group type: {}'.format(len(self._group_type))

    def __repr__(self):
        return self.__str__()

    @property
    def group_type(self):
        return self._group_type

    @property
    def prob_group(self):
        return self._prob_group

    @property
    def human_skeleton(self):
        return self._human_skeleton

    @property
    def objects(self):
        return self._objects

    def set_human_skeleton(self, skeleton):
        self._human_skeleton = skeleton

    def add_object(self, obj):
        self._objects.append(obj)


def main():
    pass

if __name__ == '__main__':
    main()
