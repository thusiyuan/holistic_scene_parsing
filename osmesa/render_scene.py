import osmesa
import os
from OpenGL import GL
from OpenGL import GLU
import math
import csv
import matplotlib.pyplot as plt
import pickle
import numpy as np
import scipy.io
ROTATE_VERTICAL_START = 0
ROTATE_VERTICAL_END = 10
VERTICAL_INTERVAL = 15
ROTATE_HORIZONTAL_START = 0
ROTATE_HORIZONTAL_END = 10
HORIZONTAL_INTERVAL = 20

'''
function to compute depth map
'''

MODEL_PATH = './'
MODEL_SUBPATH = 'models'
sunrgbd_obj_to_seg = [0, 0, 26, 0, 0, 0, 5, 0, 0, 25, 0, 0, 0, 0, 24, 14, 7, 0, 27, 6, 37, 0, 0, 0, 11, 32, 0, 19, 0, 0, 0, 0, 0, 0, 0, 0, 0, 34, 29, 0, 30, 0, 18, 0, 0, 7, 25, 7, 0, 16, 8, 0, 5, 3, 0, 10, 13, 0, 5, 4, 23, 36, 33, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0, 35, 23, 0, 0, 0, 0, 17, 0, 0, 0, 0, 0, 21, 17, 12, 0, 31, 0, 0, 0, 0, 0]

sunrgbd_cate_all = ['__background__', 'laptop', 'paper', 'oven', 'keyboard', 'ottoman', 'chair', 'monitor', 'cup', 'tv',
'bench', 'board', 'stove', 'plate', 'fridge', 'desk', 'coffee_table', 'vanity', 'towel', 'sofa',
'bag', 'tray', 'rack', 'bulletin_board', 'picture', 'night_stand', 'computer', 'mirror',
'container', 'clock', 'stool', 'microwave', 'mug', 'back_pack', 'cubby', 'electric_fan', 'cart',
'sink', 'box', 'island', 'whiteboard', 'desktop', 'pillow', 'pot', 'urinal', 'dining_table',
'tv_stand', 'table', 'projector', 'curtain', 'door', 'shelf', 'sofa_chair', 'cabinet', 'telephone',
'bookshelf', 'blinds', 'thermos', 'stack_of_chairs', 'bed', 'books', 'bathtub', 'toilet', 'scanner',
'recycle_bin', 'endtable', 'glass', 'drawer', 'tissue', 'organizer', 'mouse', 'bowl', 'machine',
'lamp', 'book', 'speaker', 'poster', 'suits_case', 'blanket', 'dresser', 'plant', 'printer',
'garbage_bin', 'podium', 'blackboard', 'cloth', 'dresser_mirror', 'counter', 'flower_vase',
'person', 'switch', 'bottle', 'basket', 'painting', 'cpu']

seg_37_list = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
                   'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor_mat', 'clothes',
                    'ceiling', 'books', 'fridge', 'tv', 'paper', 'towel', 'shower_curtain', 'box', 'whiteboard', 'person',
                    'night_stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

rotation_matrix_list = list()

FAR = 10


def get_angle(p1, p2, rotate=False):
    if rotate is True:
        dot = np.dot(p1, p2)
        det = np.cross(p1, p2)
        angle = np.arctan2(det, dot)
    else:
        dot = np.dot(p1, p2)  # dot product
        det = np.linalg.norm(np.cross(p1, p2))  # determinant
        angle = np.arctan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
    return angle


def rotation_matrix(degree, alpha, beta, gamma):   # when there is no indicator one, will multiply a matrix I
    r = np.zeros((3, 3))
    degree = float(degree)/180*np.pi
    if alpha == 1:
        r[1, 1] = np.cos(degree)
        r[1, 2] = - np.sin(degree)
        r[2, 1] = np.sin(degree)
        r[2, 2] = np.cos(degree)
        r[0, 0] = 1
    elif beta == 1:
        r[0, 0] = np.cos(degree)
        r[0, 2] = np.sin(degree)
        r[2, 0] = - np.sin(degree)
        r[2, 2] = np.cos(degree)
        r[1, 1] = 1
    elif gamma == 1:
        r[0, 0] = np.cos(degree)
        r[0, 1] = - np.sin(degree)
        r[1, 0] = np.sin(degree)
        r[1, 1] = np.cos(degree)
        r[2, 2] = 1
    else:
        r = np.identity(3)
    return r


def OnCaptureResult(render_path, img_path, width, height, true_height, if_vis, render_type='rgb'):
    if render_type == 'rgb':
        rgb_img = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)[::-1, :, :][
              height - true_height:, :, :]
        if if_vis:
            plt.imshow(rgb_img)
            plt.axis('off')
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
        # print render_path
        np.save(render_path, rgb_img)
        return rgb_img
    elif render_type == 'segmentation':
        segment = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)[::-1, :, :][
              height - true_height:, :, 0]
        if if_vis:
            plt.imshow(segment, vmin=0, vmax=38)
            # plt.colorbar()
            plt.axis('off')
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
        np.save(render_path, segment)
        return segment
    elif render_type == 'normal':
        normal = GL.glReadPixels(0, 0, width, height, GL.GL_RGB, GL.GL_UNSIGNED_BYTE, outputType=None)[::-1, :, :][
                 height - true_height:, :, :]
        if if_vis:
            plt.imshow(normal)
            plt.axis('off')
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
        np.save(render_path, normal)
        return normal
    elif render_type == 'depth':
        data = GL.glReadPixels(0, 0, width, height, GL.GL_DEPTH_COMPONENT, GL.GL_FLOAT,
                               outputType=None)  # read projected pixel info
        capturedImage = data
        for i in range(width):
            for j in range(height):
                if capturedImage[i][j] == 1.0:
                    capturedImage[i][j] = 20
                else:
                    far = FAR
                    near = 0.1
                    clip_z = (capturedImage[i][j] - 0.5) * 2.0
                    world_z = 2 * far * near / (clip_z * (far - near) - (far + near))
                    capturedImage[i][j] = -world_z  # -z#
        depth = capturedImage[::-1, :][height - true_height:, :]
        if if_vis:
            fig = plt.figure()
            ii = plt.imshow(depth, interpolation='nearest')
            # fig.colorbar(ii)
            plt.axis('off')
            plt.savefig(img_path, bbox_inches='tight')
            plt.close()
        np.save(render_path, depth)
        scipy.io.savemat(render_path + '.mat', mdict={'depth': depth})
        return depth


def dist_Point_to_Plane(point, plane_vector, plane_normal):
    sn = -np.dot(plane_normal, (point - plane_vector))
    sd = np.dot(plane_normal, plane_normal)
    sb = sn / sd

    point_b = point + sb * plane_normal
    return np.linalg.norm(point - point_b)


class OBJ_NORMAL:
    def __init__(self, obj_info, rotation_matrix, swapyz=False, layout_type=None):
        """Loads a Wavefront OBJ file. """
        self.vertices = obj_info.vertices
        self.normals = obj_info.normals
        self.texcoords = obj_info.texcoords
        self.faces = obj_info.faces
        self.rotation_matrix = rotation_matrix

        # load the mesh to OPENGL
        self.gl_list = GL.glGenLists(1)
        GL.glNewList(self.gl_list, GL.GL_COMPILE)
        for face in self.faces:
            vertices, normals, _, _ = face
            GL.glBegin(GL.GL_POLYGON)
            for i in range(len(vertices)):
                normal_color = self.rotation_matrix.dot(self.normals[normals[i] - 1])
                GL.glColor3f((normal_color[0] + 1) / 2, (-normal_color[2] + 1) / 2, (normal_color[1] + 1) / 2)
                GL.glVertex3fv(self.vertices[vertices[i] - 1])
            GL.glEnd()
        GL.glEndList()


class OBJ_SEG:
    def __init__(self, obj_info, color=None, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = obj_info.vertices
        self.normals = obj_info.normals
        self.texcoords = obj_info.texcoords
        self.faces = obj_info.faces
        self.color = color

        self.gl_list = GL.glGenLists(1)
        GL.glNewList(self.gl_list, GL.GL_COMPILE)
        # GL.glEnable(GL.GL_TEXTURE_2D)
        GL.glFrontFace(GL.GL_CCW)
        # GL.glDisable(GL.GL_LIGHT0)
        # GL.glDisable(GL.GL_LIGHTING)
        for face in self.faces:
            vertices, normals, _, _ = face
            GL.glColor3f(self.color[0], self.color[1], self.color[2])
            GL.glBegin(GL.GL_POLYGON)
            for i in range(len(vertices)):
                # if normals[i] > 0:
                #     GL.glNormal3fv(self.normals[normals[i] - 1])
                GL.glVertex3fv(self.vertices[vertices[i] - 1])
            GL.glEnd()
        GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEndList()


class OBJ:
    def __init__(self, obj_info, obj_color=None, swapyz=False):
        """Loads a Wavefront OBJ file. """
        self.vertices = obj_info.vertices
        self.normals = obj_info.normals
        self.texcoords = obj_info.texcoords
        self.faces = obj_info.faces

        self.gl_list = GL.glGenLists(1)
        GL.glNewList(self.gl_list, GL.GL_COMPILE)
        for face in self.faces:
            vertices, normals, _, texture_coords = face
            GL.glBegin(GL.GL_POLYGON)
            for i in range(len(vertices)):
                GL.glVertex3fv(self.vertices[vertices[i] - 1])
            GL.glEnd()
        # GL.glDisable(GL.GL_TEXTURE_2D)
        GL.glEndList()


'''
find default coordinates of object
'''


def findCoordinates(filename):
    vertices = []
    for line in open(filename, "r"):
        if line.startswith('#'): continue
        values = line.split()
        if not values: continue
        if values[0] == 'v':
            if values[1] == '-nan' or values[1] == 'nan':
                print 'nan appears'
                continue
            v = map(float, values[1:4])
            vertices.append(v)

    min_x = max_x = vertices[0][0]
    min_y = max_y = vertices[0][1]
    min_z = max_z = vertices[0][2]

    for i in range(len(vertices)):
        if vertices[i][0] < min_x: min_x = vertices[i][0]
        if vertices[i][0] > max_x: max_x = vertices[i][0]
        if vertices[i][1] < min_y: min_y = vertices[i][1]
        if vertices[i][1] > max_y: max_y = vertices[i][1]
        if vertices[i][2] < min_z: min_z = vertices[i][2]
        if vertices[i][2] > max_z: max_z = vertices[i][2]

    return min_x, max_x, min_y, max_y, min_z, max_z


def init_world(lamda, beta, w, h, render_type):
    if render_type == 'rgb':
        light_ambient = [0.0, 0.0, 0.0, 1.0]
        light_diffuse = [1.0, 1.0, 1.0, 1.0]
        light_specular = [1.0, 1.0, 1.0, 1.0]
        light_position0 = [1.0, 1.0, 0.0, 0.0]
        light_position1 = [-1.0, -1.0, 0.0, 0.0]
        light_position2 = [1.0, -1.0, 0.0, 0.0]
        light_position3 = [-1.0, 1.0, 0.0, 0.0]

        GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, light_ambient)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, light_specular)
        GL.glMaterialfv(GL.GL_FRONT, GL.GL_SHININESS, 10.0)
        GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position0)

        GL.glLightfv(GL.GL_LIGHT1, GL.GL_AMBIENT, light_ambient)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_SPECULAR, light_specular)
        GL.glLightfv(GL.GL_LIGHT1, GL.GL_POSITION, light_position1)

        GL.glLightfv(GL.GL_LIGHT2, GL.GL_AMBIENT, light_ambient)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_SPECULAR, light_specular)
        GL.glLightfv(GL.GL_LIGHT2, GL.GL_POSITION, light_position2)

        GL.glLightfv(GL.GL_LIGHT3, GL.GL_AMBIENT, light_ambient)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_DIFFUSE, light_diffuse)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_SPECULAR, light_specular)
        GL.glLightfv(GL.GL_LIGHT3, GL.GL_POSITION, light_position3)

        GL.glEnable(GL.GL_LIGHTING)
        GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_LIGHT1)
        GL.glEnable(GL.GL_LIGHT2)
        GL.glEnable(GL.GL_LIGHT3)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glColorMaterial(GL.GL_FRONT_AND_BACK, GL.GL_AMBIENT_AND_DIFFUSE)
        GL.glEnable(GL.GL_COLOR_MATERIAL)
        GL.glEnable(GL.GL_NORMALIZE)
    else:
        GL.glPixelStoref(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glPixelStoref(GL.GL_PACK_ALIGNMENT, 1)
        # light_ambient = [0.0, 0.0, 0.0, 1.0]
        # light_diffuse = [1.0, 1.0, 1.0, 1.0]
        # light_specular = [1.0, 1.0, 1.0, 1.0]
        # light_position = [1.0, 1.0, 1.0, 0.0]
        #
        # GL.glLightfv(GL.GL_LIGHT0, GL.GL_AMBIENT, light_ambient)
        # GL.glLightfv(GL.GL_LIGHT0, GL.GL_DIFFUSE, light_diffuse)
        # GL.glLightfv(GL.GL_LIGHT0, GL.GL_SPECULAR, light_specular)
        # GL.glLightfv(GL.GL_LIGHT0, GL.GL_POSITION, light_position)
        #
        # GL.glEnable(GL.GL_LIGHTING)
        # GL.glEnable(GL.GL_LIGHT0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        # GL.glEnable(GL.GL_COLOR_MATERIAL)

    GL.glViewport(0, 0, w, h)

    GL.glMatrixMode(GL.GL_PROJECTION)   # camera -> image intrinsic parameter
    GL.glLoadIdentity()
    fov = (np.arctan(h / 2 / lamda) * 2) * 180 / math.pi
    GLU.gluPerspective(fov, float(w) / h, 0.1, FAR)

    GL.glMatrixMode(GL.GL_MODELVIEW)    # world -> camera
    GL.glLoadIdentity()
    GL.glRotate(beta, 1, 0, 0)
    GL.glClearColor(1.0, 1.0, 1.0, 1.0)
    GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)


def add_object_normal(filename, xyz, unit, up_string, front_string, center, angle, size, beta, obj_info, layout_type=None):
    GL.glPushMatrix()
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    del rotation_matrix_list[:]
    # translate in view
    if layout_type is not None:
        GL.glTranslatef(center[0], center[1], center[2])
    else:
        GL.glTranslatef(center[0], center[1], center[2] - size[2] / 2)
    GL.glRotate(angle, 0, 1, 0)
    rotation_matrix_list.append(rotation_matrix(angle, 0, 1, 0))
    # scale object
    GL.glScalef(size[0] * 100 / x, size[1] * 100 / y, size[2] * 100 / z)
    # normalize directions, rotate object to front: front -> 0,0,1 | up -> 0,1,0
    alignment = align_directions(up_string, front_string)
    if not alignment:
        GL.glPopMatrix()
        return
    rm = np.identity(3)
    for i in range(len(rotation_matrix_list) - 1, -1, -1):
        rm = rotation_matrix_list[i].dot(rm)
    rm = rotation_matrix(beta, 1, 0, 0).dot(rm)
    obj = OBJ_NORMAL(obj_info['info'], rotation_matrix=rm, swapyz=False, layout_type=layout_type)
    GL.glCallList(obj.gl_list)
    GL.glPopMatrix()


def add_object_seg(filename, xyz, unit, up_string, front_string, center, angle, size, segment_id, obj_info, layout_type=None):
    GL.glPushMatrix()
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    # translate in view
    if layout_type is not None:
        GL.glTranslatef(center[0], center[1], center[2])
    else:
        GL.glTranslatef(center[0], center[1], center[2] - size[2] / 2)
    GL.glRotate(angle, 0, 1, 0)
    GL.glScalef(size[0] * 100 / x, size[1] * 100 / y, size[2] * 100 / z)
    # normalize directions, rotate object to front: front -> 0,0,1 | up -> 0,1,0
    alignment = align_directions(up_string, front_string)
    if not alignment:
        GL.glPopMatrix()
        return
    # obj = OBJ_NORMAL(filename, swapyz=False)
    obj = OBJ_SEG(obj_info['info'], color=[segment_id / 255.0, 0.0, 0.0], swapyz=False)
    GL.glCallList(obj.gl_list)
    GL.glPopMatrix()


# angle is the object rotation angle
def add_object(filename, xyz, unit, up_string, front_string, center, angle, size, obj_color, obj_info, layout_type=None):  # x_min,y_min,z_min,x_max,y_max,z_max
    GL.glPushMatrix()
    # after unit conversion...
    x = xyz[0]
    y = xyz[1]
    z = xyz[2]
    # print center, size
    # translate in view
    if layout_type is not None:
        GL.glTranslatef(center[0], center[1], center[2])
    else:
        GL.glTranslatef(center[0], center[1], center[2] - size[2] / 2)
    GL.glRotate(angle, 0, 1, 0)
    GL.glScalef(size[0] * 100 / x, size[1] * 100 / y, size[2] * 100 / z)   # x,y,z y is up
    # normalize directions, rotate object to front: front -> 0,0,1 | up -> 0,1,0
    alignment = align_directions(up_string, front_string)
    if not alignment:
        GL.glPopMatrix()
        return
    # obj = OBJ_NORMAL(filename, swapyz=False)
    obj = OBJ(obj_info['info'], obj_color=obj_color, swapyz=False)
    GL.glCallList(obj.gl_list)
    GL.glPopMatrix()


# read metadata info from csv file
def read_metadata(name):
    with open(os.path.join(MODEL_PATH, 'metadata_fine.csv'), "r+") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            if row[0] == "wss." + name:
                return row


def alignment_check(up_string, front_string):
    if up_string == "" and front_string == "":
        align_flag = 1
        # up y+
    elif up_string == "0\,1\,0":
        if front_string == "0\,0\,1":
            align_flag = 1
        elif front_string == "0\,0\,-1":
            align_flag = 1
        elif front_string == "-1\,0\,0":
            align_flag = 1
        elif front_string == "1\,0\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0
            # up y-
    elif up_string == "0\,-1\,0":
        if front_string == "0\,0\,1":
            align_flag = 1
        elif front_string == "1\,0\,0":
            align_flag = 1
        elif front_string == "-1\,0\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0
            # up z+
    elif up_string == "0\,0\,1":
        if front_string == "-1\,0\,0":
            align_flag = 1
        elif front_string == "1\,0\,0":
            align_flag = 1
        elif front_string == "0\,1\,0":
            align_flag = 1
        elif front_string == "0\,-1\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0


            # up z- with problem
    elif up_string == "0\,0\,-1":
        if front_string == "1\,0\,0":
            align_flag = 1
        elif front_string == "-1\,0\,0":
            align_flag = 1
        elif front_string == "0\,-1\,0":
            align_flag = 1
        elif front_string == "0\,1\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0

            # up x+
    elif up_string == "1\,0\,0":
        if front_string == "0\,0\,1":
            align_flag = 1
        elif front_string == "0\,0\,-1":
            align_flag = 1
        elif front_string == "0\,1\,0":
            align_flag = 1
        elif front_string == "0\,-1\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0

            # up x-
    elif up_string == "-1\,0\,0":
        if front_string == "0\,0\,1":
            align_flag = 1
        elif front_string == "0\,0\,-1":
            align_flag = 1
        elif front_string == "0\,1\,0":
            align_flag = 1
        elif front_string == "0\,-1\,0":
            align_flag = 1
        else:
            print 'cannot align'
            align_flag = 0
    else:
        print 'cannot align'
        align_flag = 0
    if align_flag == 0:
        return False
    else:
        return True

# rotate object to face front
def align_directions(up_string, front_string):
    # front 0,-1,0, up 0,0,1
    if up_string == "" and front_string == "":
        GL.glRotate(-90, 1, 0, 0)
        rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
    # up y+
    elif up_string == "0\,1\,0":
        if front_string == "0\,0\,1":
            pass
        elif front_string == "0\,0\,-1":
            GL.glRotate(180, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(180, 0, 1, 0))
        elif front_string == "-1\,0\,0":
            GL.glRotate(90, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 1, 0))
        elif front_string == "1\,0\,0":
            GL.glRotate(-90, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(-90, 0, 1, 0))
        else:
            print 'cannot align'
            return False

    # up y-
    elif up_string == "0\,-1\,0":
        GL.glRotate(180, 1, 0, 0)
        rotation_matrix_list.append(rotation_matrix(180, 1, 0, 0))
        if front_string == "0\,0\,1":
            GL.glRotate(180, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(180, 0, 1, 0))
        elif front_string == "1\,0\,0":
            GL.glRotate(90, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 1, 0))
        elif front_string == "-1\,0\,0":
            GL.glRotate(-90, 0, 1, 0)
            rotation_matrix_list.append(rotation_matrix(-90, 0, 1, 0))
        else:
            print 'cannot align'
            return False

    # up z+
    elif up_string == "0\,0\,1":
        if front_string == "-1\,0\,0":
            GL.glRotate(-90, 1, 0, 0)
            GL.glRotate(90, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
        elif front_string == "1\,0\,0":
            GL.glRotate(-90, 1, 0, 0)
            GL.glRotate(-90, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(-90, 0, 0, 1))
        elif front_string == "0\,1\,0":
            GL.glRotate(-90, 1, 0, 0)
            GL.glRotate(180, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(180, 0, 0, 1))
        elif front_string == "0\,-1\,0":
            GL.glRotate(-90, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
        else:
            print 'cannot align'
            return False


    # up z- with problem
    elif up_string == "0\,0\,-1":
        if front_string == "1\,0\,0":
            GL.glRotate(90, 1, 0, 0)
            GL.glRotate(90, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
        elif front_string == "-1\,0\,0":
            GL.glRotate(90, 1, 0, 0)
            GL.glRotate(-90, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(-90, 0, 0, 1))
        elif front_string == "0\,-1\,0":
            GL.glRotate(90, 1, 0, 0)
            GL.glRotate(180, 0, 0, 1)  # front
            rotation_matrix_list.append(rotation_matrix(90, 1, 0, 0))
            rotation_matrix_list.append(rotation_matrix(180, 0, 0, 1))
        elif front_string == "0\,1\,0":
            GL.glRotate(90, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 1, 0, 0))
        else:
            print 'cannot align'
            return False

    # up x+
    elif up_string == "1\,0\,0":
        if front_string == "0\,0\,1":
            GL.glRotate(90, 0, 0, 1)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
        elif front_string == "0\,0\,-1":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(180, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(180, 1, 0, 0))
        elif front_string == "0\,1\,0":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(90, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(90, 1, 0, 0))
        elif front_string == "0\,-1\,0":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(-90, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
        else:
            print 'cannot align'
            return False

            # up x-
    elif up_string == "-1\,0\,0":
        if front_string == "0\,0\,1":
            GL.glRotate(90, 0, 0, 1)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
        elif front_string == "0\,0\,-1":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(180, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(180, 1, 0, 0))
        elif front_string == "0\,1\,0":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(180, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(180, 1, 0, 0))
        elif front_string == "0\,-1\,0":
            GL.glRotate(90, 0, 0, 1)
            GL.glRotate(-90, 1, 0, 0)
            rotation_matrix_list.append(rotation_matrix(90, 0, 0, 1))
            rotation_matrix_list.append(rotation_matrix(-90, 1, 0, 0))
        else:
            print 'cannot align'
            return False
    else:
        print 'cannot align'
        return False
    return True


# render individual object
def render_object(file_name, center, angle, beta, size, segment_id, obj_color, render_type='rgb', obj_info=None):
    # get name id
    arr = file_name.split("/")
    name = arr[len(arr) - 1].split(".obj")[0]

    # read dimensions from metadata
    row = read_metadata(name)
    if row is None:
        return

    # get scale
    unit = row[6]
    if unit == "":
        unit = 1

    # get dimensions
    dim_arr = row[7]
    dim = []
    for a in dim_arr.split("\,"):
        dim.append(float(a))

    up_string = row[4]
    front_string = row[5]

    if render_type == 'rgb' or render_type == 'depth':
        add_object(file_name, dim, float(unit), up_string, front_string, center, angle, size, obj_color, obj_info)
    elif render_type == 'segmentation':
        add_object_seg(file_name, dim, float(unit), up_string, front_string, center, angle, size, segment_id, obj_info)
    elif render_type == 'normal':
        add_object_normal(file_name, dim, float(unit), up_string, front_string, center, angle, size, beta, obj_info)


def render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, segment_id, beta, render_type, obj_info, layout_type):
    obj_color = [255, 0, 0]
    angle = layout_angle
    if render_type == 'rgb' or render_type == 'depth':
        add_object(file_name, dim, float(unit), up_string, front_string, center, angle, size, obj_color, obj_info, layout_type=layout_type)
    elif render_type == 'segmentation':
        add_object_seg(file_name, dim, float(unit), up_string, front_string, center, angle, size, segment_id, obj_info, layout_type=layout_type)
    elif render_type == 'normal':
        add_object_normal(file_name, dim, float(unit), up_string, front_string, center, angle, size, beta, obj_info, layout_type=layout_type)


def get_obj_info(name):
    row = read_metadata(name)
    if row is None:
        return

    # get scale
    unit = row[6]
    if unit == "":
        unit = 1

    # get dimensions
    dim_arr = row[7]
    dim = []
    for a in dim_arr.split("\,"):
        dim.append(float(a))

    up_string = row[4]
    front_string = row[5]
    return dim, unit, up_string, front_string


def render_layout(beta, layouts, render_type='rgb', obj_info=None):
    # print 'rendering layouts'
    # read dimensions from metadata

    # get rotation angle
    mwall = layouts.mwall
    # get layout angle, just similar or object angle
    v1 = np.array([mwall[1][0] - mwall[0][0], mwall[1][2] - mwall[0][2]])
    v2 = np.array([1, 0])
    layout_angle = (get_angle(v1, v2, rotate=True) / np.pi) * 180
    # print layout_angle
    # render middle wall
    dim, unit, up_string, front_string = get_obj_info('suncgwall')
    file_name = os.path.join(MODEL_PATH, 'models', 'suncgwall.obj')

    mwall = layouts.mwall
    floor = layouts.floor
    ceiling = layouts.ceiling
    lwall = layouts.lwall
    rwall = layouts.rwall
    size_x = np.sqrt((mwall[0][0] - mwall[1][0]) ** 2 + (mwall[0][2] - mwall[1][2]) ** 2)
    size_y = mwall[2][1] - mwall[0][1]
    size_z = np.sqrt((floor[1][0] - floor[2][0]) ** 2 + (floor[1][2] - floor[2][2]) ** 2)

    center = (mwall[0] + mwall[2]) / 2
    size = np.array([size_x, size_y, 0.2])
    render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, 1, beta, render_type, obj_info[0], 'mwall')
    # render floor
    dim, unit, up_string, front_string = get_obj_info('suncgfloor')
    file_name = os.path.join(MODEL_PATH, MODEL_SUBPATH, 'suncgfloor.obj')
    center = (floor[0] + floor[2]) / 2
    size = np.array([size_x, 0.2, size_z])
    render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, 2, beta, render_type,  obj_info[1], 'floor')
    # render ceiling
    dim, unit, up_string, front_string = get_obj_info('suncgfloor')
    file_name = os.path.join(MODEL_PATH, MODEL_SUBPATH, 'suncgfloor.obj')
    center = (ceiling[0] + ceiling[2]) / 2
    size = np.array([size_x, 0.2, size_z])
    render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, 22, beta, render_type,  obj_info[2], 'ceiling')

    # render left wall
    dim, unit, up_string, front_string = get_obj_info('suncgsidewall')
    file_name = os.path.join(MODEL_PATH, MODEL_SUBPATH, 'suncgsidewall.obj')
    center = (lwall[0] + lwall[2]) / 2
    size = np.array([0.2, size_y, size_z])
    render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, 1, beta, render_type,  obj_info[3], 'lwall')

    # render right wall
    dim, unit, up_string, front_string = get_obj_info('suncgsidewall')
    file_name = os.path.join(MODEL_PATH, MODEL_SUBPATH, 'suncgsidewall.obj')
    center = (rwall[0] + rwall[2]) / 2
    size = np.array([0.2, size_y, size_z])
    render_wall(file_name, dim, unit, up_string, front_string, center, size, layout_angle, 1, beta, render_type,  obj_info[4], 'rwall')


def render_scene(pg=None, pg_path=None, save_path=None, render_type='rgb', if_vis=True, obj_info=None):
    # print 'rendering object in %s mode' % render_type
    # load pg if pg are not in the input list
    if pg is None and pg_path is not None:
        with open(pg_path, 'r') as f:
            pg = pickle.load(f)
        f.close()
    # read pg
    camera = pg.camera
    K = camera.K
    lamda = K[0, 0]
    width = int(K[0, 2] * 2)
    height = int(K[1, 2] * 2)
    beta = camera.beta * 180 / math.pi
    objects = pg.objects
    # init opengl
    osmesa.init_ctx(width, height)
    init_world(lamda, beta, width, height, render_type)
    # render_objects
    for obj_index, object_ in enumerate(objects):
        segment_id = sunrgbd_obj_to_seg[sunrgbd_cate_all.index(object_.obj_type)]
        short_name = object_.terminal.obj_id
        center = object_.terminal.obj_center
        angle = object_.terminal.angle
        size = object_.terminal.obj_size
        color = object_.terminal.obj_color
        url = os.path.join(MODEL_PATH, MODEL_SUBPATH, short_name + '.obj')
        render_object(url, center, angle, beta, size, segment_id, color, render_type=render_type, obj_info=obj_info[obj_index])
    render_layout(beta, pg.layouts, render_type=render_type, obj_info=obj_info[-5:])
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img_path = os.path.join(save_path, '%s.png' % render_type)
    array_path = os.path.join(save_path, render_type)
    GL.glFlush()
    result = OnCaptureResult(array_path, img_path, width, width, height, if_vis, render_type=render_type)
    osmesa.free_ctx('end')
    return result

