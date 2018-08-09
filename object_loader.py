"""
Created on Oct 21, 2017

@author: Siyuan Huang

Loading the object into the memory

"""

import numpy as np
import os
import glob
from osmesa.render_scene import findCoordinates, get_obj_info
import time

class OBJ_SAVER_SIMPLIFIED:
    def __init__(self, filename, swapyz=False, layout_type=None):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.flipped = []
        self.lines = list()
        with open(filename, 'r') as f:
            for line in f:
                self.lines.append(line)
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    v = map(float, values[1:4])
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    v = map(float, values[1:4])
                    if swapyz:
                        v = v[0], v[2], v[1]
                    v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                    if v_norm == 0:
                        self.normals.append([1/3.0, 1/3.0, 1/3.0])
                    else:
                        if layout_type is not None:
                            self.normals.append([v[0] / v_norm, v[1] / v_norm, v[2] / v_norm])
                        if layout_type is None:
                            self.normals.append([-v[0] / v_norm, -v[1] / v_norm, -v[2] / v_norm])

                elif values[0] == 'vt':
                    self.texcoords.append(map(float, values[1:3]))
                elif values[0] == 'f':
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        # print w
                        face.append(int(w[0]))
                        if len(w) >= 2 and len(w[1]) > 0:
                            texcoords.append(int(w[1]))
                        else:
                            texcoords.append(0)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                    # print norms
                    self.faces.append((face, norms, '', texcoords))
        f.close()


class OBJ_SAVER:
    def __init__(self, filename, swapyz=False, layout_type=None):
        """Loads a Wavefront OBJ file. """
        self.vertices = []
        self.normals = []
        self.texcoords = []
        self.faces = []
        self.flipped = []
        self.lines = list()
        with open(filename, 'r') as f:
            for line in f:
                self.lines.append(line)
                if line.startswith('#'): continue
                values = line.split()
                if not values: continue
                if values[0] == 'v':
                    v = map(float, values[1:4])
                    if swapyz:
                        v = v[0], v[2], v[1]
                    self.vertices.append(v)
                elif values[0] == 'vn':
                    v = map(float, values[1:4])
                    if swapyz:
                        v = v[0], v[2], v[1]
                    v_norm = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
                    if v_norm == 0:
                        self.normals.append([1/3.0, 1/3.0, 1/3.0])
                    else:
                        self.normals.append([v[0] / v_norm, v[1] / v_norm, v[2] / v_norm])
                elif values[0] == 'vt':
                    self.texcoords.append(map(float, values[1:3]))
                elif values[0] in ('usemtl', 'usemat'):
                    material = values[1]
                elif values[0] == 'f':
                    face = []
                    texcoords = []
                    norms = []
                    for v in values[1:]:
                        w = v.split('/')
                        # print w
                        face.append(int(w[0]))
                        if len(w) >= 2 and len(w[1]) > 0:
                            texcoords.append(int(w[1]))
                        else:
                            texcoords.append(0)
                        if len(w) >= 3 and len(w[2]) > 0:
                            norms.append(int(w[2]))
                        else:
                            norms.append(0)
                    # print norms
                    self.faces.append((face, norms, material, texcoords))
        f.close()

        # if layout_type is None:
        #     # cluster the vertex groups
        #     group_record = list()
        #     index = 0
        #     with open(filename, 'r') as f:
        #         for line in f:
        #             index += 1
        #             if line.startswith('#'): continue
        #             values = line.split()
        #             if not values: continue
        #             if values[0] in ('usemtl', 'usemat'):
        #                 group_record.append(index)
        #         else:
        #             group_record.append(index + 1)
        #     mtl_groups = list()
        #     for index_mtl in range(len(group_record) - 1):
        #         start = (group_record[index_mtl] - group_record[0] - index_mtl) * 3
        #         end = (group_record[index_mtl + 1] - group_record[index_mtl] - 1) * 3 + start - 1
        #         mtl_groups.append((start, end))
        #     # print mtl_groups
        #     for index_mtl, mtl_group in enumerate(mtl_groups):
        #         # print 'processing %d material' % index_mtl
        #         min_x = max_x = self.vertices[mtl_group[0]][0]
        #         min_y = max_y = self.vertices[mtl_group[0]][1]
        #         min_z = max_z = self.vertices[mtl_group[0]][2]
        #         vertex_set = [self.vertices[mtl_group[0]] for _ in range(6)]
        #         normal_set = [self.normals[mtl_group[0]] for _ in range(6)]
        #         for i in range(mtl_group[0], mtl_group[1]):
        #             if self.vertices[i][0] < min_x:
        #                 min_x = self.vertices[i][0]
        #                 vertex_set[0] = self.vertices[i]
        #                 normal_set[0] = self.normals[i]
        #             if self.vertices[i][0] > max_x:
        #                 max_x = self.vertices[i][0]
        #                 vertex_set[1] = self.vertices[i]
        #                 normal_set[1] = self.normals[i]
        #             if self.vertices[i][1] < min_y:
        #                 min_y = self.vertices[i][1]
        #                 vertex_set[2] = self.vertices[i]
        #                 normal_set[2] = self.normals[i]
        #             if self.vertices[i][1] > max_y:
        #                 max_y = self.vertices[i][1]
        #                 vertex_set[3] = self.vertices[i]
        #                 normal_set[3] = self.normals[i]
        #             if self.vertices[i][2] < min_z:
        #                 min_z = self.vertices[i][2]
        #                 vertex_set[4] = self.vertices[i]
        #                 normal_set[4] = self.normals[i]
        #             if self.vertices[i][2] > max_z:
        #                 max_z = self.vertices[i][2]
        #                 vertex_set[5] = self.vertices[i]
        #                 normal_set[5] = self.normals[i]
        #         expand_scale = 1
        #         center_point = np.array([(min_x + max_x) / 2, (min_y + max_y) / 2, (min_z + max_z) / 2])
        #         dist_ori = 0
        #         dist_expand = 0
        #         for i in range(len(vertex_set)):
        #             point1 = np.array(vertex_set[i])
        #             normal1 = np.array(normal_set[i])
        #             point1_new = point1 + expand_scale * normal1
        #             dist_ori += np.linalg.norm(point1 - center_point) / 100
        #             dist_expand += np.linalg.norm(point1_new - center_point) / 100
        #         if dist_expand < dist_ori:
        #             self.flipped.append(1)
        #             # print 'flipping'
        #             for i in range(mtl_group[0], mtl_group[1] + 1):
        #                 self.normals[i] = [-self.normals[i][j] for j in range(3)]
        #         else:
        #             self.flipped.append(0)



