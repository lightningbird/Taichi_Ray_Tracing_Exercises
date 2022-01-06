import taichi as ti
from ray_tracing_models import Triangle, Polygon

PI = 3.14159265
Inf = 10e8

@ti.data_oriented
class Torus:
    def __init__(self, center, inside_point, up_normal, inside_radius, nU, nV, material, color, write_to_obj_file=False, obj_filename=''):
        # up_normal is the normal of the plane passing through the center of torus
        # splitting its interface into concentric circles
        # inside_point is one point that is on central line inside the torus body
        self.center = center
        self.inside_point = inside_point
        self.up_normal = up_normal
        self.outside_R = (inside_point - center).norm()
        self.inside_r = inside_radius
        self.material = material
        self.color = color
        self.nU = nU
        self.nV = nV
        self.write_to_obj_file = write_to_obj_file
        self.obj_filename = obj_filename
        self.num_polygons = self.nU * self.nV
        self.x_axis = (self.inside_point - self.center).normalized()
        self.y_axis = self.up_normal.normalized()
        self.z_axis = self.x_axis.cross(self.y_axis)
        self.polygons = []
        self.make_surface_mesh()
        if self.write_to_obj_file == True:
            self.write_mesh_to_file()
        self.aabb_faces = []
        self.aabb_boundingbox()

    def aabb_boundingbox(self):
        # build axis align bounding box in general case
        vts = []
        radius = self.outside_R + self.inside_r
        upper_c = self.center + self.inside_r * self.y_axis
        lower_c = self.center - self.inside_r * self.y_axis
        offset1 = -radius * self.x_axis - radius * self.z_axis
        offset2 = -radius * self.x_axis + radius * self.z_axis
        offset3 = radius * self.x_axis + radius * self.z_axis
        offset4 = radius * self.x_axis - radius * self.z_axis
        vts = [upper_c+offset1, upper_c+offset2, upper_c+offset3, upper_c+offset4,
            lower_c+offset1, lower_c+offset2, lower_c+offset3, lower_c+offset4]

        for i in range(3):
            min_val, max_val = vts[0][i], vts[0][i]
            for j in range(8):
                if vts[j][i] < min_val:
                    min_val = vts[j][i]
                if vts[j][i] > max_val:
                    max_val = vts[j][i]
            self.aabb_faces.append(min_val)
            self.aabb_faces.append(max_val)

    def make_surface_mesh(self):
        for i in range(self.nU):
            for j in range(self.nV):
                theta = i * 2*PI/self.nU
                theta2 = theta + 2*PI/self.nU
                phi = j * 2*PI/self.nV
                phi2 = phi + 2*PI/self.nV
                xx_axis = ti.cos(theta)*self.x_axis + ti.sin(theta)*self.z_axis
                xx_axis2 = ti.cos(theta2)*self.x_axis + ti.sin(theta2)*self.z_axis
                oc = self.center + self.outside_R * xx_axis
                oc2 = self.center + self.outside_R * xx_axis2
                pt1 = oc + self.inside_r * (ti.cos(phi)*xx_axis + ti.sin(phi)*self.y_axis) # i, j
                pt2 = oc + self.inside_r * (ti.cos(phi2)*xx_axis + ti.sin(phi2)*self.y_axis) # i, j+1
                pt3 = oc2 + self.inside_r * (ti.cos(phi2)*xx_axis2 + ti.sin(phi2)*self.y_axis) # i+1, j+1
                pt4 = oc2 + self.inside_r * (ti.cos(phi)*xx_axis2 + ti.sin(phi)*self.y_axis) # i+1, j
                vts = [pt1, pt2, pt3, pt4]
                self.polygons.append(Polygon(vertices = vts, material = self.material, color = self.color))

    def write_mesh_to_file(self):
    # vertex index starts from 1
        with open(self.obj_filename, 'w') as f:
            f.write('# torus \n')
            f.write('# number of vertices is ' + str(self.num_polygons) + '\n')
            f.write('# number of faces is ' + str(self.num_polygons) + '\n')
            for i in range(self.nU):
                for j in range(self.nV):
                    index = i*self.nV + j
                    vt = self.polygons[index].vertices[0]
                    f.write('v ' + str(vt[0]) + ' ' + str(vt[1]) + ' ' + str(vt[2]) + '\n')
            for i in range(self.nU):
                for j in range(self.nV):
                    i2, j2 = i+1, j+1
                    if i2==self.nU:
                        i2 = 0
                    if j2==self.nV:
                        j2 = 0
                    id1 = i*self.nV + j + 1
                    id2 = i*self.nV + j2 + 1
                    id3 = i2*self.nV + j2 + 1
                    id4 = i2*self.nV + j + 1
                    f.write('f ' + str(id1) + ' ' + str(id2) + ' ' + str(id3) + ' ' + str(id4) + '\n' )

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        front_face = False
        closest_t = tmax
        # intersect with aabb first
        dir = ray.direction
        origin = ray.origin
        insect_with_box = False
        box_tmin = tmin
        box_tmax = tmax
        for i in ti.static(range(3)):
            if dir[i] > 1e-5 or dir[i] < -1e-5:
                t0 = (self.aabb_faces[2*i] - origin[i]) / dir[i]
                t1 = (self.aabb_faces[2*i+1] - origin[i]) / dir[i]
                if t0>t1:
                    t0, t1 = t1, t0
                if t0 > box_tmin:
                    box_tmin = t0
                if t1 < box_tmax:
                    box_tmax = t1
        if box_tmin < box_tmax and box_tmax>tmin and box_tmin < closest_t:
            insect_with_box = True

        if insect_with_box == True:
            # iterate over polygon faces
            for i in ti.static(range(self.num_polygons)):
                is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = self.polygons[i].hit(ray, tmin, closest_t)
                if is_hit_tmp:
                    closest_t = root_tmp
                    is_hit = is_hit_tmp
                    hit_point = hit_point_tmp
                    hit_point_normal = hit_point_normal_tmp
                    front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Mesh:
    def __init__(self, obj_filename, material, color):
        self.obj_filename = obj_filename
        self.material = material
        self.color = color
        self.polygons = []
        self.read_obj_file()
        self.num_polygons = len(self.polygons)
        self.aabb_faces = []
        self.aabb_boundingbox()
    
    def read_obj_file(self):
    # vertex id in obj file starts from 1
        with open(self.obj_filename, 'r') as f:
            lines = f.readlines()
    
        vts = []
        for i in range(len(lines)):
            line = lines[i]
            x = line.split(' ')
            if x[0] == 'v':
                vts.append(ti.Vector([float(x[1]), float(x[2]), float(x[3]) ]))
        for i in range(len(lines)):
            line = lines[i]
            y = line.split(' ')
            if y[0] == 'f':
                fvts = []
                for j in range(len(y)-1):
                    vid = int(y[j+1]) - 1
                    fvts.append(vts[vid])
                self.polygons.append(Polygon(vertices = fvts, material = self.material, color = self.color))

    
    def aabb_boundingbox(self):
        pass
    
    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        front_face = False
        closest_t = tmax
        # iterate over polygon faces
        for i in ti.static(range(self.num_polygons)):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = self.polygons[i].hit(ray, tmin, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color