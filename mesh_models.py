import taichi as ti

PI = 3.14159265
Inf = 10e8
Epsilon = 1e-5

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
        self.vertex1 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex2 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex3 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex4 = ti.Vector.field(3, dtype = ti.f32)
        self.normal = ti.Vector.field(3, dtype = ti.f32)
        self.polygons_node = ti.root.dense(ti.ij, (self.nU, self.nV))
        self.polygons_node.place(self.vertex1, self.vertex2, self.vertex3, self.vertex4, self.normal)

        self.x_axis = (self.inside_point - self.center).normalized()
        self.y_axis = self.up_normal.normalized()
        self.z_axis = self.x_axis.cross(self.y_axis)
        self.make_surface_mesh()
        if self.write_to_obj_file == True:
            self.write_mesh_to_file()
        self.aabb_faces = [] # xmin, xmax, ymin, ymax, zmin, zmax
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
                self.vertex1[i, j] = pt1
                self.vertex2[i, j] = pt2
                self.vertex3[i, j] = pt3
                self.vertex4[i, j] = pt4
                self.normal[i, j] = (pt2 - pt1).cross(pt3 - pt2).normalized()

    def write_mesh_to_file(self):
    # vertex index starts from 1
        with open(self.obj_filename, 'w') as f:
            f.write('# torus \n')
            f.write('# number of vertices is ' + str(self.num_polygons) + '\n')
            f.write('# number of faces is ' + str(self.num_polygons) + '\n')
            for i in range(self.nU):
                for j in range(self.nV):
                    pt1 = self.vertex1[i, j]
                    f.write('v ' + str(pt1[0]) + ' ' + str(pt1[1]) + ' ' + str(pt1[2]) + '\n')
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
    def inside_check(self, i, j, point):
        v1 = self.vertex1[i, j] - point
        v2 = self.vertex2[i, j] - point
        v3 = self.vertex3[i, j] - point
        v4 = self.vertex4[i, j] - point
        n1 = v1.cross(v2)
        n2 = v2.cross(v3)
        n3 = v3.cross(v4)
        n4 = v4.cross(v1)
        isInside = False
        if n1.dot(n2) > 0 and n1.dot(n3) > 0 and n1.dot(n4)>0:
            isInside = True
        return isInside

    @ti.func
    def hit_quad(self, i, j, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal[i, j]
        p = self.vertex1[i, j]
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -n
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                if self.inside_check(i, j, hit_point):
                    is_hit = True
                    if d.dot(n) < 0:
                        front_face = True
                        hit_point_normal = n
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

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
            if dir[i] > Epsilon or dir[i] < -Epsilon:
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
            for i in range(self.nU):
                for j in range(self.nV):
                    is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = self.hit_quad(i, j, ray, tmin, closest_t)
                    if is_hit_tmp:
                        closest_t = root_tmp
                        is_hit = is_hit_tmp
                        hit_point = hit_point_tmp
                        hit_point_normal = hit_point_normal_tmp
                        front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Quad_Mesh:
    def __init__(self, obj_filename, material, color, center=ti.Vector([0.0, 0.0, 0.0]), scale=1.0, tx=0.0, ty=0.0, tz=0.0):
        self.obj_filename = obj_filename
        self.material = material
        self.color = color
        self.center = center
        self.scale = scale
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.max_num_polygons= 4000
        self.num_polygons = 0
        self.vertex1 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex2 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex3 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex4 = ti.Vector.field(3, dtype = ti.f32)
        self.normal = ti.Vector.field(3, dtype = ti.f32)
        self.polygons_node = ti.root.dense(ti.i, self.max_num_polygons)
        self.polygons_node.place(self.vertex1, self.vertex2, self.vertex3, self.vertex4, self.normal)
        self.aabb_faces = ti.Vector.field(6, dtype=ti.f32, shape=()) # xmin, xmax, ymin, ymax, zmin, zmax
        self.read_obj_file()

    @ti.pyfunc
    def vertex_update_aabb(self, v, xmin, xmax, ymin, ymax, zmin, zmax):
        if v[0] < xmin:
            xmin = v[0]
        if v[0] > xmax:
            xmax = v[0]
        if v[1] < ymin:
            ymin = v[1]
        if v[1] > ymax:
            ymax = v[1]
        if v[2] < zmin:
            zmin = v[2]
        if v[2] > zmax:
            zmax = v[2]
        return xmin, xmax, ymin, ymax, zmin, zmax

    def read_obj_file(self):
    # vertex id in obj file starts from 1
        with open(self.obj_filename, 'r') as f:
            lines = f.readlines()
        vts = []
        num_faces = 0
        xmin, xmax, ymin, ymax, zmin, zmax = Inf, -Inf, Inf, -Inf, Inf, -Inf
        for i in range(len(lines)):
            line = lines[i]
            x = line.split(' ')
            if x[0] == 'v':
                vx = float(x[1])
                vy = float(x[2])
                vz = float(x[3])
                vnew = ti.Vector([vx, vy, vz]) - self.center
                vnew = self.scale * vnew
                vnew = vnew + self.center + ti.Vector([self.tx, self.ty, self.tz])
                vts.append(vnew)
                xmin, xmax, ymin, ymax, zmin, zmax = self.vertex_update_aabb(vnew, xmin, xmax, ymin, ymax, zmin, zmax)
            if x[0] == 'f':
                num_faces +=1
        self.aabb_faces[None] = ti.Vector([xmin, xmax, ymin, ymax, zmin, zmax])

        assert num_faces <= self.max_num_polygons
        fid = 0
        for i in range(len(lines)):
            line = lines[i]
            y = line.split(' ')
            if y[0] == 'f':
                vid1 = int(y[1])
                vid2 = int(y[2])
                vid3 = int(y[3])
                vid4 = int(y[4])
                self.vertex1[fid] = vts[vid1 - 1]
                self.vertex2[fid] = vts[vid2 - 1]
                self.vertex3[fid] = vts[vid3 - 1]
                self.vertex4[fid] = vts[vid4 - 1]
                self.normal[fid] = (vts[vid2-1] - vts[vid1-1]).cross(vts[vid3-1] - vts[vid2-1]).normalized()
                fid += 1
        self.num_polygons = fid

    @ti.func
    def inside_check(self, i, point):
        v1 = self.vertex1[i] - point
        v2 = self.vertex2[i] - point
        v3 = self.vertex3[i] - point
        v4 = self.vertex4[i] - point
        n1 = v1.cross(v2)
        n2 = v2.cross(v3)
        n3 = v3.cross(v4)
        n4 = v4.cross(v1)
        isInside = False
        if n1.dot(n2) > 0 and n1.dot(n3) > 0 and n1.dot(n4)>0:
            isInside = True
        return isInside

    @ti.func
    def hit_quad(self, i, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal[i]
        p = self.vertex1[i]
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -n
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                if self.inside_check(i, hit_point):
                    is_hit = True
                    if d.dot(n) < 0:
                        front_face = True
                        hit_point_normal = n
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

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
            if dir[i] > Epsilon or dir[i] < -Epsilon:
                t0 = (self.aabb_faces[None][2*i] - origin[i]) / dir[i]
                t1 = (self.aabb_faces[None][2*i+1] - origin[i]) / dir[i]
                if t0>t1:
                    t0, t1 = t1, t0
                if t0 > box_tmin:
                    box_tmin = t0
                if t1 < box_tmax:
                    box_tmax = t1
        if box_tmin < box_tmax and box_tmax>tmin and box_tmin < closest_t:
            insect_with_box = True
        # iterate over polygon faces
        if insect_with_box == True:
            for i in range(self.num_polygons):
                is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = self.hit_quad(i, ray, tmin, closest_t)
                if is_hit_tmp:
                    closest_t = root_tmp
                    is_hit = is_hit_tmp
                    hit_point = hit_point_tmp
                    hit_point_normal = hit_point_normal_tmp
                    front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Triangle_Mesh:
    def __init__(self, obj_filename, material, color, center=ti.Vector([0.0, 0.0, 0.0]), scale=1.0, tx=0.0, ty=0.0, tz=0.0):
        self.obj_filename = obj_filename
        self.material = material
        self.color = color
        self.max_num_polygons= 10000
        self.num_polygons = 0
        self.center = center
        self.scale = scale
        self.tx = tx
        self.ty = ty
        self.tz = tz
        self.vertex1 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex2 = ti.Vector.field(3, dtype = ti.f32)
        self.vertex3 = ti.Vector.field(3, dtype = ti.f32)
        self.normal = ti.Vector.field(3, dtype = ti.f32)
        self.polygons_node = ti.root.dense(ti.i, self.max_num_polygons)
        self.polygons_node.place(self.vertex1, self.vertex2, self.vertex3, self.normal)

        self.large_aabb_faces = ti.Vector.field(6, dtype = ti.f32, shape=()) # xmin, xmax, ymin, ymax, zmin, zmax
        self.num_small_aabb = 50
        self.small_aabb_faces = ti.Vector.field(6, dtype = ti.f32, shape=self.num_small_aabb)
        self.num_polygons_per_box = 1
        self.read_obj_file()
        self.num_polygons_per_box = self.num_polygons // self.num_small_aabb + 1
        self.build_small_aabb()

    @ti.pyfunc
    def vertex_update_aabb(self, v, xmin, xmax, ymin, ymax, zmin, zmax):
        if v[0] < xmin:
            xmin = v[0]
        if v[0] > xmax:
            xmax = v[0]
        if v[1] < ymin:
            ymin = v[1]
        if v[1] > ymax:
            ymax = v[1]
        if v[2] < zmin:
            zmin = v[2]
        if v[2] > zmax:
            zmax = v[2]
        return xmin, xmax, ymin, ymax, zmin, zmax

    def read_obj_file(self):
    # vertex id in obj file starts from 1
        with open(self.obj_filename, 'r') as f:
            lines = f.readlines()
        xmin, xmax, ymin, ymax, zmin, zmax = Inf, -Inf, Inf, -Inf, Inf, -Inf
        vts = []
        num_faces = 0
        for i in range(len(lines)):
            line = lines[i]
            x = line.split(' ')
            if x[0] == 'v':
                vx = float(x[1])
                vy = float(x[2])
                vz = float(x[3])
                vnew = ti.Vector([vx, vy, vz]) - self.center
                vnew = self.scale * vnew
                vnew = vnew + self.center + ti.Vector([self.tx, self.ty, self.tz])
                vts.append(vnew)
                xmin, xmax, ymin, ymax, zmin, zmax = self.vertex_update_aabb(vnew, xmin, xmax, ymin, ymax, zmin, zmax)
            if x[0] == 'f':
                num_faces += 1
        self.large_aabb_faces[None] = ti.Vector([xmin, xmax, ymin, ymax, zmin, zmax])

        assert num_faces <= self.max_num_polygons
        fid = 0
        for i in range(len(lines)):
            line = lines[i]
            y = line.split(' ')
            if y[0] == 'f':
                vid1 = int(y[1])
                vid2 = int(y[2])
                vid3 = int(y[3])
                self.vertex1[fid] = vts[vid1 - 1]
                self.vertex2[fid] = vts[vid2 - 1]
                self.vertex3[fid] = vts[vid3 - 1]
                self.normal[fid] = (vts[vid2-1] - vts[vid1-1]).cross(vts[vid3-1] - vts[vid2-1]).normalized()
                fid += 1
        self.num_polygons = fid

    @ti.kernel
    def build_small_aabb(self):
        for bi in range(self.num_small_aabb):
            bxmin, bymin, bzmin = Inf, Inf, Inf
            bxmax, bymax, bzmax = -Inf, -Inf, -Inf
            fi_start = bi * self.num_polygons_per_box
            fi_end = min(fi_start + self.num_polygons_per_box, self.num_polygons)
            for fi in range(fi_start, fi_end):
                bxmin, bxmax, bymin, bymax, bzmin, bzmax = self.vertex_update_aabb(self.vertex1[fi], bxmin, bxmax, bymin, bymax, bzmin, bzmax)
                bxmin, bxmax, bymin, bymax, bzmin, bzmax = self.vertex_update_aabb(self.vertex2[fi], bxmin, bxmax, bymin, bymax, bzmin, bzmax)
                bxmin, bxmax, bymin, bymax, bzmin, bzmax = self.vertex_update_aabb(self.vertex3[fi], bxmin, bxmax, bymin, bymax, bzmin, bzmax)
            self.small_aabb_faces[bi] = ti.Vector([bxmin, bxmax, bymin, bymax, bzmin, bzmax])

    @ti.func
    def inside_check(self, i, point):
        v1 = self.vertex1[i] - point
        v2 = self.vertex2[i] - point
        v3 = self.vertex3[i] - point
        n1 = v1.cross(v2)
        n2 = v2.cross(v3)
        n3 = v3.cross(v1)
        isInside = False
        if n1.dot(n2) > 0 and n1.dot(n3) > 0:
            isInside = True
        return isInside

    @ti.func
    def hit_triangle(self, i, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal[i]
        p = self.vertex1[i]
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -n
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                if self.inside_check(i, hit_point):
                    is_hit = True
                    if d.dot(n) < 0:
                        front_face = True
                        hit_point_normal = n
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

    @ti.func
    def aabb_intersect(self, ray, aabb_face, tmin=0.001, tmax=10e8):
        intersect_with_box = False
        dir = ray.direction
        origin = ray.origin
        box_tmin = tmin
        box_tmax = tmax
        for i in ti.static(range(3)):
            if dir[i] > Epsilon or dir[i] < -Epsilon:
                t0 = (aabb_face[2*i] - origin[i]) / dir[i]
                t1 = (aabb_face[2*i+1] - origin[i]) / dir[i]
                if t0>t1:
                    t0, t1 = t1, t0
                if t0 > box_tmin:
                    box_tmin = t0
                if t1 < box_tmax:
                    box_tmax = t1
        if box_tmin < box_tmax and box_tmax>tmin and box_tmin < tmax:
            intersect_with_box = True
        return intersect_with_box

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        front_face = False
        closest_t = tmax
        # check if intersect with large aabb
        intersect_with_large_box = self.aabb_intersect(ray, self.large_aabb_faces[None], tmin, closest_t)
        if intersect_with_large_box == True:
            # check if intersect with smaller aabb
            for i_box in range(self.num_small_aabb):
                intersect_with_small_box = self.aabb_intersect(ray, self.small_aabb_faces[i_box], tmin, closest_t)
                if intersect_with_small_box == True:
                    # check if intersect with triangles inside this small aabb
                    # iterate over triangles
                    i_start = i_box * self.num_polygons_per_box
                    i_end = ti.min(i_start+ self.num_polygons_per_box, self.num_polygons)
                    for i in range(i_start, i_end):
                        is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = self.hit_triangle(i, ray, tmin, closest_t)
                        if is_hit_tmp:
                            closest_t = root_tmp
                            is_hit = is_hit_tmp
                            hit_point = hit_point_tmp
                            hit_point_normal = hit_point_normal_tmp
                            front_face = front_face_tmp
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color
