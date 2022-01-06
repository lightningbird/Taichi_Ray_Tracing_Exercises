import taichi as ti

PI = 3.14159265
Inf = 10e8

@ti.func
def rand3():
    return ti.Vector([ti.random(), ti.random(), ti.random()])

@ti.func
def random_in_unit_sphere():
    p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    while p.norm() >= 1.0:
        p = 2.0 * rand3() - ti.Vector([1, 1, 1])
    return p

@ti.func
def random_unit_vector():
    return random_in_unit_sphere().normalized()

@ti.func
def to_light_source(hit_point, light_source):
    return light_source - hit_point

@ti.func
def reflect(v, normal):
    return v - 2 * v.dot(normal) * normal

@ti.func
def refract(uv, n, etai_over_etat):
    cos_theta = min(n.dot(-uv), 1.0)
    r_out_perp = etai_over_etat * (uv + cos_theta * n)
    r_out_parallel = -ti.sqrt(abs(1.0 - r_out_perp.dot(r_out_perp))) * n
    return r_out_perp + r_out_parallel

@ti.func
def reflectance(cosine, ref_idx):
    # Use Schlick's approximation for reflectance.
    r0 = (1 - ref_idx) / (1 + ref_idx)
    r0 = r0 * r0
    return r0 + (1 - r0) * pow((1 - cosine), 5)

@ti.data_oriented
class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
    def at(self, t):
        return self.origin + t * self.direction

@ti.data_oriented
class Sphere:
    def __init__(self, center, radius, material, color, use_texture=False):
        self.center = center
        self.radius = radius
        self.material = material
        self.color = color
        self.use_texture = use_texture
        self.res_y = 128
        self.res_x = 128
        self.texture_image = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_y, self.res_x))
        if self.use_texture:
            self.generate_texture()

    @ti.kernel
    def generate_texture(self):
    # reference ==> https://www.shadertoy.com/view/MdlXz8
        time = 0.1
        TAU = 2*PI
        for i, j in self.texture_image:
            color = ti.Vector([0.0, 0.0, 0.0])
            uv = ti.Vector([float(i)/self.res_y, float(j)/self.res_x])
            p = uv*TAU - TAU * ti.floor(uv * TAU / TAU) - 250.0
            p2 = p
            c = 1.0
            inten = 0.005
            max_iter = 5
            for k in range(max_iter):
                t = time * (1.0 - (3.5 / float(k+1)))
                p2 = p + ti.Vector([ti.cos(t-p2.x) + ti.sin(t+p2.y), ti.sin(t-p2.y)+ti.cos(t+p2.x)])
                c += 1.0/(ti.Vector([p.x / (ti.sin(p2.x+t)/inten), p.y / (ti.cos(p2.y + t) / inten)])).norm()
            c /= float(max_iter)
            c = 1.17 - c**1.4 # reverse
            c = c ** 8 # making it sharp
            color = ti.Vector([1.0, 1.0, 1.0]) * c
            color += ti.min(ti.max(ti.Vector([0.0, 0.35, 0.5]), 0.0), 1.0) # turning blue
            self.texture_image[i,j] = color

    @ti.func
    def compute_texture_coord(self, point):
        vec = point - self.center
        theta = ti.acos(vec.dot(ti.Vector([0.0, -1.0, 0.0]))/vec.norm())
        phi = ti.atan2(-vec[2], vec[0]) + PI
        u = int(phi / (2*PI) * self.res_y)
        v = int(theta / PI * self.res_x)
        u = int(ti.min(ti.max(u, 0), self.res_y))
        v = int(ti.min(ti.max(v, 0), self.res_x))
        return u, v

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        oc = ray.origin - self.center
        a = ray.direction.dot(ray.direction)
        b = 2.0 * oc.dot(ray.direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c
        is_hit = False
        front_face = False
        root = 0.0
        hit_point =  ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = self.color
        if discriminant > 0:
            sqrtd = ti.sqrt(discriminant)
            root = (-b - sqrtd) / (2 * a)
            if root < t_min or root > t_max:
                root = (-b + sqrtd) / (2 * a)
                if root >= t_min and root <= t_max:
                    is_hit = True
            else:
                is_hit = True
        if is_hit:
            hit_point = ray.at(root)
            hit_point_normal = (hit_point - self.center) / self.radius
            # Check which side does the ray hit, we set the hit point normals always point outward from the surface
            if ray.direction.dot(hit_point_normal) < 0:
                front_face = True
            else:
                hit_point_normal = -hit_point_normal

        if is_hit == True and self.use_texture==True:
            u, v = self.compute_texture_coord(hit_point)
            color = self.texture_image[u, v]

        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, color

@ti.data_oriented
class Triangle:
    def __init__(self, vertex1, vertex2, vertex3, material, color):
        # three vertices counterclockwise
        self.vertex1 = vertex1
        self.vertex2 = vertex2
        self.vertex3 = vertex3
        self.material = material
        self.color = color
        self.normal = (vertex2 - vertex1).cross(vertex3 - vertex2).normalized()

    @ti.func
    def inside_check(self, point):
        v1 = self.vertex1 - point
        v2 = self.vertex2 - point
        v3 = self.vertex3 - point
        n1 = v1.cross(v2)
        n2 = v2.cross(v3)
        n3 = v3.cross(v1)
        isInside = False
        if n1.dot(n2) > 0 and n1.dot(n3) > 0:
            isInside = True
        return isInside

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal
        p = self.vertex1
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -self.normal
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                if self.inside_check(hit_point):
                    is_hit = True
                    if d.dot(n) < 0:
                        front_face = True
                        hit_point_normal = self.normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Polygon:
    def __init__(self, vertices, material, color):
        self.n = len(vertices)
        assert self.n >=3
        self.vertices = vertices
        v1 = vertices[0]
        v2 = vertices[1]
        v3 = vertices[2]
        self.normal = (v2 - v1).cross(v3 - v2).normalized()
        self.material = material
        self.color = color

    @ti.func
    def inside_check(self, point):
        vecs = []
        norms = []
        for i in ti.static(range(self.n)):
            vec = self.vertices[i] - point
            vecs.append(vec)
        for i in ti.static(range(self.n-1)):
            norm = vecs[i].cross(vecs[i+1])
            norms.append(norm)
        norms.append(vecs[self.n-1].cross(vecs[0]))

        isInside = True
        for i in ti.static(range(self.n-1)):
            if norms[0].dot(norms[i+1])<0:
                isInside = False
        return isInside

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal
        p = self.vertices[0]
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -self.normal
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                if self.inside_check(hit_point):
                    is_hit = True
                    if d.dot(n) < 0:
                        front_face = True
                        hit_point_normal = self.normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Plane:
    def __init__(self, point, normal, material, color):
        self.point = point
        self.normal = normal.normalized()
        self.material = material
        self.color = color

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal
        p = self.point
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -self.normal
        front_face = False
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                is_hit = True
                if d.dot(n) < 0:
                    front_face = True
                    hit_point_normal = self.normal
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, self.color

@ti.data_oriented
class Plane_Textured:
    def __init__(self, point, normal, material, color, texture_vertices):
        self.point = point
        self.normal = normal.normalized()
        self.material = material
        self.color = color

        self.vertices = texture_vertices
        self.res_y = 128
        self.res_x = 128
        self.texture_image = ti.Vector.field(3, dtype=ti.f32, shape=(self.res_y, self.res_x))
        self.generate_texture()

    @ti.kernel
    def generate_texture(self):
    # reference ==> https://www.shadertoy.com/view/MdlXz8
        time = 0.1
        TAU = 2*PI
        for i, j in self.texture_image:
            color = ti.Vector([0.0, 0.0, 0.0])
            uv = ti.Vector([float(i)/self.res_y, float(j)/self.res_x])
            p = uv*TAU - TAU * ti.floor(uv * TAU / TAU) - 250.0
            p2 = p
            c = 1.0
            inten = 0.005
            max_iter = 5
            for k in range(max_iter):
                t = time * (1.0 - (3.5 / float(k+1)))
                p2 = p + ti.Vector([ti.cos(t-p2.x) + ti.sin(t+p2.y), ti.sin(t-p2.y)+ti.cos(t+p2.x)])
                c += 1.0/(ti.Vector([p.x / (ti.sin(p2.x+t)/inten), p.y / (ti.cos(p2.y + t) / inten)])).norm()
            c /= float(max_iter)
            c = 1.17 - c**1.4 # reverse
            c = c ** 8 # making it sharp
            color = ti.Vector([1.0, 1.0, 1.0]) * c
            color += ti.min(ti.max(ti.Vector([0.0, 0.35, 0.5]), 0.0), 1.0) # turning blue
            self.texture_image[i,j] = color

    @ti.func
    def compute_texture_coord(self, point):
        x_vec = self.vertices[3] - self.vertices[0]
        y_vec = self.vertices[1] - self.vertices[0]
        xp = (point - self.vertices[0]).dot(x_vec.normalized()) / x_vec.norm()
        yp = (point - self.vertices[0]).dot(y_vec.normalized()) / y_vec.norm()
        u = int(yp * self.res_y)
        v = int(xp * self.res_x)
        return u, v

    @ti.func
    def hit(self, ray, tmin=0.001, tmax=10e8):
        d = ray.direction
        o = ray.origin
        n = self.normal
        p = self.point
        is_hit = False
        root = 0.0
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = -self.normal
        front_face = False
        color = self.color
        if ti.abs(d.dot(n)) > 0:
            root = n.dot(p - o) / d.dot(n)
            if root >= tmin and root <= tmax:
                hit_point = ray.at(root)
                is_hit = True
                if d.dot(n) < 0:
                    front_face = True
                    hit_point_normal = self.normal
        if is_hit == True:
            u, v = self.compute_texture_coord(hit_point)
            if u>=0 and u<self.res_y and v>=0 and v<self.res_x:
                color = self.texture_image[u, v]
        return is_hit, root, hit_point, hit_point_normal, front_face, self.material, color

@ti.data_oriented
class Hittable_list:
    def __init__(self):
        self.objects = []
    def add(self, obj):
        self.objects.append(obj)
    def clear(self):
        self.objects = []

    @ti.func
    def hit(self, ray, t_min=0.001, t_max=10e8):
        closest_t = t_max
        is_hit = False
        front_face = False
        hit_point = ti.Vector([0.0, 0.0, 0.0])
        hit_point_normal = ti.Vector([0.0, 0.0, 0.0])
        color = ti.Vector([0.0, 0.0, 0.0])
        material = 1
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, closest_t)
            if is_hit_tmp:
                closest_t = root_tmp
                is_hit = is_hit_tmp
                hit_point = hit_point_tmp
                hit_point_normal = hit_point_normal_tmp
                front_face = front_face_tmp
                material = material_tmp
                color = color_tmp
        return is_hit, hit_point, hit_point_normal, front_face, material, color

    @ti.func
    def hit_shadow(self, ray, t_min=0.001, t_max=10e8):
        is_hit_source = False
        is_hit_source_temp = False
        hitted_dielectric_num = 0
        is_hitted_non_dielectric = False
        # Compute the t_max to light source
        is_hit_tmp, root_light_source, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp = \
        self.objects[0].hit(ray, t_min)
        for index in ti.static(range(len(self.objects))):
            is_hit_tmp, root_tmp, hit_point_tmp, hit_point_normal_tmp, front_face_tmp, material_tmp, color_tmp =  self.objects[index].hit(ray, t_min, root_light_source)
            if is_hit_tmp:
                if material_tmp != 3 and material_tmp != 0:
                    is_hitted_non_dielectric = True
                if material_tmp == 3:
                    hitted_dielectric_num += 1
                if material_tmp == 0:
                    is_hit_source_temp = True
        if is_hit_source_temp and (not is_hitted_non_dielectric) and hitted_dielectric_num == 0:
            is_hit_source = True
        return is_hit_source, hitted_dielectric_num, is_hitted_non_dielectric


@ti.data_oriented
class Camera:
    def __init__(self, fov=60, aspect_ratio=1.0):
        # Camera parameters
        self.lookfrom = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.lookat = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.vup = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.fov = fov
        self.aspect_ratio = aspect_ratio

        self.cam_lower_left_corner = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_horizontal = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_vertical = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.cam_origin = ti.Vector.field(3, dtype=ti.f32, shape=())
        self.reset()

    @ti.kernel
    def reset(self):
        self.lookfrom[None] = [0.0, 1.0, -5.0]
        self.lookat[None] = [0.0, 1.0, -1.0]
        self.vup[None] = [0.0, 1.0, 0.0]
        theta = self.fov * (PI / 180.0)
        half_height = ti.tan(theta / 2.0)
        half_width = self.aspect_ratio * half_height
        self.cam_origin[None] = self.lookfrom[None]
        w = (self.lookfrom[None] - self.lookat[None]).normalized()
        u = (self.vup[None].cross(w)).normalized()
        v = w.cross(u)
        self.cam_lower_left_corner[None] = ti.Vector([-half_width, -half_height, -1.0])
        self.cam_lower_left_corner[
            None] = self.cam_origin[None] - half_width * u - half_height * v - w
        self.cam_horizontal[None] = 2 * half_width * u
        self.cam_vertical[None] = 2 * half_height * v

    @ti.func
    def get_ray(self, u, v):
        return Ray(self.cam_origin[None], self.cam_lower_left_corner[None] + u * self.cam_horizontal[None] + v * self.cam_vertical[None] - self.cam_origin[None])