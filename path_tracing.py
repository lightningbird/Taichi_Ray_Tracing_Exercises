import taichi as ti
import numpy as np
import argparse
from ray_tracing_models import Ray, Camera, Hittable_list, Sphere, Triangle, Polygon, Plane, Plane_Textured, Torus, Mesh, \
PI, Inf, random_in_unit_sphere, refract, reflect, reflectance, random_unit_vector

ti.init(kernel_profiler = True, arch=ti.gpu)

# Canvas
aspect_ratio = 1.0
image_width = 400
image_height = int(image_width / aspect_ratio)
canvas = ti.Vector.field(3, dtype=ti.f32, shape=(image_width, image_height))

# Rendering parameters
samples_per_pixel = 4
max_depth = 10
sample_on_unit_sphere_surface = True


@ti.kernel
def render():
    for i, j in canvas:
        u = (i + ti.random()) / image_width
        v = (j + ti.random()) / image_height
        color = ti.Vector([0.0, 0.0, 0.0])
        for n in range(samples_per_pixel):
            ray = camera.get_ray(u, v)
            color += ray_color(ray)
        color /= samples_per_pixel
        canvas[i, j] += color

# Path tracing
@ti.func
def ray_color(ray):
    color_buffer = ti.Vector([0.0, 0.0, 0.0])
    brightness = ti.Vector([1.0, 1.0, 1.0])
    scattered_origin = ray.origin
    scattered_direction = ray.direction
    p_RR = 0.8
    for n in range(max_depth):
        if ti.random() > p_RR:
            break
        is_hit, hit_point, hit_point_normal, front_face, material, color = scene.hit(Ray(scattered_origin, scattered_direction))
        if is_hit:
            if material == 0:
                color_buffer = color * brightness
                break
            else:
                # Diffuse
                if material == 1:
                    target = hit_point + hit_point_normal
                    if sample_on_unit_sphere_surface:
                        target += random_unit_vector()
                    else:
                        target += random_in_unit_sphere()
                    scattered_direction = target - hit_point
                    scattered_origin = hit_point
                    brightness *= color
                # Metal and Fuzz Metal
                elif material == 2 or material == 4:
                    fuzz = 0.0
                    if material == 4:
                        fuzz = 0.4
                    scattered_direction = reflect(scattered_direction.normalized(),
                                                  hit_point_normal)
                    if sample_on_unit_sphere_surface:
                        scattered_direction += fuzz * random_unit_vector()
                    else:
                        scattered_direction += fuzz * random_in_unit_sphere()
                    scattered_origin = hit_point
                    if scattered_direction.dot(hit_point_normal) < 0:
                        break
                    else:
                        brightness *= color
                # Dielectric
                elif material == 3:
                    refraction_ratio = 1.5
                    if front_face:
                        refraction_ratio = 1 / refraction_ratio
                    cos_theta = min(-scattered_direction.normalized().dot(hit_point_normal), 1.0)
                    sin_theta = ti.sqrt(1 - cos_theta * cos_theta)
                    # total internal reflection
                    if refraction_ratio * sin_theta > 1.0 or reflectance(cos_theta, refraction_ratio) > ti.random():
                        scattered_direction = reflect(scattered_direction.normalized(), hit_point_normal)
                    else:
                        scattered_direction = refract(scattered_direction.normalized(), hit_point_normal, refraction_ratio)
                    scattered_origin = hit_point
                    brightness *= color
                brightness /= p_RR
    return color_buffer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Naive Ray Tracing')
    parser.add_argument('--test_number', type=int, default=1, help='test scene number (default: 1)')
    parser.add_argument(
        '--max_depth', type=int, default=10, help='max depth (default: 10)')
    parser.add_argument(
        '--samples_per_pixel', type=int, default=4, help='samples_per_pixel  (default: 4)')
    parser.add_argument(
        '--samples_in_unit_sphere', action='store_true', help='whether sample in a unit sphere')
    args = parser.parse_args()

    test_number = args.test_number
    assert test_number<=2
    max_depth = args.max_depth
    samples_per_pixel = args.samples_per_pixel
    sample_on_unit_sphere_surface = not args.samples_in_unit_sphere
    scene = Hittable_list()

    # Light source
    scene.add(Sphere(center=ti.Vector([0, 5.4, -1]), radius=3.0, material=0, color=ti.Vector([10.0, 10.0, 10.0])))
    # Ground
    scene.add(Plane(point=ti.Vector([0, -0.5, 0]), normal=ti.Vector([0, 1.0, 0]), material=2, color=ti.Vector([0.8, 0.8, 0.8])))
    # ceiling
    scene.add(Plane(point=ti.Vector([0, 2.5, 0]), normal=ti.Vector([0, -1.0, 0]), material=1, color=ti.Vector([0.8, 0.8, 0.8])))
    # back wall
    scene.add(Plane_Textured(point=ti.Vector([0, 0, 1.0]), normal=ti.Vector([0, 0, -1.0]), material=1, color=ti.Vector([0.8, 0.8, 0.8]),
                            texture_vertices=[ti.Vector([1.5, -0.5,-1]), ti.Vector([1.5, 2.5,-1]), ti.Vector([-1.5, 2.5, -1]), ti.Vector([-1.5, -0.5,-1])]))
    # right wall
    scene.add(Plane(point=ti.Vector([-1.5, 0, 0]), normal=ti.Vector([1.0, 0, 0]), material=1, color=ti.Vector([0.6, 0.0, 0.0])))
    # left wall
    scene.add(Plane(point=ti.Vector([1.5, 0, 0]), normal=ti.Vector([-1.0, 0, 0]), material=1, color=ti.Vector([0.0, 0.6, 0.0])))

    if test_number == 1:
        ## Glass ball
        scene.add(Sphere(center=ti.Vector([0, -0.1, -1.5]), radius=0.3, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
        ## Diffuse Torus
        scene.add(Torus(center=ti.Vector([0, -0.4, -1.5]), inside_point = ti.Vector([0.3, -0.4, -1.5]), up_normal = ti.Vector([0, 1.0, 0]), inside_radius=0.1, 
                       nU = 10, nV = 10, material=1, color=ti.Vector([0.8, 0.3, 0.3]), write_to_obj_file = True, obj_filename = 'torus1.obj'))
        ## Diffuse Rectangular Pyramid
        top_vertex = ti.Vector([-0.8, 0.9, -1.0])
        sq_vertex1 = ti.Vector([-0.3, -0.5, -0.5])
        sq_vertex2 = ti.Vector([-0.3, -0.5, -1.5])
        sq_vertex3 = ti.Vector([-1.3, -0.5, -1.5])
        sq_vertex4 = ti.Vector([-1.3, -0.5, -0.5])
        scene.add(Triangle(vertex1=top_vertex, vertex2=sq_vertex1, vertex3=sq_vertex2, material=1, color=ti.Vector([0.0, 0.8, 0.8])))
        scene.add(Triangle(vertex1=top_vertex, vertex2=sq_vertex2, vertex3=sq_vertex3, material=1, color=ti.Vector([0.0, 0.8, 0.8])))
        scene.add(Triangle(vertex1=top_vertex, vertex2=sq_vertex3, vertex3=sq_vertex4, material=1, color=ti.Vector([0.0, 0.8, 0.8])))
        scene.add(Triangle(vertex1=top_vertex, vertex2=sq_vertex4, vertex3=sq_vertex1, material=1, color=ti.Vector([0.0, 0.8, 0.8])))
        ## Glass Pentagon Torus
        scene.add(Torus(center=ti.Vector([0.7, 0.1, -0.5]), inside_point = ti.Vector([0.7, 0.6, -0.5]), up_normal = ti.Vector([0, 0, -1.0]), inside_radius=0.1, 
                       nU = 5, nV = 8, material=3, color=ti.Vector([1.0, 1.0, 1.0])))
        ## Glass Triangle Torus
        scene.add(Torus(center=ti.Vector([0.6, -0.3, -2.0]), inside_point = ti.Vector([0.9, -0.3, -2.0]), up_normal = ti.Vector([0, 1.0, 0]), inside_radius=0.1, 
                       nU = 3, nV = 8, material=3, color=ti.Vector([0.8, 0.6, 0.2])))

    camera = Camera()
    gui = ti.GUI("Ray Tracing", res=(image_width, image_height))
    canvas.fill(0)
    cnt = 0
    while gui.running and cnt< 500:
        render()
        cnt += 1
        gui.set_image(np.sqrt(canvas.to_numpy() / cnt))
        gui.show()
    ti.print_kernel_profile_info('count')
    ti.imwrite(np.sqrt(canvas.to_numpy() / cnt), f"test_scene_1.png")
