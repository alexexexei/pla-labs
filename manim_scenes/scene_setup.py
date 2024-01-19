from manim import ThreeDAxes, GRAY, DEGREES

def add_axes_to_scene(scene, phi, theta):
    axes = ThreeDAxes().add_coordinates()
    axes.set_color(GRAY).add(axes.get_axis_labels())
    scene.set_camera_orientation(phi=phi * DEGREES, theta=theta * DEGREES)
    scene.add(axes)