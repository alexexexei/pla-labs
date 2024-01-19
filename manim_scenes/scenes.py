from manim import *
from transform_matrices import *
from scene_setup import *
from object3d import Cube

class CubeCreationScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=60, theta=45)

        cube = Cube(default_vertices)
        self.play(FadeIn(cube), run_time=2)

        self.wait(2)

        self.play(FadeOut(cube), run_time=2)

class CubeScalingScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=60, theta=45)

        original_cube = Cube(default_vertices)

        self.play(FadeIn(original_cube), run_time=2)
        self.wait(1)

        scaled_cube_A = original_cube.copy().get_transformed(scale_matrix_A)

        self.play(ReplacementTransform(original_cube, scaled_cube_A), run_time=2)
        self.wait(1)

        scaled_cube_B = scaled_cube_A.copy().get_transformed(scale_matrix_B)
        
        self.play(ReplacementTransform(scaled_cube_A, scaled_cube_B), run_time=2)
        self.wait(1)

        scaled_not_cube = scaled_cube_B.copy().get_transformed(scale_matrix_C)

        self.play(ReplacementTransform(scaled_cube_B, scaled_not_cube), run_time=2)
        self.wait(1)

        self.play(FadeOut(scaled_not_cube), run_time=2)

class CubeMovingScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=60, theta=45)

        original_cube = Cube(default_vertices*0.5)

        self.play(FadeIn(original_cube), run_time=2)
        self.wait(1)

        moved_cube_A = original_cube.copy().get_transformed(translation_matrix_A)

        self.play(ReplacementTransform(original_cube, moved_cube_A), run_time=2)
        self.wait(1)

        moved_cube_B = moved_cube_A.copy().get_transformed(translation_matrix_B)

        self.play(ReplacementTransform(moved_cube_A, moved_cube_B), run_time=2)
        self.wait(1)

        moved_cube_C = moved_cube_B.copy().get_transformed(translation_matrix_C)

        self.play(ReplacementTransform(moved_cube_B, moved_cube_C), run_time=2)
        self.wait(1)

        moved_cube_D = moved_cube_C.copy().get_transformed(translation_matrix_D)

        self.play(ReplacementTransform(moved_cube_C, moved_cube_D), run_time=2)
        self.wait(1)

        moved_cube_E = moved_cube_D.copy().get_transformed(translation_matrix_E)

        self.play(ReplacementTransform(moved_cube_D, moved_cube_E), run_time=2)
        self.wait(1)

        self.play(FadeOut(moved_cube_E), run_time=2)

class CubeRotatingScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=60, theta=45)

        original_cube = Cube(default_vertices*0.5)

        self.play(FadeIn(original_cube), run_time=2)
        self.wait(1)

        rotated_cube_X = original_cube.copy().get_transformed(rotate_matrix_X_axis)

        self.play(ReplacementTransform(original_cube, rotated_cube_X), run_time=2)
        self.wait(1)

        rotated_cube_Y = rotated_cube_X.copy().get_transformed(rotate_matrix_Y_axis)

        self.play(ReplacementTransform(rotated_cube_X, rotated_cube_Y), run_time=2)
        self.wait(1)

        rotated_cube_Z = rotated_cube_Y.copy().get_transformed(rotate_matrix_Z_axis)

        self.play(ReplacementTransform(rotated_cube_Y, rotated_cube_Z), run_time=2)
        self.wait(1)

        self.play(FadeOut(rotated_cube_Z), run_time=2)

class CubeVerticeRotatingScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=60, theta=45)

        original_cube = Cube(default_vertices)

        self.play(FadeIn(original_cube), run_time=2)
        self.wait(1)

        temp_1 = np.dot(inversed_translation_matrix_v_1_1_1, rotate_matrix_Z_axis)
        temp_2 = np.dot(temp_1, rotate_matrix_Y_axis)
        temp_3 = np.dot(temp_2, rotate_matrix_X_axis)
        transformation_matrix = np.dot(temp_3, translation_matrix_v_1_1_1)

        vertice_rotated_cube = original_cube.copy().get_transformed(transformation_matrix)

        self.play(ReplacementTransform(original_cube, vertice_rotated_cube), run_time=2)
        self.wait(1)

        self.play(FadeOut(vertice_rotated_cube), run_time=2)

class CameraImplementationScene(ThreeDScene):
    def construct(self):
        add_axes_to_scene(self, phi=90, theta=0)

        cube_1 = Cube(default_vertices*0.5)
        cube_1 = cube_1.copy().get_transformed(translation_matrix_A)

        cube_2 = Cube(default_vertices*0.7)
        cube_2 = cube_2.copy().get_transformed(translation_matrix_B)

        cube_3 = Cube(default_vertices*0.3)
        cube_3 = cube_3.copy().get_transformed(translation_matrix_C)

        self.play(FadeIn(cube_1), FadeIn(cube_2), FadeIn(cube_3), run_time=2)
        self.wait(1)

        self.begin_ambient_camera_rotation(rate=90 * DEGREES, about="theta")
        self.wait(3)

        self.play(FadeOut(cube_1), FadeOut(cube_2), FadeOut(cube_3), run_time=2)
