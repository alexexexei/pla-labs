from manim import np, VGroup, Polygon, Line, BLUE, WHITE

class Cube(VGroup):
    def __init__(self, vertices, **kwargs):
        super().__init__(**kwargs)

        self.vertices = vertices
        vertices = np.transpose(np.delete(vertices, -1, axis=0))

        faces = [
            [vertices[0], vertices[1], vertices[2], vertices[3]],
            [vertices[4], vertices[5], vertices[6], vertices[7]],
            [vertices[0], vertices[1], vertices[5], vertices[4]],
            [vertices[2], vertices[3], vertices[7], vertices[6]],
            [vertices[0], vertices[3], vertices[7], vertices[4]],
            [vertices[1], vertices[2], vertices[6], vertices[5]]
        ]
        
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]

        faces = VGroup(*[Polygon(*face, fill_opacity=0.3, color=BLUE) for face in faces])
        edges = VGroup(*[Line(vertices[start], vertices[end], color=WHITE) for start, end in edges])

        self.add(faces, edges)

    def get_transformed(self, transform_matrix):
        new_vertices = np.dot(transform_matrix, self.vertices)
        return Cube(new_vertices)