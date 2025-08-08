import numpy as np

class CubeSplitter3D:
    """Split volumes into non-overlapping cubes and project them."""

    def __init__(self, cube_size: int = 8, embed_dim: int = 32, face_embed: bool = False) -> None:
        self.cube_size = cube_size
        self.embed_dim = embed_dim
        self.face_embed = face_embed
        self.proj = np.random.randn(cube_size ** 3, embed_dim).astype(np.float32)

    def forward(self, x: np.ndarray) -> np.ndarray:
        b, c, s, h, w = x.shape
        cubes = []
        for bi in range(b):
            for ci in range(c):
                for zz in range(0, s, self.cube_size):
                    for yy in range(0, h, self.cube_size):
                        for xx in range(0, w, self.cube_size):
                            cube = x[
                                bi,
                                ci,
                                zz:zz + self.cube_size,
                                yy:yy + self.cube_size,
                                xx:xx + self.cube_size,
                            ]
                            if cube.shape == (self.cube_size,)*3:
                                cubes.append(cube.reshape(-1))
        if not cubes:
            return np.zeros((b, 0, self.embed_dim), dtype=x.dtype)
        cubes = np.stack(cubes, axis=0) @ self.proj
        n_per_sample = cubes.shape[0] // b
        return cubes.reshape(b, n_per_sample, -1)
