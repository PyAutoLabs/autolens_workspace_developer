import autoarray as aa

shape_native = (10, 10)

grid = aa.Grid2D.uniform(shape_native=shape_native, pixel_scales=0.1)

mesh = aa.Mesh2DRectangularUniform.overlay_grid(
    shape_native=(5, 5), grid=grid, buffer=1e-8
)

mapper_grids = aa.MapperGrids(
    mask=None,
    source_plane_data_grid=grid,
    source_plane_mesh_grid=mesh,
)

mapper = aa.Mapper(mapper_grids=mapper_grids, regularization=None)

print(mapper_grids.source_plane_mesh_grid)
print(mapper.edges_transformed)
