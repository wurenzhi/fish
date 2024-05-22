import numpy as np
from scipy.spatial import cKDTree
from pyproj import Transformer

def setup_transformer(utm_zone=15, north=True):
    """
    Setup a transformer to convert lat/lon to UTM.
    """
    hemi = 'north' if north else 'south'
    crs_from = f"epsg:4326"
    crs_to = f"+proj=utm +zone={utm_zone} +{hemi} +ellps=WGS84"
    transformer = Transformer.from_crs(crs_from, crs_to, always_xy=True)
    return transformer

class Interpolator:
    """
    Performs IDW interpolation using k-nearest neighbors for multiple query points.
    """
    def __init__(self, lon_grid, lat_grid, depth_grid):
        self.transformer = setup_transformer()
        lon_grid_m, lat_grid_m = self.transformer.transform(lon_grid.ravel(), lat_grid.ravel())
        points = np.column_stack((lon_grid_m, lat_grid_m, depth_grid.ravel()))
        self.tree = cKDTree(points)
    
    def interpolate(self, grid_vals_list,query_lons, query_lats, query_depths, k=5, power=2):
        lon_m, lat_m = self.transformer.transform(query_lons, query_lats)
        distances, indices = self.tree.query(np.column_stack((lon_m, lat_m, query_depths)), k=k, workers=-1)
        weights = 1 / (np.power(distances, power) + np.finfo(float).eps)
        
        interpolated_vals = []
        for grid_vals in grid_vals_list:
            grid_vals = grid_vals.ravel()
            weighted_vals = weights * grid_vals[indices]
            interpolated_val = np.sum(weighted_vals, axis=1) / np.sum(weights, axis=1)
            interpolated_vals.append(interpolated_val)
        return interpolated_vals

if __name__ == "__main__":
    # Example grids and data
    lon_grid, lat_grid = np.meshgrid(np.linspace(-95, -85, 100), np.linspace(25, 30, 100))
    depth_grid = np.random.random((100, 100)) * 5000
    temp_grid = np.random.random((100, 100)) * 30

    # Example points to interpolate (1000 points)
    query_points = np.column_stack((
        np.random.uniform(-95, -85, 1000),
        np.random.uniform(25, 30, 1000),
        np.random.uniform(0, 5000, 1000)
    ))
    inter = Interpolator(lon_grid, lat_grid, depth_grid)

    vals = inter.interpolate([temp_grid],query_points[:, 0], query_points[:, 1], query_points[:, 2])
