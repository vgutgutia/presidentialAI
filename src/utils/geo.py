"""
Geospatial utilities for marine debris detection.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path

import rasterio
from rasterio.transform import Affine, from_bounds
from rasterio.crs import CRS
from rasterio.warp import reproject, Resampling, calculate_default_transform
import geopandas as gpd
from shapely.geometry import box, Polygon, Point


def create_geotiff(
    data: np.ndarray,
    output_path: str,
    transform: Affine,
    crs: CRS,
    nodata: Optional[float] = None,
    compress: str = "lzw",
) -> str:
    """
    Create a GeoTIFF file from numpy array.
    
    Args:
        data: Image data (C, H, W) or (H, W)
        output_path: Path to save GeoTIFF
        transform: Affine transform
        crs: Coordinate reference system
        nodata: NoData value
        compress: Compression method
        
    Returns:
        Path to created file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Handle single band
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    
    count, height, width = data.shape
    
    profile = {
        "driver": "GTiff",
        "dtype": data.dtype,
        "width": width,
        "height": height,
        "count": count,
        "crs": crs,
        "transform": transform,
        "compress": compress,
    }
    
    if nodata is not None:
        profile["nodata"] = nodata
    
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(data)
    
    return str(output_path)


def reproject_raster(
    input_path: str,
    output_path: str,
    dst_crs: str = "EPSG:4326",
    resolution: Optional[float] = None,
) -> str:
    """
    Reproject a raster to a new CRS.
    
    Args:
        input_path: Input raster path
        output_path: Output raster path
        dst_crs: Destination CRS
        resolution: Output resolution (optional)
        
    Returns:
        Path to reprojected raster
    """
    with rasterio.open(input_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds,
            resolution=resolution
        )
        
        profile = src.profile.copy()
        profile.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
        })
        
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.bilinear,
                )
    
    return output_path


def get_raster_bounds(raster_path: str, as_gdf: bool = False):
    """
    Get bounds of a raster file.
    
    Args:
        raster_path: Path to raster
        as_gdf: If True, return as GeoDataFrame
        
    Returns:
        Bounds tuple or GeoDataFrame
    """
    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
    
    if as_gdf:
        geom = box(*bounds)
        return gpd.GeoDataFrame({"geometry": [geom]}, crs=crs)
    
    return bounds


def pixel_to_geo(
    row: int,
    col: int,
    transform: Affine,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates.
    
    Args:
        row: Row index
        col: Column index
        transform: Affine transform
        
    Returns:
        Tuple of (x, y) geographic coordinates
    """
    x, y = transform * (col + 0.5, row + 0.5)  # Center of pixel
    return x, y


def geo_to_pixel(
    x: float,
    y: float,
    transform: Affine,
) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates.
    
    Args:
        x: X coordinate (longitude)
        y: Y coordinate (latitude)
        transform: Affine transform
        
    Returns:
        Tuple of (row, col) pixel coordinates
    """
    inv_transform = ~transform
    col, row = inv_transform * (x, y)
    return int(row), int(col)


def calculate_area_m2(
    polygon: Polygon,
    crs: CRS,
) -> float:
    """
    Calculate area of polygon in square meters.
    
    Args:
        polygon: Shapely polygon
        crs: Coordinate reference system
        
    Returns:
        Area in square meters
    """
    # If geographic CRS, reproject to appropriate UTM
    if crs.is_geographic:
        # Get centroid for UTM zone calculation
        centroid = polygon.centroid
        utm_zone = int((centroid.x + 180) / 6) + 1
        hemisphere = "north" if centroid.y >= 0 else "south"
        
        if hemisphere == "north":
            utm_crs = CRS.from_epsg(32600 + utm_zone)
        else:
            utm_crs = CRS.from_epsg(32700 + utm_zone)
        
        # Reproject polygon
        gdf = gpd.GeoDataFrame({"geometry": [polygon]}, crs=crs)
        gdf_utm = gdf.to_crs(utm_crs)
        
        return gdf_utm.geometry[0].area
    else:
        # Already in projected CRS
        return polygon.area


def create_hotspot_report(
    hotspots_gdf: gpd.GeoDataFrame,
    output_path: str,
    include_visualization: bool = True,
) -> str:
    """
    Create a summary report of detected hotspots.
    
    Args:
        hotspots_gdf: GeoDataFrame of hotspots
        output_path: Path to save report
        include_visualization: Whether to include map
        
    Returns:
        Path to report file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create summary statistics
    summary = {
        "total_hotspots": len(hotspots_gdf),
        "total_area_km2": hotspots_gdf["area_m2"].sum() / 1e6,
        "mean_confidence": hotspots_gdf["confidence"].mean(),
        "max_confidence": hotspots_gdf["confidence"].max(),
        "min_confidence": hotspots_gdf["confidence"].min(),
    }
    
    # Create report
    report = [
        "=" * 60,
        "MARINE DEBRIS DETECTION REPORT",
        "=" * 60,
        "",
        f"Total hotspots detected: {summary['total_hotspots']}",
        f"Total estimated debris area: {summary['total_area_km2']:.2f} km²",
        f"Confidence range: {summary['min_confidence']:.2f} - {summary['max_confidence']:.2f}",
        f"Mean confidence: {summary['mean_confidence']:.2f}",
        "",
        "-" * 60,
        "TOP 10 HOTSPOTS BY CONFIDENCE",
        "-" * 60,
    ]
    
    for _, row in hotspots_gdf.head(10).iterrows():
        report.append(
            f"Rank {row['rank']}: "
            f"({row['centroid_lat']:.4f}, {row['centroid_lon']:.4f}) "
            f"Area: {row['area_m2']/1000:.1f} km² "
            f"Conf: {row['confidence']:.2f}"
        )
    
    report.extend(["", "=" * 60])
    
    # Save report
    with open(output_path, "w") as f:
        f.write("\n".join(report))
    
    return str(output_path)


def merge_hotspots(
    hotspots_list: List[gpd.GeoDataFrame],
    buffer_distance_m: float = 100,
) -> gpd.GeoDataFrame:
    """
    Merge overlapping hotspots from multiple scenes.
    
    Args:
        hotspots_list: List of hotspot GeoDataFrames
        buffer_distance_m: Buffer distance for merging nearby polygons
        
    Returns:
        Merged GeoDataFrame
    """
    if not hotspots_list:
        return gpd.GeoDataFrame()
    
    # Concatenate all hotspots
    all_hotspots = gpd.GeoDataFrame(
        pd.concat(hotspots_list, ignore_index=True),
        crs=hotspots_list[0].crs,
    )
    
    # Buffer and dissolve overlapping
    buffered = all_hotspots.geometry.buffer(buffer_distance_m)
    dissolved = buffered.unary_union
    
    # Convert back to polygons
    if dissolved.geom_type == "MultiPolygon":
        geometries = list(dissolved.geoms)
    else:
        geometries = [dissolved]
    
    # Calculate statistics for merged polygons
    merged_data = []
    for geom in geometries:
        # Find original hotspots that overlap
        mask = all_hotspots.geometry.intersects(geom)
        overlapping = all_hotspots[mask]
        
        merged_data.append({
            "geometry": geom.buffer(-buffer_distance_m),  # Remove buffer
            "area_m2": geom.buffer(-buffer_distance_m).area,
            "confidence": overlapping["confidence"].mean(),
            "n_detections": len(overlapping),
            "centroid_lat": geom.centroid.y,
            "centroid_lon": geom.centroid.x,
        })
    
    merged_gdf = gpd.GeoDataFrame(merged_data, crs=all_hotspots.crs)
    merged_gdf = merged_gdf.sort_values("confidence", ascending=False).reset_index(drop=True)
    merged_gdf["rank"] = merged_gdf.index + 1
    
    return merged_gdf
