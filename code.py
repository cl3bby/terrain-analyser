# STEP 1: Install geospatial library
!pip install rasterio -q

# STEP 2: Run the automated high-speed prominence analyzer
import math
import requests
import rasterio
from rasterio.io import MemoryFile
import numpy as np
from scipy.ndimage import minimum_filter

def fetch_ga_dem_wcs(min_lon, min_lat, max_lon, max_lat):
    """Fetches raw DEM grids flawlessly via Geoscience Australia WCS."""
    wcs_url = "https://services.ga.gov.au/gis/services/DEM_SRTM_1Second_2024/MapServer/WCSServer"

    # FIXED: Use "1" as the coverage identifier (not the long layer name)
    params = {
        "service": "WCS",
        "version": "1.0.0",
        "request": "GetCoverage",
        "coverage": "1",  # ✅ This is the correct coverage ID
        "crs": "EPSG:4326",
        "bbox": f"{min_lon},{min_lat},{max_lon},{max_lat}",
        "format": "GeoTIFF",
        "resx": "0.000277777777778",
        "resy": "0.000277777777778",
        "interpolation": "NEAREST"
    }

    print("📡 Querying Geoscience Australia Cloud Database...")
    response = requests.get(wcs_url, params=params, timeout=60)

    # Safety Check: Capture server XML errors cleanly instead of throwing a format crash
    if response.status_code != 200 or b"Exception" in response.content[:500] or b"xml" in response.content[:100]:
        print("\n❌ Server rejected parameters. Raw response from Geoscience Australia:")
        print(response.content.decode('utf-8', errors='ignore')[:500])
        raise Exception("WCS Request failed due to server-side constraints.")

    return response.content

def find_max_prominence_fast(wcs_content, max_distance_meters=3000):
    """
    Instantly locates the peak with the maximum vertical prominence drop
    within a strict horizontal distance radius using vectorized math.
    """
    with MemoryFile(wcs_content) as memfile:
        with memfile.open() as dataset:
            elevation_grid = dataset.read(1).astype(np.float32)
            elevation_grid = np.where(elevation_grid < -50, np.nan, elevation_grid)
            transform = dataset.transform

            # Compute exact size of a single data cell (~30m)
            cell_size_x = abs(transform[0] * 111000)

            # Map distance constraint into a matrix cell radius footprint
            pixel_radius = int(max_distance_meters / cell_size_x)
            if pixel_radius < 1: pixel_radius = 1

            print(f"⚡ Vector-scanning terrain layout within a {max_distance_meters}m radial window...")

            # Create a localized spatial moving circular footprint window
            y, x = np.ogrid[-pixel_radius:pixel_radius+1, -pixel_radius:pixel_radius+1]
            footprint = x**2 + y**2 <= pixel_radius**2

            # Blazing fast C-level matrix evaluation of all valleys
            local_mins = minimum_filter(elevation_grid, footprint=footprint, mode='nearest')

            # Prominence array isolation
            prominence_map = elevation_grid - local_mins

            if np.isnan(prominence_map).all():
                print("❌ No valid terrain found in this area matrix grid.")
                return

            max_prom_idx = np.unravel_index(np.nanargmax(prominence_map), prominence_map.shape)

            peak_elev = elevation_grid[max_prom_idx]
            base_elev = local_mins[max_prom_idx]
            max_prominence = prominence_map[max_prom_idx]

            # Reverse-engineer precise horizontal step scale to verify map distance
            sub_grid = elevation_grid[
                max(0, max_prom_idx[0]-pixel_radius):max_prom_idx[0]+pixel_radius+1,
                max(0, max_prom_idx[1]-pixel_radius):max_prom_idx[1]+pixel_radius+1
            ]
            min_sub_idx = np.unravel_index(np.nanargmin(sub_grid), sub_grid.shape)
            actual_base_row = max(0, max_prom_idx[0]-pixel_radius) + min_sub_idx[0]
            actual_base_col = max(0, max_prom_idx[1]-pixel_radius) + min_sub_idx[1]

            # Get coordinates for peak
            peak_lon, peak_lat = rasterio.transform.xy(transform, max_prom_idx[0], max_prom_idx[1])

            # Get coordinates for base
            base_lon, base_lat = rasterio.transform.xy(transform, actual_base_row, actual_base_col)

            # Calculate true distance between peak and base
            dx = (base_lon - peak_lon) * 111000 * math.cos(math.radians((peak_lat + base_lat) / 2))
            dy = (base_lat - peak_lat) * 111000
            true_distance = math.sqrt(dx**2 + dy**2)

            gradient = (max_prominence / true_distance) * 100
            slope_angle = math.degrees(math.atan(max_prominence / true_distance))

            print("\n" + "="*60)
            print(f"🏆 MAXIMUM DRAMATIC PROMINENCE FOUND (Under {max_distance_meters}m):")
            print(f"   • Peak Elevation:      {peak_elev:.1f} m")
            print(f"   • Peak Coordinates:    {peak_lat:.6f}°, {peak_lon:.6f}°")
            print(f"   • Base Elevation:      {base_elev:.1f} m")
            print(f"   • Base Coordinates:    {base_lat:.6f}°, {base_lon:.6f}°")
            print(f"   • Prominence Drop:     {max_prominence:.1f} m (Net Drop)")
            print(f"   • Precise Distance:    {true_distance:.1f} m")
            print(f"   • Profile Gradient:    {gradient:.2f}%")
            print(f"   • Slope Angle:         {slope_angle:.2f}°")
            print("="*60)
            print(f"\n📍 Quick Copy Coordinates:")
            print(f"   Peak: {peak_lat:.6f}, {peak_lon:.6f}")
            print(f"   Base: {base_lat:.6f}, {base_lon:.6f}")

# =====================================================================
# 📥 TARGET SELECTION
# Runs the range profile instantly
# =====================================================================
try:
    # Target Box format: (Min Longitude, Min Latitude, Max Longitude, Max Latitude)
    raw_data = fetch_ga_dem_wcs(152.982559,-28.515159,153.443985,-28.172507)

    # Execute with your exact maximum search window boundary
    find_max_prominence_fast(raw_data, max_distance_meters=3000)

except Exception as e:
    print(f"\nAn error occurred: {e}")
