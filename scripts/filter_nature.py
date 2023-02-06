"Filter nature extracted from database to use it with OpenStreeMap graph."

import os
import pandas as pd
import geopandas as gpd
import networkx as nx


if __name__ == "__main__":
    # Avoid useless warnings to be printed
    pd.options.mode.chained_assignment = None
    # Size of the buffer for the bounding box of the nature in meters, to be compared with 
    # the buffer used around the edges for the relationship
    buff_size = 20
    graphpath = "../data/cities_osm_graph/" # Path for city graph
    natpath = "../data/cities_nature_poly/" # Path for raw ground and cover nature of cities
    filtpath = "../data/cities_nature_filtered_poly/" # Path for filtered ground and cover nature of cities
    filenames = [val for val in os.listdir(graphpath) if val.split('.')[1] == 'gpickle']
    for file in filenames:
        cityname = file.split("_")[0]
        cityname = cityname.replace(" ", "_")
        print(f"Working on {cityname}")
        G = nx.read_gpickle(graphpath + file)
        esa_nat = gpd.read_file(natpath + cityname + "_esa.geojson")
        osm_nat = gpd.read_file(natpath + cityname + "_osm.geojson")
        # Keep map code for ground nature, every nature map code except 10
        ground_esa = esa_nat[esa_nat['code'].isin([20, 30, 40, 80, 90, 95, 100])]
        # Separate green and blue ground nature
        ground_esa['type'] = ground_esa['code'].apply(
            lambda x: 'blue' if x in [80, 90, 95, 100] else 'green').values
        ground_nat = pd.concat([ground_esa, osm_nat], ignore_index=True)
        cover_nat = esa_nat[esa_nat['code'] == 10]
        # Project it to the graph CRS.
        ground_nat.to_crs(crs=G.graph['crs'], inplace=True)
        u, v, data = zip(*G.edges(data=True))
        gdf_edges = gpd.GeoDataFrame(data)
        gdf_edges.set_geometry("geometry", inplace=True)
        # Find the bounding box of the graph, use it to filter ground and cover nature
        bounds = gdf_edges.bounds
        xmin = min(bounds['minx'])
        ymin = min(bounds['miny'])
        xmax = max(bounds['maxx'])
        ymax = max(bounds['maxy'])
        ground_nat = ground_nat.cx[
            xmin - 2*buff_size : xmax + 2*buff_size,
            ymin - 2*buff_size : ymax + 2*buff_size]
        # Merge in a unary_union intersecting nature polygons if they're of the same type of nature
        blue_ground_nat = ground_nat[ground_nat['type'] == 'blue']
        green_ground_nat = ground_nat[ground_nat['type'] == 'green']
        blue_merged = gpd.GeoDataFrame(geometry=[geom for geom in blue_ground_nat.unary_union.geoms], crs=G.graph['crs'])
        blue_merged['type'] = 'blue'
        green_merged = gpd.GeoDataFrame(geometry=[geom for geom in green_ground_nat.unary_union.geoms], crs=G.graph['crs'])
        green_merged['type'] = 'green'
        ground_nat = pd.concat([green_merged, blue_merged], ignore_index=True)
        ground_nat['size_ha'] = ground_nat.geometry.area / 10000
        cover_nat.to_crs(crs=G.graph['crs'], inplace=True)
        cover_nat = cover_nat.cx[xmin:xmax, ymin:ymax]
        cover_nat = gpd.GeoDataFrame(geometry=[geom for geom in cover_nat.unary_union.geoms], crs=G.graph['crs'])
        # Coverage is a treen canopy, always green
        cover_nat['type'] = 'green'
        cover_nat['size_ha'] = cover_nat.geometry.area / 10000
        # Can't always save projected nature, need to project it back to global CRS
        ground_nat.to_crs(crs="epsg:4326", inplace=True)
        cover_nat.to_crs(crs="epsg:4326", inplace=True)
        # The projection might create invalid geometries, need to make them valid later on
        ground_nat.to_file(filtpath + cityname + "_ground.geojson", driver='GeoJSON')
        cover_nat.to_file(filtpath + cityname + "_cover.geojson", driver='GeoJSON')




