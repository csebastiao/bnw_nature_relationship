"Create a summary of the results of all graph."

import os
import numpy as np
import shapely
import pandas as pd
import geopandas as gpd
import networkx as nx
import osmnx as ox
from collections import Counter
import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Read UCDB database to get the city we wanted to explore, even the ones without a graph yet
    ucdb=gpd.read_file('../data/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg')
    ucdb['P15'] = pd.to_numeric(ucdb['P15'])
    ucdb_eu = ucdb[ucdb['GRGN_L2'].isin(['Southern Europe', 'Northern Europe', 'Western Europe'])]
    largest = ucdb_eu.nlargest(50, 'P15')
    cities = largest['UC_NM_MN'].values
    countries = largest['CTR_MN_NM'].values
    pop = largest['P15'].values
    graphpath = "../data/cities_osm_graph/" # Path for city graph without results of relationship with nature
    # Get cities in gpickle format inside the folder
    graphnames = [val for val in os.listdir(graphpath) if val.split('.')[1] == 'gpickle']
    respath = "../data/cities_graph_results/" # Path for city with results
    resnames = [val for val in os.listdir(respath) if val.split('.')[1] == 'gpickle']
    natpath = "../data/cities_nature_filtered_poly/" # Path for filtered ground and cover nature for cities
    anpath = "../data/cities_analysis/" # Path to save the results
    stat_graphs = []
    ngraph_metrics = 5 # Number of metrics for graph with or without results
    nres_metrics = 19 # Number of metrics for graph only with results
    plt.ioff() # Remove interactive plotting to avoid opening figures
    for city, country, p in zip(cities, countries, pop):
        cityname = city.replace(" ", "_")
        # If there is a graph with results, full analysis
        if f"{cityname}_graph_res.gpickle" in resnames:
            G = nx.read_gpickle(respath + f"{cityname}_graph_res.gpickle")
            # Get number of nodes
            gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
            total_length = int(sum([G.edges[edge]['length'] for edge in G.edges]))
            lat = [val for val in gdf_nodes['lat'].values if str(val) != 'nan']
            lon = [val for val in gdf_nodes['lon'].values if str(val) != 'nan']
            points = np.transpose(np.array([lon, lat]))
            # Create a convex hull from the non-projected nodes as the borders of the city
            poly = shapely.geometry.MultiPoint(points).convex_hull
            # Find the area of the convex hull from the projected nodes
            area = int(shapely.geometry.MultiPoint(gdf_nodes.geometry.values).convex_hull.area)
            # Load nature
            ground_nat = gpd.read_file(natpath + f"{cityname}_ground.geojson")
            cover_nat = gpd.read_file(natpath + f"{cityname}_cover.geojson")
            # Nature need to be projected to the graph CRS.
            ground_nat.to_crs(crs=G.graph['crs'], inplace=True)
            cover_nat.to_crs(crs=G.graph['crs'], inplace=True)
            # Find number of green, blue, and cover areas
            ground_count = Counter(ground_nat['type'])
            green_area = int(sum(ground_nat[ground_nat['type'] == 'green'].geometry.area))
            blue_area = int(sum(ground_nat[ground_nat['type'] == 'blue'].geometry.area))
            cover_area = int(sum(cover_nat.geometry.area))
            # From the GeoDataFrame of the edges, find values from ground status, cover status, and bikeability
            # See find_graph_nature_relationship
            u, v, data = zip(*G.edges(data=True))
            gdf_edges = gpd.GeoDataFrame(data)
            gdf_edges.set_geometry("geometry", inplace=True)
            s_count = Counter(gdf_edges['status'])
            c_count = Counter(gdf_edges['coverage'])
            b_count = Counter(gdf_edges['bikeability_level'])
            stat_graphs.append([
                city, country, p, True, True,
                poly, len(G), len(G.edges), area, total_length,
                ground_count['green'], green_area, ground_count['blue'], blue_area, len(cover_nat), cover_area,
                b_count[1], b_count[2], b_count[3], b_count[4], b_count[5],
                s_count['isolated'], s_count['alongside'], s_count['inbetween'], s_count['surrounded'], s_count['inside'],
                c_count['none'], c_count['full'], c_count['partial']])
            # Order categorical values for plotting with a colormap
            gdf_edges['status'] = pd.Categorical(gdf_edges['status'], ['isolated', 'alongside', 'inbetween', 'surrounded', 'inside'])
            gdf_edges['coverage'] = pd.Categorical(gdf_edges['coverage'], ['none', 'partial', 'full'])
            sns.set_style('white')
            # Plot edges colored with ground nature status and ground nature
            fig, ax = plt.subplots(figsize=(32, 18))
            custom_cmap = mpl.colors.ListedColormap(["gray", "darkred", "darkorange", "gold", "darkgreen"])
            gdf_edges.plot(column='status', alpha=0.8, linewidth=1, ax=ax, cmap=custom_cmap, legend=True)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ground_nat.plot(column='type', alpha=0.3, ax=ax, cmap='winter')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Relationship to ground nature")
            # Avoid useless blank space
            fig.set_tight_layout(True)
            # Save the figure in the results folder
            fig.savefig(anpath + f"{cityname}_ground.png")
            # Always close every figure to avoid blotting the memory
            plt.close(fig)
            # Plot edges colored with cover nature status and cover nature
            fig, ax = plt.subplots(figsize=(32, 18))
            custom_cmap = mpl.colors.ListedColormap(["grey", "yellow", "green"])
            gdf_edges.plot(column='coverage', alpha=0.8, linewidth=1, ax=ax, cmap=custom_cmap, legend=True)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            cover_nat.plot(column='type', alpha=0.3, ax=ax, cmap='winter_r')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Relationship to cover nature")
            fig.set_tight_layout(True)
            fig.savefig(anpath + f"{cityname}_cover.png")
            plt.close(fig)
            # Plot edges colored with bikeability level and ground nature
            fig, ax = plt.subplots(figsize=(32, 18))
            custom_cmap = mpl.colors.ListedColormap(["darkgreen", "gold", "darkorange", "darkred", "gray"])
            gdf_edges.plot(column='bikeability_level', alpha=0.8, linewidth=1, ax=ax, cmap=custom_cmap, legend=True)
            ylim = ax.get_ylim()
            xlim = ax.get_xlim()
            ground_nat.plot(column='type', alpha=0.3, ax=ax, cmap='winter')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_yticks([])
            ax.set_xticks([])
            ax.set_title("Bikeability level")
            fig.set_tight_layout(True)
            fig.savefig(anpath + f"{cityname}_bikeability.png")
            plt.close(fig)
        # If there is only a graph without results, partial analysis
        elif f"{city}_{country}_graph.gpickle" in graphnames:
            G = nx.read_gpickle(graphpath + f"{city}_{country}_graph.gpickle")
            gdf_nodes = ox.graph_to_gdfs(G, nodes=True, edges=False)
            total_length = int(sum([G.edges[edge]['length'] for edge in G.edges]))
            lat = [val for val in gdf_nodes['lat'].values if str(val) != 'nan']
            lon = [val for val in gdf_nodes['lon'].values if str(val) != 'nan']
            points = np.transpose(np.array([lon, lat]))
            poly = shapely.geometry.MultiPoint(points).convex_hull
            area = int(shapely.geometry.MultiPoint(gdf_nodes.geometry.values).convex_hull.area)
            stat_graphs.append([
                city, country, p, True, False,
                poly, int(len(G)), int(len(G.edges)), int(area), int(total_length),
                *[np.nan]*nres_metrics])
        # If there is not even a graph without results, only give informations from the UCDB
        else:
            stat_graphs.append([
                city, country, p, False, False,
                *[np.nan] * (ngraph_metrics+nres_metrics)])
    # Aggregate results to a GeoDataFrame, save it in the results folder
    gdf = gpd.GeoDataFrame(data=stat_graphs, columns=[
        'city', 'country', 'population', 'wgraph', "wres",
        'geometry', '#nodes', '#edges', 'area', 'total road length',
        '#green', 'green area', '#blue', 'blue area', '#cover', 'cover area',
        '#bikeability=1', '#bikeability=2', '#bikeability=3', '#bikeability=4', '#bikeability=5',
        '#isolated', '#alongside', '#inbetween', '#surrounded', '#inside',
        '#none', '#partial', '#full'])
    gdf.to_file(anpath + 'recap_cities.geojson', driver='GeoJSON')
