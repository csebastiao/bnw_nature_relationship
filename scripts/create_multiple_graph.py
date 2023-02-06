"Extract graph from OpenStreetMap. Script using osmnx, as of now not compatible with shapely > 2.0"

import pandas as pd
import geopandas as gpd
from func_osmnx import create_city_graph

if __name__ == "__main__":
    # Read the UCDB file to find cities we want to explore
    ucdb=gpd.read_file('../data/GHS_STAT_UCDB2015MT_GLOBE_R2019A_V1_2.gpkg')
    # Transform str to integers to rank by population
    ucdb['P15'] = pd.to_numeric(ucdb['P15'])
    # Filter only to a specific region
    ucdb_eu = ucdb[ucdb['GRGN_L2'].isin(['Southern Europe', 'Northern Europe', 'Western Europe'])]
    # Get in this region the 50 largest cities
    largest = ucdb_eu.nlargest(50, 'P15')
    # Get the names of the cities and the corresponding countries
    cities = largest['UC_NM_MN'].values
    countries = largest['CTR_MN_NM'].values
    graphpath = "../data/cities" # Path for city graph
    for loc in zip(cities, countries):
        print(f"Create graph for {loc[0]}, {loc[1]}")
        G = create_city_graph(*loc, graphpath)
