"Find for every graph in a folder the relationship with ground and cover nature. This script uses dask_geopandas, networkx > 3.0, and shapely > 2.0."

import os
os.environ['USE_PYGEOS'] = '0' # Because PyGeos is installed and shapely 2.0 is also installed
import pickle
import geopandas as gpd
import networkx as nx
import shapely
import dask
import dask_geopandas

def dask_find_graph_nature_relationship(G, ground_nat, cover_nat, buff_size=20, inside_threshold=40):
    """
    Find for every edges of the graph G the relationship with ground nature (isolated, alongside, surrounded, inbetween and inside)
    and with cover nature like tree canopies (none, partial, full) based on a geometric relationship. Each status is defined as such:
    - Isolated: no ground nature around the edge on a buffer of buff_size
    - Alongside: ground nature around the edge on a single side
    - Surrounded: ground nature around the edge on both sides
    - Inbetween: ground nature intersecting with the edge, intersection is smaller than the inside_threshold percentage
    - Inside: ground nature intersecting with the edge, intersection is larger than the inside_threshold percentage
    - None: no cover nature intersecting with the edge
    - Partial: cover nature intersecting with the edge, intersection is smaller than the inside_threshold percentage
    - Full: cover nature intersecting with the edge, intersection is larger than the inside_threshold percentage

    ----------

    Parameters:
    - G : networkx.Graph
        Graph of a city, projected. Needs to be simplified, such that every edges have a geometry attribute.
    - ground_nat : geopandas.GeoDataFrame
        GeoDataFrame of ground nature, projected, which is nature where there is no coverage or that we can't decide.
        OpenStreetMap nature can be considered as ground nature, and every nature map code of ESA except 10 is ground nature.
    - cover_nat : geopandas.GeoDataFrame
        GeoDataFrame of cover nature, projected, which is nature where there is a natural coverage with trees.
        Map code 10 of ESA is cover nature.
    - buff_size : int, optional
        Size in meters of the buffer around the edge. We perform the spatial join with edges buffered to find nature
        that around the edges. By default is 20.
    - inside_threshold : int, optional
        Ratio in percentage of the threshold between inbetween and inside for ground nature and partial and full for cover
        nature. By default is 40.

    ----------

    Returns:
    - ground_nat_nearby : dict
        Dictionary with edges as keys and the ID of close ground nature as values.
        Dictionary only for edges with nature close.
    - cover_nat_nearby : dict
        Dictionary with edges as keys and the ID of close cover nature as values.
        Dictionary only for edges with nature close.
    - size_ha_dict : dict
        Dictionary with edges as keys and the sum of the size of close ground nature as values.
        Dictionary for every edges, edges that are isolated have a size of 0.
    - stat_dict : dict
        Dictionary with edges as keys and the status of the edge with ground nature as values.
    - cov_dict : dict
        Dictionary with edges as keys and the status of the edge with cover nature as values.

    """
    # Separate in chunks nature GeoDataFrame with dask so we can parallelize intensive computations
    d_ground_nat = dask_geopandas.from_geopandas(ground_nat, npartitions=4)
    d_cover_nat = dask_geopandas.from_geopandas(cover_nat, npartitions=4)
    # Transform the Graph into a GeoDataFrame
    u, v, data = zip(*G.edges(data=True))
    gdf_edges_raw = gpd.GeoDataFrame(data)
    gdf_edges_raw.set_geometry("geometry", inplace=True)
    gdf_edges_raw.crs = G.graph['crs']
    gdf_edges_raw["u"] = u
    gdf_edges_raw["v"] = v
    column_list = list(gdf_edges_raw)
    column_list.remove("u")
    column_list.remove("v")
    column_list.remove("geometry")
    # We remove every unnecessary columns that are edges' attributes to lighten the object
    gdf_edges = gdf_edges_raw.drop(columns=column_list)
    # We can't use a MultiIndex in a dask GeoDataFrame, need dictionaries to translate index to edges
    translate_dict = dict()
    translate_dict_rev = dict()
    for count, edge in enumerate(G.edges):
        translate_dict[count] = edge
        translate_dict_rev[edge] = count
    d_gdf_edges = dask_geopandas.from_geopandas(gdf_edges, npartitions=4)
    # Buffer around the edges instead of sjoin_nearest to change cap style and join style of the buffer zone
    d_gdf_edges_buff = d_gdf_edges.copy()
    d_gdf_edges_buff['geometry'] = d_gdf_edges_buff.geometry.buffer(buff_size, cap_style=2, join_style=3)
    # Dask_geopandas sjoin only support how='inner' as of now
    d_edge_sjoin = dask_geopandas.sjoin(d_gdf_edges_buff, d_ground_nat, how='inner')
    edge_sjoin = d_edge_sjoin.compute()
    # We use the difference between indexes with spatial join and the edges' GeoDataFrame to find isolated edges
    iso = gdf_edges.index.difference(edge_sjoin.index)
    # Aggregate by edges, as u and v are nodes' ID of the edges.
    edge_nature = edge_sjoin.groupby(["u", "v"])['index_right'].apply(lambda x: list(x))
    # Reindex
    edge_nature = edge_nature.set_axis([translate_dict_rev[edge] for edge in edge_nature.index.values])
    # Separate edges with a single polygon of close nature to the others
    one_poly = edge_nature[[True if len(val)==1 else False for val in edge_nature.values ]].apply(lambda x: x[0])
    one_poly.crs = G.graph['crs']
    mul_poly = edge_nature.drop(one_poly.index)
    mul_poly.crs = G.graph['crs']
    # Create GeoSeries with the single polygons, and do a unary union for the edges with multiple polygons close
    # Create a MultiPolygon where we can use the same functions than for the single polygon edges
    mp_geom = gpd.GeoSeries(data=mul_poly.apply(lambda x: shapely.ops.unary_union(ground_nat.geometry[x].values)),
                            index=mul_poly.index, crs=G.graph['crs'])
    op_geom = gpd.GeoSeries(data=ground_nat.geometry[one_poly.values].values,
                            index=one_poly.index, crs=G.graph['crs'])
    # Put it into dask
    d_mp_geom = dask_geopandas.from_geopandas(mp_geom, npartitions=4)
    d_op_geom = dask_geopandas.from_geopandas(op_geom, npartitions=4)
    # Merge the GeoSeries and work on all of them in the same way
    d_ap_geom = dask.dataframe.concat([d_op_geom, d_mp_geom])
    # Find the size in ha of close nature
    size_ha = d_ap_geom.geometry.area.compute() / 10000
    size_ha = dict(size_ha)
    # Add 0 for edges without close nature
    for edge in iso:
        size_ha[edge] = 0
    # Compute to get into a classic GeoDataFrame so we use it for indexing
    ap_geom = d_ap_geom.compute()
    # Find the length of intersection between edges and their close nature
    # Length of the intersection is divided by the length of the edge to find percentage
    ap_inter_length = 100 * (gdf_edges.geometry[ap_geom.index].intersection(ap_geom, align=False).length
                            / gdf_edges.geometry[ap_geom.index].length)
    # Separate ones with 0 intersection length, < than inside_threshold% of the length inbetween, > inside
    wo_inter = ap_inter_length[ap_inter_length.apply(lambda x: True if x==0 else False)].index
    w_inter = ap_inter_length.drop(wo_inter)
    inbet = list(w_inter[w_inter.apply(lambda x: True if x<=inside_threshold else False)].index)
    inside = list(w_inter.drop(inbet).index)
    # Ones without intersection are used to find surrounded and alongside edges
    gdf_edges_wo_inter = gdf_edges.geometry[wo_inter]
    ap_geom_wo_inter = ap_geom.geometry[wo_inter]
    d_ap_geom_wo_inter = dask_geopandas.from_geopandas(ap_geom_wo_inter, npartitions=4)
    d_pos_buff = dask_geopandas.from_geopandas(
        gdf_edges_wo_inter, npartitions=4).buffer(buff_size, cap_style=2, join_style=3, single_sided=True)
    d_neg_buff = dask_geopandas.from_geopandas(
        gdf_edges_wo_inter, npartitions=4).buffer(-buff_size, cap_style=2, join_style=3, single_sided=True)
    # Surrounded if both positive and negative buffer intersect with the polygons, else alongside
    pos_inter = d_pos_buff.geometry.intersects(d_ap_geom_wo_inter, align=False).compute()
    neg_inter = d_neg_buff.geometry.intersects(d_ap_geom_wo_inter, align=False).compute()
    surr = list(wo_inter[pos_inter & neg_inter])
    along = list(wo_inter.drop(surr))
    # Create a dict of status
    # We used the indexes, need to use dictionary to translate index to edges
    stat_dict = dict()
    st_res = ['isolated', 'alongside', 'surrounded', 'inbetween', 'inside']
    for num, st in enumerate([iso, along, surr, inbet, inside]):
        choice = st_res[num]
        for ind in st:
            stat_dict[translate_dict[ind]] = choice
    # ground_nat_nearby is not on every edges
    ground_nat_nearby = dict()
    for ind, val in zip(edge_nature.index.values, edge_nature.values):
        ground_nat_nearby[translate_dict[ind]] = val
    size_ha_dict = dict()
    for key, val in size_ha.items():
        size_ha_dict[translate_dict[key]] = val
    # For cover nature don't need to make a buffer since we only care about coverage of the edge
    d_edge_sjoin = dask_geopandas.sjoin(d_gdf_edges, d_cover_nat, how='inner')
    edge_sjoin = d_edge_sjoin.compute()
    # Edges without intersection found with difference of indexes
    # Procedure close to the one for ground nature, except that status are not the same
    none = gdf_edges.index.difference(edge_sjoin.index)
    edge_nature = edge_sjoin.groupby(["u", "v"])['index_right'].apply(lambda x: list(x))
    edge_nature = edge_nature.set_axis([translate_dict_rev[edge] for edge in edge_nature.index.values])
    one_poly = edge_nature[[True if len(val)==1 else False for val in edge_nature.values ]].apply(lambda x: x[0])
    one_poly.crs = G.graph['crs']
    mul_poly = edge_nature.drop(one_poly.index)
    mul_poly.crs = G.graph['crs']
    mp_geom = gpd.GeoSeries(
        data=mul_poly.apply(lambda x: shapely.ops.unary_union(cover_nat.geometry[x].values)),
        index=mul_poly.index, crs=G.graph['crs'])
    op_geom = gpd.GeoSeries(
        data=cover_nat.geometry[one_poly.values].values,
        index=one_poly.index, crs=G.graph['crs'])
    d_mp_geom = dask_geopandas.from_geopandas(mp_geom, npartitions=4)
    d_op_geom = dask_geopandas.from_geopandas(op_geom, npartitions=4)
    d_ap_geom = dask.dataframe.concat([d_op_geom, d_mp_geom])
    size_ha = d_ap_geom.geometry.area.compute() / 10000
    size_ha = dict(size_ha)
    for edge in none:
        size_ha[edge] = 0
    ap_geom = d_ap_geom.compute()
    ap_inter_length = 100 * (gdf_edges.geometry[ap_geom.index].intersection(ap_geom, align=False).length
                            / gdf_edges.geometry[ap_geom.index].length)
    partial = list(ap_inter_length[ap_inter_length.apply(lambda x: True if x<=inside_threshold else False)].index)
    full = list(ap_inter_length.drop(partial).index)
    cov_dict = dict()
    st_res = ['none', 'partial', 'full']
    for num, st in enumerate([none, partial, full]):
        choice = st_res[num]
        for ind in st:
            cov_dict[translate_dict[ind]] = choice
    cover_nat_nearby = dict()
    for ind, val in zip(edge_nature.index.values, edge_nature.values):
        cover_nat_nearby[translate_dict[ind]] = val
    return ground_nat_nearby, cover_nat_nearby, size_ha_dict, stat_dict, cov_dict


if __name__ == "__main__":
    graphpath = "../data/cities_osm_graph_newshapely/" # Path for city graph, graph need to be with geometry of shapely > 2.0
    natpath = "../data/cities_nature_filtered_poly/" # Path for ground and cover nature of cities
    respath = "../data/cities_graph_results_newshapely/" # Path for city graph with results
    # See graph_nature_relationship for these parameters
    buff_size = 20
    inside_threshold = 40
    # Get every graph in gpickle format inside the folder and their sizes
    filenames = [val for val in os.listdir(graphpath) if val.split('.')[1] == 'gpickle']
    sizes = [os.path.getsize(graphpath + file) for file in filenames]
    # Work on every graph, in ascending order of sizes
    for size, file in sorted(zip(sizes, filenames))[29:40]:
        # Template for names is {city}_graph.gpickle, remove last part to get the name
        cityname = file[:-14]
        print(f"Working on {cityname}")
        # New way to load graph on gpickle file with networkx > 3.0, from the graph folder
        with open(graphpath + file, 'rb') as f:
            G = pickle.load(f)
        # Load ground and cover nature from the nature folder
        ground_nat = gpd.read_file(f"../data/cities_nature_filtered_poly/{cityname}_ground.geojson")
        # Nature can't be save projected, need to project it to the city graph's CRS.
        ground_nat = ground_nat.to_crs(crs=G.graph['crs'])
        # Use buffer of 0 to make every geometry valid
        ground_nat.geometry = ground_nat.geometry.buffer(0)
        cover_nat = gpd.read_file(f"../data/cities_nature_filtered_poly/{cityname}_cover.geojson")
        cover_nat = cover_nat.to_crs(crs=G.graph['crs'])
        cover_nat.geometry = cover_nat.geometry.buffer(0)
        # Find the relationship between nature and edges for a city
        ground_nat_nearby, cover_nat_nearby, size_ha_dict, stat_dict, cov_dict = dask_find_graph_nature_relationship(
            G, ground_nat, cover_nat, buff_size=buff_size, inside_threshold=inside_threshold)
        G_res = G.copy()
        # Add as new attributes the relationships
        nx.set_edge_attributes(G_res, ground_nat_nearby, "ground_nat_nearby")
        nx.set_edge_attributes(G_res, cover_nat_nearby, "cover_nat_nearby")
        nx.set_edge_attributes(G_res, size_ha_dict, "size_ha")
        nx.set_edge_attributes(G_res, stat_dict, "status")
        nx.set_edge_attributes(G_res, cov_dict, "coverage")
        # Save the graph in the city graph with results folder
        with open(f"../data/cities_graph_results_newshapely/" + cityname + "_graph_res.gpickle", 'wb') as f:
            pickle.dump(G_res, f, pickle.HIGHEST_PROTOCOL)
