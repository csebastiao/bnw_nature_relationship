# -*- coding: utf-8 -*-
"""
Set of functions using osmnx, networkx < 3.0, and shapely < 2.0.
"""

### Modules
# Math
import numpy as np
# Geometry
import shapely
# DataFrame
import pandas as pd
import geopandas as gpd
# Network
import networkx as nx
import osmnx as ox
# Custom
from nerds_osmnx.simplification import simplify_graph, multidigraph_to_graph

### Functions
def create_additional_point(fp, lp, dist):
    """
    Create a new point that is at a specific distance of the first point fp,
    on the same line as fp and lp.
    """
    fp = np.array(fp)
    lp = np.array(lp)
    uv = (fp - lp)/np.linalg.norm(fp - lp) # unit vector between fp and vp
    newp = fp - uv*dist
    return newp


def threshold_edge_size(tested_edge, max_length=20, verbose=False):
    """Divide a LineString into multiple smaller LineString of even size such
    as every LineString is smaller than the max_length"""
    old_edge = tested_edge
    # if > than the max_length, divide it into smaller even pieces
    if old_edge.length > max_length:
        new_edges = []
        # if exact multiple, no need to add 1 to the integer division
        if old_edge.length%max_length != 0:
            number_div = old_edge.length // max_length + 1
        else:
            number_div = old_edge.length // max_length
        edge_size = old_edge.length / number_div
        if verbose is True: # debug option
            print("Number of new edges = ", number_div,
                  "edges of length", edge_size)
        # number of coordinates of the LineString for the new edge
        npoint = 2 
        while len(new_edges) < number_div:
            # Add coord until edge > edge_size, then cut to exact value
            # by creating a new coord
            if shapely.geometry.LineString(
                    old_edge.coords[:npoint]).length > edge_size:
                # need to separate because we can't have an empty LineString
                if npoint > 2:
                    new_point = create_additional_point(
                        old_edge.coords[npoint-2], old_edge.coords[npoint-1],
                        edge_size - shapely.geometry.LineString(
                            old_edge.coords[:npoint-1]).length)
                else:
                    new_point = create_additional_point(
                        old_edge.coords[0], old_edge.coords[1],
                        edge_size)
                new_coords = old_edge.coords[:npoint-1]
                new_coords.append(new_point)
                new_edges.append(shapely.geometry.LineString(new_coords))
                updated_old_coords = old_edge.coords[npoint-1:]
                updated_old_coords.insert(0, new_point)
                old_edge = shapely.geometry.LineString(updated_old_coords)
                npoint = 2
            elif len(old_edge.coords[:]) == npoint:
                new_edges.append(old_edge)
            else:
                npoint += 1
        return new_edges
    else:
        return old_edge

    
def divide_evenly_edges(G, max_length=20):
    # Node dataframe
    u, data = zip(*G.nodes(data=True))
    gdf_nodes = gpd.GeoDataFrame(data)
    node_stored = []
    edge_stored = []
    edge_removed = []
    number = max(G)
    for edge in G.edges:
        # Create a list of LineString if edge's length > max_length
        saved_edges = threshold_edge_size(G.edges[edge]['geometry'], max_length=max_length, verbose=False)
        if isinstance(saved_edges, list):
            # Store every edge attribute for every new edge except geometry and length
            edge_attr_dict = G.edges[edge].copy()
            edge_attr_dict.pop('geometry')
            edge_attr_dict.pop('length')
            # Remove old long edge
            edge_removed.append(edge)
            f_coord = G.edges[edge]['geometry'].coords[0]
            # Find the node which is the first coordinate of the edge geometry
            bef = gdf_nodes[(gdf_nodes['x']==f_coord[0]) & (gdf_nodes['y']==f_coord[1])].index[0]
            first = bef
            for e in saved_edges[:-1]:
                # Add new node with unique ID
                number += 1
                node_stored.append([number,
                        {'x':e.coords[-1][0], 'y':e.coords[-1][1]}])
                # Add specific geometry and length to the edge's attribute dictionary
                temp_dict = edge_attr_dict.copy()
                temp_dict['length'] = e.length
                temp_dict['geometry'] = e
                edge_stored.append([bef, number, temp_dict])
                bef = number
            temp_dict = edge_attr_dict.copy()
            temp_dict['geometry'] = saved_edges[-1]
            temp_dict['length'] = saved_edges[-1].length
            if first == edge[0]:
                last = edge[1]
            elif first == edge[1]:
                last = edge[0]
            edge_stored.append([bef, last, temp_dict])
        else:
            pass
    H = G.copy()
    H.add_nodes_from(node_stored)
    H.remove_edges_from(edge_removed)
    H.add_edges_from(edge_stored)
    for edge in H.edges:
        # Not a new edge but an old edge without the right edge's length
        if H.edges[edge]['length'] > max_length and edge in G.edges:
            print("""Miscalculation for edge {},
                show length as {} while it is {}""".format(
                edge, H.edges[edge]['length'], H.edges[edge]['geometry'].length))
            H.edges[edge]['length'] = H.edges[edge]['geometry'].length
        else:
            pass
    return H


def desirable_cycling(G, w_init=3, threshold_ha=1, w_comfort=1,
                      w_green=0.125, val_status=None, attr='cost_function'):
    """
    Add a new attribute accounting for the desirability of cycling on 
    an edge based on the attribute: either there is bicycling infrastructure
    and the relationship to nature and the size of nearby nature.
    """
    G = G.copy()
    if val_status is None:
        val_status = {'isolated':1, 'alongside':2, 'inbetween':3,
                      'surrounded':4, 'inside':8}
    cost_dict = dict()
    desi_dict = dict()
    for edge in G.edges:
        ed = G.edges[edge]
        cost_function = ed['length'] * (
            max(w_comfort + w_green * max(val_status.values()) * threshold_ha,
                w_init) # Such that cost_function always positive or null
            - w_comfort * int(ed['protected_bicycling'])
            - w_green * val_status[ed['status']]
            * min(threshold_ha, ed['size_green_ha']))
        cost_dict[edge] = cost_function
        desi_function = (w_comfort * int(ed['protected_bicycling'])
                         + w_green * val_status[ed['status']]
                         * min(threshold_ha, ed['size_green_ha']))
        desi_dict[edge] = desi_function
    if attr == 'cost_function':
        nx.set_edge_attributes(G, cost_dict, name="cost_function")
    elif attr == 'desirability':
        nx.set_edge_attributes(G, desi_dict, name="desirability")
    elif attr == 'both':
        nx.set_edge_attributes(G, cost_dict, name="cost_function")
        nx.set_edge_attributes(G, desi_dict, name="desirability")
    else:
        raise ValueError('Incorrect value for attr.')
    return G


def merge_dicts(dicts):
    """
    Merge a list of dictionaries as one. If a key is present in multiple
    dictionaries the values of multiple dictionaries are merged.
    """
    longest_dict_id = np.argmax([len(elem) for elem in dicts])
    new_dict = dicts[longest_dict_id]
    other_dict = dicts
    other_dict.pop(longest_dict_id)
    for d in other_dict:
        for key in d:
            if key in new_dict:
                for val in d[key]:
                    if val not in new_dict[key]:
                        new_dict[key].append(val)
            else:
                new_dict[key] = d[key]
    return new_dict


def add_edge_attribute(G, attr_dict, name, bool_response=True):
    """
    Add an edge attribute where the value are binary bool based on
    whether the edge have a specific value for a given attribute,
    given as a dictionary.

    Parameters
    ----------
    G : networkx Graph/DiGraph/MultiGraph/MultiDiGraph/...
        Graph on which we want to add an attribute.
    attr_dict : dict
        Dictionary where the key are the key of the edges' attributes
        and values are the values of those attributes that we want to
        take into account.
    name : str
        Name of the new attribute.
    bool_response : bool, optional
        Bool response if we find one of the values on one of the
        attributes of the edges from the dictionary.
        The default is True.

    Raises
    ------
    NameError
        Raised if the name is already an attribute of an edge
        of the graph, in order to avoid unintended mix.

    Returns
    -------
    G : networkx Graph/DiGraph/MultiGraph/MultiDiGraph/...
        Graph with the new binary attribute.

    """
    G = G.copy()
    for edge in G.edges:
        if name in G.edges[edge]:
            raise NameError(
                "New attribute {} already in edge {}, use a new name".format(
                    name, edge)
                )
        for key in list(attr_dict.keys()):
            if key in list(G.edges[edge].keys()):
                if G.edges[edge][key] in attr_dict[key]:
                    G.edges[edge][name] = bool_response
                    break # otherwise next key can replace the value
                else:
                    G.edges[edge][name] = not bool_response
    return G


def filter_bicycle_network(city=None, G=None, level=1, merge_level=False):
    """
    Add new attribute of bikeability based on OSM tags of the graph G,
    either given in the function or retrieved from the name of the city.
    """
    if level not in [1, 2, 3, 4]:
        raise ValueError("Level chosen not between 1 and 4.")
    if G is None and city is None:
        raise ValueError(
            "You have not specified a graph or the name of a city.")
    # Filters include bikeable roads
    prot_bnw = {
        'cycleway': ['track'],
        'cycleway:left': ['track'],
        'cycleway:right': ['track'],
        'cycleway:both': ['track'],
        'bicycle': ['designated'],
        'bicycle_road': ['yes'],
        'cyclestreet': ['yes'],
        'highway': ['cycleway', 'path']}
    unprot_bnw = {
        'cycleway': ['lane', 'opposite_lane', 'share_busway',
                     'shared_lane', 'designated' , 'yes'] ,
        'cycleway:left': ['lane', 'opposite_lane', 'share_busway',
                          'shared_lane', 'designated' , 'yes'] ,
        'cycleway:right': ['lane', 'opposite_lane', 'share_busway',
                           'shared_lane', 'designated' , 'yes'] ,
        'cycleway:both': ['lane', 'opposite_lane', 'share_busway',
                          'shared_lane', 'designated' , 'yes'] ,
        'bicycle': [ 'yes', 'permissive', 'destination' , 'private' ],
        'highway': ['living_street']}
    ext_bnw = {
        'highway': ['residential'],
        'maxspeed': ['5', '10', '15', '20', '30']}
    # Work in the opposite way: exclude not include bikeable roads
    leg_bnw = {'area': ['yes'],
               'highway': ['footway', 'steps', 'corridor', 'elevator',
                           'escalator', 'motor', 'proposed', 'construction',
                           'abandoned', 'platform', 'raceway', 'motorway',
                           'motorway_link', 'planned', 'proposed',
                           'bus_guideway'],
               'bicycle': ['no'],
               'service': ['private']}

    if G is None:
        for tag_name in list(['cycleway', 'cycleway:left', 'cycleway:right',
                              'cycleway:both', 'bicycle', 'bicycle_road',
                              'cyclestreet', 'highway', 'maxspeed', 'service',
                              'area']):
            if tag_name not in ox.settings.useful_tags_way:
                ox.settings.useful_tags_way += [tag_name]
        G = ox.graph_from_place(city, simplify=False)
    #TODO: Test if level 4 works
    if merge_level is False:
        for l, d in enumerate([prot_bnw, unprot_bnw, ext_bnw][:level]):
            G = add_edge_attribute(G, d, "level_{}".format(l+1))
        if level == 4:
            G = add_edge_attribute(G, leg_bnw, "level_4", bool_response=False)
    else:
        if level == 1:
            G = add_edge_attribute(G, prot_bnw, "level_1")
        elif level == 4:
            merged_dict = merge_dicts([prot_bnw, unprot_bnw, ext_bnw])
            G = add_edge_attribute(G, merged_dict, "level_1-3")
            G = add_edge_attribute(G, leg_bnw, "level_4", bool_response=False)
            G = add_edge_attribute(G, {'level_1-3':[True], 'level_4':[True]},
                                   'level_1-4')
            for e in G.edges:
                for attr in ['level_1-3', 'level_4']:
                    G.edges[e].pop(attr)
        else:
            merged_dict = merge_dicts([prot_bnw, unprot_bnw, ext_bnw][:level])
            G = add_edge_attribute(G, merged_dict, "level_1-{}".format(level))
    return G


def graph_to_geojson(G, path):
    """
    Transform a networkx graph G to two geojson file, one for node and
    one for edges at the specified path.
    """
    if type(G) != nx.MultiDiGraph:
        G = nx.MultiDiGraph(G)
    # can't have list as attribute so change every list to a string
    for edge in G.edges:
        for attr in G.edges[edge]:
            if type(G.edges[edge][attr]) == list:
                G.edges[edge][attr] = str(G.edges[edge][attr])[1:-1]
    gdf = ox.graph_to_gdfs(G, nodes=True, edges=True)
    gdf[0].to_file(path + '_node.geojson', driver='GeoJSON')
    gdf[1].to_file(path + '_edge.geojson', driver='GeoJSON')


def create_city_graph(city, country, save_folder, crs=None, max_length_edge=20, G=None):
    """
    Create and save a graph from OpenStreetMap with a bikeability level attribute and with edges of even sizes of max_length_edge.

    ----------

    Parameters:
    - city : str
        Name of the city where we want to create the graph.
    - country : str
        Name of the country in which the city is.
    - save_folder : str
        Path to the folder where the graph is saved.
    - crs : str, optional
        Projected CRS to accurately project the graph. By default is None.
    - max_length_edge : int, optional
        Maximum size of the edges, in meters. By default is 20.
    - G : networkx.Graph, optional
        If the graph is already extracted, use this one instead of getting it with osmnx.
        By default is None.

    ----------

    Returns:
    - H : networkx.Graph or None
        If G is None and the graph is not found, returns None, else returns the curated graph.

    """
    # If no graph is given, extract it with osmnx
    if G is None:
        # Add useful tags for bikeability
        for tag_name in list(['cycleway', 'cycleway:left', 'cycleway:right',
                            'cycleway:both', 'bicycle', 'bicycle_road',
                            'cyclestreet', 'highway', 'maxspeed', 'service',
                            'area']):
            if tag_name not in ox.settings.useful_tags_way:
                ox.settings.useful_tags_way += [tag_name]
        # A graph is not always found, shouldn't simplify at first
        try:
            G = ox.graph_from_place(city + ', ' + country, simplify=False)
        except:
            print(f"No graph found for {city}, {country} in OSM !")
            return None
    # Now that there is a graph, create bikeability attribute
    G = filter_bicycle_network(G=G, level=4, merge_level=False)
    # Simplify 4 boolean attributes to one with 5 values
    for edge in G.edges:
        if G.edges[edge]['level_1'] == True:
            G.edges[edge]['bikeability_level'] = 1
        elif G.edges[edge]['level_2'] == True:
            G.edges[edge]['bikeability_level'] = 2
        elif G.edges[edge]['level_3'] == True:
            G.edges[edge]['bikeability_level'] = 3
        elif G.edges[edge]['level_4'] == True:
            G.edges[edge]['bikeability_level'] = 4
        else:
            G.edges[edge]['bikeability_level'] = 5
    # We simplify the edges after adding the bikeability attribute to keep discriminating them
    # We need to simplify to reduce as much as we can the number of nodes
    G = simplify_graph(G, attributes=['bikeability_level'])
    # Project to make a threshold in meters on edges' length
    G = ox.project_graph(G, to_crs=crs)
    # Reduce size of the graph by removing directions of the edges
    G = multidigraph_to_graph(G, attributes=['bikeability_level'])
    # Relabel as the nodes' ID are meaningless and easier to add new nodes with unique ID
    G_relab = nx.convert_node_labels_to_integers(G)
    for e in G_relab.edges:
        G_relab.edges[e]['length'] = G_relab.edges[e]['geometry'].length
    # Divide edges evenly so most of them make the same small size
    H = divide_evenly_edges(G_relab, max_length=max_length_edge)
    # Save the graph, with networkx < 3.0
    # Since we use osmnx, will be saved with geometry of shapely < 2.0
    nx.write_gpickle(H, save_folder + f"{city}_{country}_graph.gpickle")
    return H

def find_graph_nature_relationship(G, ground_nat, cover_nat, buff_size=20, inside_threshold=40):
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
    translate_dict = dict()
    translate_dict_rev = dict()
    for count, edge in enumerate(G.edges):
        translate_dict[count] = edge
        translate_dict_rev[edge] = count
    # Buffer around the edges instead of sjoin_nearest to change cap style and join style of the buffer zone
    gdf_edges_buff = gdf_edges.copy()
    gdf_edges_buff['geometry'] = gdf_edges_buff.geometry.buffer(buff_size, cap_style=2, join_style=3)
    # Dask_geopandas sjoin only support how='inner' as of now
    edge_sjoin = gpd.sjoin(gdf_edges_buff, ground_nat, how='inner')
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
    # Merge the GeoSeries and work on all of them in the same way
    ap_geom = pd.concat([op_geom, mp_geom])
    # Find the size in ha of close nature
    size_ha = ap_geom.geometry.area / 10000
    size_ha = dict(size_ha)
    # Add 0 for edges without close nature
    for edge in iso:
        size_ha[edge] = 0
    # Compute to get into a classic GeoDataFrame so we use it for indexing
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
    pos_buff = gdf_edges_wo_inter.buffer(buff_size, cap_style=2, join_style=3, single_sided=True)
    neg_buff = gdf_edges_wo_inter.buffer(-buff_size, cap_style=2, join_style=3, single_sided=True)
    # Surrounded if both positive and negative buffer intersect with the polygons, else alongside
    pos_inter = pos_buff.geometry.intersects(ap_geom_wo_inter, align=False)
    neg_inter = neg_buff.geometry.intersects(ap_geom_wo_inter, align=False)
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
    d_edge_sjoin = gpd.sjoin(gdf_edges, cover_nat, how='inner')
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
    ap_geom = pd.concat([op_geom, mp_geom])
    size_ha = ap_geom.geometry.area / 10000
    size_ha = dict(size_ha)
    for edge in none:
        size_ha[edge] = 0
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