import networkx as nx
from torch_geometric.datasets import TUDataset
import os
import torch_geometric.utils as pyg_utils
import pickle
from loguru import logger
import numpy as np
import random
import scipy.stats
from random import randrange
from pebble import ProcessPool


def check_subiso(data):
  gc,gq = data
  return nx.algorithms.isomorphism.GraphMatcher(gc, gq).subgraph_is_isomorphic()


def run_parallel_pool(func,input_list):
    all_result = []
    len_list = len(input_list)

    with ProcessPool(max_workers=100) as pool:
        future = pool.map(func, input_list, timeout=30)
        iterator = future.result()

        for c_i in range(len_list):
              try:
                  result = next(iterator)
                  all_result.append(result)

              except StopIteration:
                  break
              except TimeoutError:
                  all_result.append(None)
              except AssertionError:
                  all_result.append(None)
    return all_result


def get_nx_graph(pyg_graphs):
    d_nx = []
    for g in list(pyg_graphs):
        g_nx = pyg_utils.to_networkx(g, to_undirected=True)
        d_nx.append(g_nx)
    return d_nx


def fetch_tudataset_graphs(conf):
    """
      Fetches the dataset from the path specified in av.DIR_PATH
      If dataset is not present, it downloads the dataset
    """
    directory = f"{conf.base_dir}{conf.dataset.path}/{conf.dataset.name}"
    fname = f"{directory}/graphs.pkl"
    logger.info(f"Looking for Original graphs in {fname}")
    if os.path.isfile(fname):
        pyg_graphs = pickle.load(open(fname,"rb"))
    else:
        if conf.dataset.name == "ptc_fr":
            pyg_graphs = TUDataset(root="/tmp/PTC_FR", name="PTC_FR")
        elif conf.dataset.name == "ptc_fm":
            pyg_graphs = TUDataset(root="/tmp/PTC_FM", name="PTC_FM")
        elif conf.dataset.name == "ptc_mr":
            pyg_graphs = TUDataset(root="/tmp/PTC_MR", name="PTC_MR")
        elif conf.dataset.name == "ptc_mm":
            pyg_graphs = TUDataset(root="/tmp/PTC_MM", name="PTC_MM")
        elif conf.dataset.name == "mutag":
            pyg_graphs = TUDataset(root="/tmp/MUTAG", name="MUTAG")
        elif conf.dataset.name == "aids":
            pyg_graphs = TUDataset(root="/tmp/AIDS", name="AIDS")
        elif conf.dataset.name == "enzymes":
            pyg_graphs = TUDataset(root="/tmp/ENZYMES", name="ENZYMES")
        elif conf.dataset.name == "dd":
            pyg_graphs = TUDataset(root="/tmp/DD", name="DD")
        elif conf.dataset.name == "cox2":
            pyg_graphs = TUDataset(root="/tmp/COX2", name="COX2")
        elif conf.dataset.name == "msrc_21":
            pyg_graphs = TUDataset(root="/tmp/MSRC_21", name="MSRC_21")
        else:
            raise NotImplementedError()
        if not os.path.exists(directory):
          os.makedirs(directory)
        with open(fname,"wb") as f:
            pickle.dump(pyg_graphs,f)

    graphs = get_nx_graph(pyg_graphs)
    logger.info(f"Loaded {len(graphs)} graphs from {conf.dataset.name} dataset")
    return graphs
  

def random_graph_generator(seed_graph, num_new_nodes, max_edges_per_node, a, b):
    """
      seed_graph: nx graph - may be nx.empty_graph()
      num_new_nodes: int denoting o of new vertices to add 
      a*degree + b : choose nodes of new connection 
      TODO: how to factor in edge density ? 
    """
    if seed_graph is None:
        seed_graph = nx.empty_graph(1)
        num_new_nodes = num_new_nodes-1

    new_graph = seed_graph.copy()
    new_node_list = np.arange(num_new_nodes) + seed_graph.number_of_nodes()

    for node in new_node_list:
        degree_list = np.array([new_graph.degree[x] for x in new_graph.nodes])
        weighted_degree_list = degree_list*a+b
        probs =weighted_degree_list / np.sum(weighted_degree_list)
        dist = scipy.stats.rv_discrete(values=(np.arange(len(probs)), probs))
        num_new_edges = min(randrange(1,max_edges_per_node),len(probs))
        nbr_set = set()
        while(len(nbr_set)!=num_new_edges):
            nbr_set.add(dist.rvs())
        for nbr in nbr_set:
            new_graph.add_edge(nbr,node)

    return new_graph


class OnTheFlySubgraphSampler(object):
  """
    Randomly sample subgraph pairs 
  """
  def __init__(self,graphs, min_size, max_size):
    """
    """
    self.graphs = graphs
    self.graphs_dist = self.generate_graph_dist()
    self.min_subgraph_size = min_size
    self.max_subgraph_size = max_size

  def generate_graph_dist(self):
    """
      Generates a distribution according to which graphs are selected from list of graphs
      Here, graphs are selected proportional to the no. of nodes
    """
    ps = np.array([len(g) for g in self.graphs], dtype=np.float32)
    ps /= np.sum(ps)
    dist = scipy.stats.rv_discrete(values=(np.arange(len(self.graphs)), ps))
    return dist

  def sample_subgraph(self):
    """
      Returns a randomly selected (connected) list of nodes constituting a subgraph
      and the anchor node and graph_id (to id source graph from a list of graphs)
    """
    while True:
      #Select random graph from list of graphs
      graph_id = self.graphs_dist.rvs()
      graph = self.graphs[graph_id]
      #pick random anchor node
      anchor = random.randint(0,graph.number_of_nodes()-1)
      #pick size of subgraph to be generated
      size = random.randint(self.min_subgraph_size+1,self.max_subgraph_size)
      #init subgraph with anchor node
      subgraph = {anchor}
      #bfs_neigh is set of nodes we consider for adding to subgraph in future
      bfs_neigh = set(graph.neighbors(anchor)) - subgraph
      while len(subgraph)<size and bfs_neigh:
        curr_node = random.choice(list(bfs_neigh))
        subgraph.add(curr_node)
        bfs_neigh.remove(curr_node)
        bfs_neigh.update(set(graph.neighbors(curr_node))-subgraph)
      #if condition is not satisfied go through entire sungraph generation procedure again
      if len(subgraph)>self.min_subgraph_size:
        return graph.subgraph(list(subgraph)),anchor,graph_id
      
