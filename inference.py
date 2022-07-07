import get_layer
import dgl
from dgl.nn import GraphConv
import score_model

graphls, names = get_layer.out(config='inference')

#batched_graph = dgl.batch(graphls)
#print(batched_graph)
