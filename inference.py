import get_layer
import dgl
from dgl.nn import GraphConv
import models.score_model as score_model

graphls, names = get_layer.out(config='inference')

#batched_graph = dgl.batch(graphls)
#print(batched_graph)
