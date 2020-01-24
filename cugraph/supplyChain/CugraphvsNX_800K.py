import cudf
import cugraph
import numpy as np
from collections import OrderedDict
import time

df = cudf.DataFrame()

# read a data file - this works for both Nodes and Edges
#import pandas as pd
def read_data(f, col, dt) :
    return cudf.read_csv(f, 
                         names=col, 
                         dtype=dt, 
                         delimiter=',', 
                         comment='#',
                         skiprows=1,
                         skip_blank_lines=True,
                         skipinitialspace=True
                        )

files = [
    '../data/800K_after_concat.csv',
    '../data/400K_Input.csv',
#    '/content/drive/My Drive/colab data/25K_after_concat.csv'
]
fileid=1

start_time = time.time()
datafile = files[fileid]
raw_data = cudf.read_csv(datafile, delimiter=",", names=['node_1', 'node_2'], skiprows=1)
raw_data.rename(columns={'node_1':'src_str', 'node_2':'dst_str'}, inplace=True)
raw_data = raw_data.drop_duplicates()
print ("Time taken to read csv and drop duplicates using RAPIDS : " + str(time.time() - start_time) + " seconds")


hash_time = time.time()
new_series=cudf.concat([raw_data['src_str'],raw_data['dst_str']])
temp = new_series.hash_values()
raw_data['src_hash'] = temp[:len(raw_data)]
raw_data['dst_hash'] = temp[len(raw_data):]

# Renumber the hash values to a smaller contiguous range 
raw_data['src'], raw_data['dst'], N = cugraph.renumber(raw_data['src_hash'], raw_data['dst_hash'])
print ("Time taken to hashing using RAPIDS " + str(time.time() - hash_time) + " seconds")


### Now build the Graph
graphtime = time.time()
G = cugraph.Graph() # Graph() is undirected graph
G.from_cudf_edgelist(raw_data, source='src',target='dst')

# get a list of vertices
wcc = cugraph.weakly_connected_components(G)
print ("Time taken to partition the skus using RAPIDS: " + str(time.time() - graphtime) + " seconds")

extract_time = time.time()
wcc['vertex'] = wcc.index

src_side = cudf.DataFrame()
src_side['node']   = raw_data['src_str']
src_side['vertex'] = raw_data['src'].astype(np.int64)

dst_side = cudf.DataFrame()
dst_side['node']   = raw_data['dst_str']
dst_side['vertex'] = raw_data['dst'].astype(np.int64)

answer = cudf.DataFrame()
answer = cudf.concat([src_side, dst_side])
answer = answer.drop_duplicates()
del src_side
del dst_side

a = wcc.merge(answer, on='vertex', how='left')
a = a.drop_duplicates()
a = a.sort_values(by='labels', ascending=True)

print ("Time taken to extract connected skus using RAPIDS:" + str(time.time() - extract_time) + " seconds\n")
print ("**** Total Time taken from import csv to extract connected skus using RAPIDS :" + str(time.time() - start_time) + " seconds\n")

print ("detected " + str( len(wcc['labels'].unique()) ) + " components using RAPIDS")
print ("**********************************************\n")

####################################################
# NetworkX

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

start_time = time.time()
datafile = files[fileid]
df = pd.read_csv(datafile, delimiter=",", names=['node_1', 'node_2'], skiprows=1)
df.rename(columns={'node_1':'src', 'node_2':'dst'}, inplace=True)
df.drop_duplicates()
msg = "Time taken to read csv and drop duplicates using CPU: " + str(time.time() - start_time) + " seconds"
print(msg)

graph_time = time.time()
cpuG=nx.from_pandas_edgelist(df, source='src', target='dst',create_using=nx.DiGraph)
msg = "Time taken to partition the skus using CPU: " + str(time.time() - graph_time) + " seconds"
print(msg)

extract_time = time.time()
connectedskus = sorted(nx.weakly_connected_components(cpuG), key=len, reverse=True)
nodeslist = []
batchnum = 0
for consku in connectedskus:
  batchnum = batchnum + 1
  for val in consku:
          splitsku = val.split("@@")
          skuname = splitsku[0]
          locname = splitsku[1]
          t = (skuname, locname, batchnum)
          nodeslist.append(t)

msg = "Time taken to extract connected skus using CPU: " + str(time.time() - extract_time) + " seconds\n"
print(msg)

print ("**** Total Time taken from import csv to extract connected skus using CPU :" + str(time.time() - start_time) + " seconds\
")

print ("detected " + str(len(connectedskus)) + " components using NetworkX")
