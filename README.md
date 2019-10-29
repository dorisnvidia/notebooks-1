# RAPIDS Notebooks and Utilities

## XGBoost Notebook
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| XGBoost   | [XGBoost Demo](xgboost/XGBoost_Demo.ipynb)           | This notebook shows the acceleration one can gain by using GPUs with XGBoost in RAPIDS.                                                                                                                                       |
## CuML Notebooks
 The cuML notebooks showcase how to use the machine learning algorithms implemented in cuML along with the advantages of using cuML over scikit-learn. These notebooks compare the time required and the performance of the algorithms. Below are a list of such algorithms:

| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cuML      | [Coordinate Descent](cuml/coordinate_descent_demo.ipynb)     | This notebook includes code examples of lasso and elastic net models. These models are placed together so a comparison between the two can also be made in addition to their sklearn equivalent.                                                                                                                                                                |
| cuML      | [DBSCAN Demo](cuml/dbscan_demo.ipynb)            | This notebook showcases density-based spatial clustering of applications with noise (dbscan) algorithm using the `fit` and `predict` functions                                                                              |
| cuML      | [HoltWinters Demo](cuml/holtwinters_demo.ipynb)  | This notebook includes code example for the holt-winters algorithm and it showcases the `fit` and `forecast` functions.                 |
| cuML      | [Forest Inference](cuml/forest_inference_demo.ipynb)   | This notebook shows how to use the forest inference library to load saved models and perform prediction using them. In addition, it also shows how to perform training and prediction using xgboost and lightgbm models.|
| cuML      | [K-Means Demo](cuml/kmeans_demo.ipynb) | This notebook includes code example for the k-means algorithm and it showcases the `fit` and `predict` functions.                                                                                                                                             |
| cuML      | [K-Means MNMG Demo](cuml/kmeans_demo-mnmg.ipynb) | This notebook includes code example for the k-means multi-node multi-GPU algorithm and it showcases the `fit` and `predict` functions.                                                                                                                                             |
| cuML      | [Linear Regression Demo](cuml/linear_regression_demo.ipynb) | This notebook includes code example for linear regression algorithm and it showcases the `fit` and `predict` functions.                                                                                                                                             |
| cuML      | [Nearest Neighbors_demo](cuml/nearest_neighbors_demo.ipynb)               | This notebook showcases k-nearest neighbors (knn) algorithm using the `fit` and `kneighbors` functions                                                                                                                          |
| cuML      | [PCA Demo](cuml/pca_demo.ipynb)               | This notebook showcases principal component analysis (PCA) algorithm where the model can be used for prediction (using `fit_transform`) as well as converting the transformed data into the original dataset (using `inverse_transform`).                                                                                                                |
| cuML      | [Random Forest Multi-node / Multi-GPU](cuml/random_forest_mnmg_demo.ipynb)   | Demonstrates how to fit Random Forest models using multiple GPUs via Dask. |
| cuML      | [Ridge Regression Demo](cuml/ridge_regression_demo.ipynb)  | This notebook includes code examples of ridge regression and it showcases the `fit` and `predict` functions.                                                                                                                                          |
| cuML      | [SGD_Demo](cuml/sgd_demo.ipynb)               | The stochastic gradient descent algorithm is demostrated in the notebook using `fit` and `predict` functions                                                                        |
| cuML      | [TSNE_Demo](cuml/tsne_demo.ipynb)               | In this notebook, T-Distributed Stochastic Neighborhood Embedding is demonstrated applying the Barnes Hut method on the Fashion MNIST dataset using our `fit_transform` function                                                                      |
| cuML      | [TSVD_Demo](cuml/tsvd_demo.ipynb	)              | This notebook showcases truncated singular value decomposition (tsvd) algorithm which like PCA performs both prediction and transformation of the converted dataset into the original data using `fit_transform` and `inverse_transform` functions respectively                                                                                                     |
| cuML      | [UMAP_Demo](cuml/umap_demo.ipynb)              | The uniform manifold approximation & projection algorithm is compared with the original author's equivalent non-GPU Python implementation using `fit` and `transform` functions                       |
| cuML      | [UMAP_Demo_Graphed](cuml/umap_demo_graphed.ipynb)      | Demonstration of cuML uniform manifold approximation & projection algorithm's supervised approach against mortgage dataset and comparison of results against the original author's equivalent non-GPU \Python implementation. |
| cuML      | [UMAP_Demo_Supervised](cuml/umap_supervised_demo.ipynb)   | Demostration of UMAP supervised training.  Uses a set of labels to perform supervised dimensionality reduction. UMAP can also be trained on datasets with incomplete labels, by using a label of "-1" for unlabeled samples. |


## CuDF Notebooks
| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| cuDF      | [notebooks_Apply_Operations_in_cuDF](cudf/notebooks_Apply_Operations_in_cuDF.ipynb)            | This notebook showcases two special methods where cuDF goes beyond the Pandas library: apply_rows and apply_chunk functions. They utilized the Numba library to accelerate the data transformation via GPU in parallel.                                                                            |
| cuDF      | [notebooks_numba_cuDF_integration](cudf/notebooks_numba_cuDF_integration.ipynb)               | This notebook showcases how to use Numba CUDA to accelerate cuDF data transformation and how to step by step accelerate it using CUDA programming tricks                                                                                                                          |

## CuGraph Notebooks
| Folder  | Notebook Title                                               | Description                                                  |
| ------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| cuGraph | [Katz](cugraph/centrality/Katz.ipynb)                        | Compute the Katz centrality for every vertex                 |
| cuGraph | [Louvain](cugraph/community/Louvain.ipynb)                   | Demonstration of using cuGraph to identify clusters in a test graph using the Louvain algorithm |
| cuGraph | [Spectral-Clustering](cugraph/community/Spectral-Clustering.ipynb) | Demonstration of using cuGraph to identify clusters in a test graph using Spectral Clustering using both the (A) Balance Cut and (B) the Modularity Maximization quality metrics |
| cuGraph | [Subgraph Extraction](cugraph/community/Sungraph-Extraction.ipynb) | Compute a subgraph of the existing graph including only the specified vertices |
| cuGraph | [Triangle Counting](cugraph/community/Triangle-Counting.ipynb) | Demonstration of using both NetworkX and cuGraph  to compute the the number of Triangles in our test dataset |
| cuGraph | [Connected Components](cugraph/components/ConnectedComponents.ipynb) | Find weakly and strongly connected components in a graph     |
| cuGraph | [K-Core](cugraphcores/kcore.ipynb)                           | Extracts the K-core cluster                                  |
| cuGraph | [Core Number](cugraph/cores/core-number.ipynb)               | Computer the Core number for each vertex in a graph          |
|         | [Pagerank](cugraph/link_analysis/Pagerank.ipynb)             | Demonstration of using both NetworkX and cuGraph to compute the PageRank of each vertex in our test dataset |
| cuGraph | [Jacard Similarity](cugraph/link_prediction/Jaccard-Similarity.ipynb) | Compute vertex similarity score using both:<br />- Jaccard Similarity<br />- Weighted Jaccard |
| cuGraph | [Overlap Similarity](cugraph/link_prediction/Overlap-Similarity.ipynb) | Compute vertex similarity score using the Overlap Coefficient |
| cuGraph | [BFS](cugraph/traversal/BFS.ipynb)                           | Demonstration of using cuGraph to computer the Breadth First Search space from a given vertex to all other in our training graph |
| cuGraph | [SSSP](cugraph/traversal/SSSP.ipynb)                         | Demonstration of using cuGraph to computer the The Shortest Path from a given vertex to all other in our training graph |
| cuGraph | [Renumber](cugraph/structure/Renumber.ipynb)                 | Demonstrate of using the renumbering features to assigned new vertex IDs to the test graph.  This is useful for when the data sets is  non-contiguous or not integer values |
| cuGraph | [Symmetrize](cugraph/structure/Symmetrize.ipynb)             | Symmetrize the edges in a graph                              |
|         |                                                              |                                                              |

## Tutorial with an End to End workflow

| Folder    | Notebook Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Tutorials | [DBSCAN_demo_full](tutorials/DBSCAN_Demo_Full.ipynb)       | Demonstration of how to use DBSCAN - a popular clustering algorithm - and how to use the GPU accelerated implementation of this algorithm in RAPIDS.                                                                               |
| Tutorials | [HoltWinters_demo_full](tutorials/holtwinters_demo_full.ipynb)   | Demonstration of how to use Holt-Winters, a time-series forecasting algorithm, on a dataset to make GPU accelerated out-of-sample predictions.      |

## Utils Scripts
| Folder    | Script Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Utils     | start-jupyter.sh       | starts a JupyterLab environment for interacting with, and running, notebooks                                                                                                                                                  |
| Utils     | stop-jupyter.sh        | identifies all process IDs associated with Jupyter and kills them                                                                                                                                                             |
| Utils     | dask-cluster.py        | launches a configured Dask cluster (a set of nodes) for use within a notebook                                                                                                                                                 |
| Utils     | dask-setup.sh          | a low-level script for constructing a set of Dask workers on a single node                                                                                                                                                    |
| Utils     | split-data-mortgage.sh | splits mortgage data files into smaller parts, and saves them for use with the mortgage notebook                                                                                                                              |

## Documentation (WIP)
| Folder    | Document Title         | Description                                                                                                                                                                                                                   |
|-----------|------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Docs      | ngc-readme             |                                                                                                                                                                                                                               |
| Docs      | dockerhub-readme       |                                                                                                                                                                                                                               |

## Additional Information
* The `cuml` folder also includes a small subset of the Mortgage Dataset used in the notebooks and the full image set from the Fashion MNIST dataset.

* `utils`: contains a set of useful scripts for interacting with RAPIDS

* For additional, community driven notebooks, which will include our blogs, tutorials, workflows, and more intricate examples, please see the [Notebooks Extended Repo](https://github.com/rapidsai/notebooks-extended)
