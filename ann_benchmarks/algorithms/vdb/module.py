import numpy
import pinecone
from pinecone.core.grpc import index_grpc

from ..base.module import BaseANN


def metric_mapping(_metric: str):
    _metric_type = {"angular": "cosine", "euclidean": "l2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[vdb] Not support metric type: {_metric}!!!")
    return _metric_type


class VDB(BaseANN):

    def __init__(self, metric, dim, index_param):
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self._search_ef = None
        self.client = None

    def fit(self, X):
        self.client = pinecone.GRPCIndex("myindex", None, index_grpc.GRPCClientConfig(secure=False), "127.0.0.1:20811")
        self.client.upsert((X, numpy.arange(len(X)), [{"key": "value"} for _ in range(len(X))]), namespace="mynamespace")

    def query(self, v, n):
        return self.client.query(vector=v, top_k=n)
