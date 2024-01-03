import numpy
import pinecone
from pinecone.core.grpc import index_grpc
from pinecone.core.grpc.protos.vector_service_pb2 import QueryVector as GRPCQueryVector

from ..base.module import BaseANN

def metric_mapping(_metric: str):
    _metric_type = {"angular": "cosine", "euclidean": "l2"}.get(_metric, None)
    if _metric_type is None:
        raise Exception(f"[vdb] Not support metric type: {_metric}!!!")
    return _metric_type

class vdb(BaseANN):
    def __init__(self, metric, dim, method_param):
        self.name = "vdb (%s)" % (method_param)
        self._collection_name = "ann_benchmarks_vdb"
        self._metric = metric
        self._dim = dim
        self._metric_type = metric_mapping(self._metric)
        self._search_ef = None
        self.count = 0
        self.client = pinecone.GRPCIndex("myindex", None, index_grpc.GRPCClientConfig(secure=False), "127.0.0.1:20811")

    def fit(self, X):
        self.count += 1
        print("inserting", self.count)
        step = 10000
        for i in range(0, len(X), step):
            print(i)
            min = i
            max = i + step
            if max > len(X):
                max = len(X)
            self.client.upsert([(str(i), X[i]) for i in range(min, max, 1)], namespace="mynamespace")

    def set_query_arguments(self, ef):
        pass

    def query(self, v, n):
        resp = self.client.query(vector=v, top_k=n, namespace="mynamespace", include_values=False, include_metadata=False)
        return [int(x['id']) for x in resp['matches']]

    def batch_query(self, X: numpy.array, n: int) -> None:
        queries = [GRPCQueryVector(values=x, top_k=n, namespace="mynamespace") for x in X.tolist()]
        resp = self.client.query(queries=queries, top_k=n, include_values=False, include_metadata=False)
        self.res = numpy.asarray([[int(y['id']) for y in x['matches']] for x in resp['results']])
