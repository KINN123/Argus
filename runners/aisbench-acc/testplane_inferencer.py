"""Carry the dataset index through AISBench's async request call."""
from ais_bench.benchmark.openicl.icl_inferencer import GenInferencer as AISGenInferencer

from testplane_model import REQUEST_INDEX


class GenInferencer(AISGenInferencer):
    async def do_request(self, data, token_bucket, session):
        token = REQUEST_INDEX.set(data["index"])
        try:
            return await super().do_request(data, token_bucket, session)
        finally:
            REQUEST_INDEX.reset(token)
