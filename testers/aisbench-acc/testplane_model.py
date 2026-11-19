"""AISBench model adapter: environment-only auth and one audit file per request."""
import json
import os
from pathlib import Path

from ais_bench.benchmark.models import VLLMCustomAPIChat

from runtime import redact, tokenizer_info


class Chat(VLLMCustomAPIChat):
    def __init__(self, **kwargs):
        # No secret in any Config object, including AISBench's expanded configs.
        kwargs["api_key"] = os.environ.get("ENDPOINT_API_KEY", "")
        self._tp_url = kwargs["url"].rstrip("/") + "/chat/completions"
        super().__init__(**kwargs)
        self._tp_tokenizer = None

    def _get_url(self):
        return self._tp_url

    async def get_request_body(self, input, max_out_len, output, **args):
        body = await super().get_request_body(input, max_out_len, output, **args)
        output._tp_request = body
        return body

    async def generate(self, input, max_out_len, output, **kwargs):
        try:
            return await super().generate(input, max_out_len, output, **kwargs)
        finally:
            body = getattr(output, "_tp_request", {})
            if hasattr(output, "_tp_request"):
                del output._tp_request  # don't add a second large payload to upstream DBs
            prediction = output.get_prediction()
            source = "usage"
            if output.success and (output.input_tokens <= 0 or output.output_tokens <= 0):
                if self._tp_tokenizer is None:
                    self._tp_tokenizer, _ = tokenizer_info(Path(self.path))
                if output.input_tokens <= 0:
                    template_kwargs = body.get("chat_template_kwargs", {})
                    output.input_tokens = len(self._tp_tokenizer.apply_chat_template(body["messages"], tokenize=True, add_generation_prompt=True, **template_kwargs))
                if output.output_tokens <= 0:
                    output.output_tokens = len(self._tp_tokenizer.encode(prediction, add_special_tokens=False))
                source = "usage_or_local_tokenizer"
            record = dict(id=output.data_id, request=body, success=bool(output.success),
                          prediction=prediction, error=output.error_info,
                          input_tokens=output.input_tokens, output_tokens=output.output_tokens,
                          token_source=source, time_points=list(output.time_points))
            destination = Path(os.environ["TESTPLANE_AUDIT_DIR"]) / f"{output.data_id:09d}.json"
            with destination.open("x", encoding="utf-8") as handle:
                handle.write(redact(json.dumps(record, ensure_ascii=False, allow_nan=False)))
