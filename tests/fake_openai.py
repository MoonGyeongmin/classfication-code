"""테스트용 가짜 OpenAI 호환 서버(LiteLLM 대용). 번역은 '[lang] 원문'으로, 임베딩은 글자 빈도 벡터로 돌려준다."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _embed(text, dim=64):
    v = [0.0] * dim
    for ch in text.lower():
        v[ord(ch) % dim] += 1.0
    return v


class Handler(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Handler.calls.append((self.path, self.headers.get("Authorization"), body))
        if self.headers.get("Authorization") != "Bearer sk-test":
            return self._send(401, {"error": {"message": "bad key"}})
        if self.path.endswith("/chat/completions"):
            sys_msg, user = body["messages"][0]["content"], body["messages"][-1]["content"]
            lang = sys_msg.split("into ")[1].split(" the way")[0]
            out = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                   "choices": [{"index": 0, "finish_reason": "stop",
                                "message": {"role": "assistant", "content": f"[{lang}] {user}"}}],
                   "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}
        elif self.path.endswith("/embeddings"):
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            out = {"object": "list", "model": body["model"],
                   "data": [{"object": "embedding", "index": i, "embedding": _embed(t)} for i, t in enumerate(inputs)],
                   "usage": {"prompt_tokens": 1, "total_tokens": 1}}
        else:
            return self._send(404, {})
        self._send(200, out)

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def start():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/v1"
