"""Local HTTP gateway for interactive DREAMS conversations."""

import argparse
import json
import os
import secrets
import threading
import webbrowser
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .model.demo import DemoTrace


MAX_REQUEST_BYTES = 16 * 1024
MAX_MESSAGE_CHARS = 4000


class ConversationSession:
    """Own one mutable CRS agent and serialize its conversation turns."""

    def __init__(self, agent, trace):
        self.agent = agent
        self.trace = trace
        self.context = []
        self._lock = threading.Lock()

    def chat(self, message):
        if not isinstance(message, str):
            raise ValueError("Message must be a string.")
        message = message.strip()
        if not message:
            raise ValueError("Message cannot be empty.")
        if len(message) > MAX_MESSAGE_CHARS:
            raise ValueError(f"Message cannot exceed {MAX_MESSAGE_CHARS} characters.")

        with self._lock:
            next_context = [*self.context, message]
            conv_dict = {"context": next_context}
            action = self.agent.select_action(conv_dict)
            result = self.agent.execute_action(action, conv_dict)
            reply, meta = self._unpack_result(result)
            self.context = [*next_context, reply]

            self.trace.record(
                "real_turn",
                action=action,
                user=message,
                assistant=reply,
                meta=meta,
            )
            trace = self.trace.render(state=self.agent.conversation_state)
            return {
                "reply": reply,
                "action": action,
                "state": trace["state"],
                "trace": trace,
            }

    @staticmethod
    def _unpack_result(result):
        if not isinstance(result, tuple):
            raise RuntimeError("CRS action returned an invalid result.")
        if len(result) == 2:
            meta, reply = result
        elif len(result) == 4:
            candidates, meta, reply, refined_query = result
            try:
                candidate_count = len(candidates[0])
            except (IndexError, TypeError):
                candidate_count = 0
            meta = {
                "details": meta,
                "candidate_count": candidate_count,
                "refined_query": refined_query,
            }
        else:
            raise RuntimeError(f"CRS action returned {len(result)} values; expected 2 or 4.")
        if not isinstance(reply, str) or not reply.strip():
            raise RuntimeError("CRS action returned an empty reply.")
        return reply.strip(), meta


class WebDemoApp:
    """In-memory session registry shared by HTTP request handlers."""

    def __init__(self, agent_factory, max_sessions=32):
        self.agent_factory = agent_factory
        self.max_sessions = max_sessions
        self.sessions = OrderedDict()
        self._lock = threading.Lock()

    def create_session(self):
        trace = DemoTrace(None, auto_open=False)
        agent = self.agent_factory(trace)
        session_id = secrets.token_urlsafe(24)
        with self._lock:
            self.sessions[session_id] = ConversationSession(agent, trace)
            self.sessions.move_to_end(session_id)
            while len(self.sessions) > self.max_sessions:
                self.sessions.popitem(last=False)
        return {"session_id": session_id, "trace": trace.snapshot()}

    def chat(self, session_id, message):
        if not isinstance(session_id, str):
            raise ValueError("A valid session_id is required.")
        with self._lock:
            session = self.sessions.get(session_id)
            if session is not None:
                self.sessions.move_to_end(session_id)
        if session is None:
            raise KeyError("Session not found. Start a new conversation.")
        return session.chat(message)


def create_handler(app, page):
    class DemoHandler(BaseHTTPRequestHandler):
        server_version = "DREAMS/1.0"

        def do_GET(self):
            if self.path == "/":
                self._send_bytes(200, page, "text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self._send_bytes(204, b"", "image/x-icon")
            else:
                self._send_json(404, {"error": "Not found."})

        def do_POST(self):
            try:
                payload = self._read_json()
                if self.path == "/api/session":
                    response = app.create_session()
                elif self.path == "/api/chat":
                    response = app.chat(payload.get("session_id"), payload.get("message", ""))
                else:
                    self._send_json(404, {"error": "Not found."})
                    return
                self._send_json(200, response)
            except (ValueError, json.JSONDecodeError) as error:
                self._send_json(400, {"error": str(error)})
            except KeyError as error:
                self._send_json(404, {"error": error.args[0]})
            except Exception as error:
                self._send_json(500, {"error": str(error)})

        def _read_json(self):
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError as error:
                raise ValueError("Invalid Content-Length.") from error
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError(f"Request body must be 1-{MAX_REQUEST_BYTES} bytes.")
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            if not isinstance(payload, dict):
                raise ValueError("JSON body must be an object.")
            return payload

        def _send_json(self, status, payload):
            body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
            self._send_bytes(status, body, "application/json; charset=utf-8")

        def _send_bytes(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline'; connect-src 'self'")
            self.end_headers()
            if body:
                self.wfile.write(body)

        def log_message(self, message_format, *args):
            print(f"[web] {self.address_string()} - {message_format % args}")

    return DemoHandler


def create_parser():
    parser = argparse.ArgumentParser(description="Start the DREAMS interactive web demo")
    parser.add_argument("--dataset", choices=["redial", "opendialkg"], default="redial")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="HTTP port (default: 8000)")
    parser.add_argument("--mcts_iterations", type=int, default=3)
    parser.add_argument("--simulation_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_browser", action="store_true", help="Do not open the browser automatically")
    parser.add_argument("--api_key", help="Deprecated; prefer OPENAI_API_KEY")
    return parser


def run(args, repo_root):
    if args.api_key:
        os.environ["OPENAI_API_KEY"] = args.api_key
    if not os.getenv("OPENAI_API_KEY", "").strip() and not os.getenv("DREAMS_OPENAI_API_KEYS", "").strip():
        raise RuntimeError("Set OPENAI_API_KEY before starting the web demo.")
    if not 0 <= args.port <= 65535:
        raise ValueError("Port must be between 0 and 65535.")

    repo_root = Path(repo_root)
    embedding_path = repo_root / "save" / "embed" / "item" / args.dataset
    if not embedding_path.is_dir():
        raise FileNotFoundError(
            f"Missing item embeddings at {embedding_path}. "
            f"Run: python script/cache_item.py --dataset {args.dataset}"
        )

    # Keep --help and the gateway/session unit tests dependency-free.
    from .model.chatgpt_mcts_dual import CHATGPT

    page = (Path(__file__).with_name("web_demo.html")).read_bytes()

    def agent_factory(trace):
        return CHATGPT(
            seed=args.seed,
            debug=args.debug,
            kg_dataset=args.dataset,
            mcts_iterations=args.mcts_iterations,
            simulation_workers=args.simulation_workers,
            trace=trace,
            repo_root=repo_root,
        )

    app = WebDemoApp(agent_factory)
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app, page))
    actual_port = server.server_address[1]
    browser_host = "127.0.0.1" if args.host in {"0.0.0.0", "::"} else args.host
    url = f"http://{browser_host}:{actual_port}"
    print(f"DREAMS web demo: {url}")
    if args.host not in {"127.0.0.1", "localhost", "::1"}:
        print("Warning: this demo has no authentication; use 127.0.0.1 unless remote access is intentional.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping DREAMS web demo.")
    finally:
        server.server_close()
