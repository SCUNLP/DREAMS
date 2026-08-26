import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from src.web_gateway import ConversationSession, WebDemoApp, create_handler


class FakeAgent:
    def __init__(self):
        self.conversation_state = {"turn_count": 0, "actions_taken": []}
        self.seen_contexts = []

    def select_action(self, conv_dict):
        self.seen_contexts.append(list(conv_dict["context"]))
        self.conversation_state["turn_count"] += 1
        return "GenreInquiry"

    def execute_action(self, action, conv_dict):
        self.conversation_state["actions_taken"].append(action)
        return {"ans_type": "inquiry"}, f"reply to {conv_dict['context'][-1]}"


class WebGatewayTest(unittest.TestCase):
    def test_session_preserves_alternating_context(self):
        app = WebDemoApp(lambda trace: FakeAgent())
        session_id = app.create_session()["session_id"]
        first = app.chat(session_id, "comedy")
        second = app.chat(session_id, "with Tom Hanks")
        agent = app.sessions[session_id].agent
        self.assertEqual(first["action"], "GenreInquiry")
        self.assertEqual(second["reply"], "reply to with Tom Hanks")
        self.assertEqual(
            agent.seen_contexts,
            [["comedy"], ["comedy", "reply to comedy", "with Tom Hanks"]],
        )
        self.assertEqual(second["trace"]["state"]["turn_count"], 2)

    def test_http_session_and_chat_endpoints(self):
        app = WebDemoApp(lambda trace: FakeAgent())
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app, b"<h1>DREAMS</h1>"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_address[1]}"
        try:
            session = self._post(base + "/api/session", {})
            response = self._post(base + "/api/chat", {
                "session_id": session["session_id"], "message": "science fiction",
            })
            self.assertEqual(response["reply"], "reply to science fiction")
            self.assertEqual(response["action"], "GenreInquiry")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_unknown_session_returns_404(self):
        app = WebDemoApp(lambda trace: FakeAgent())
        server = ThreadingHTTPServer(("127.0.0.1", 0), create_handler(app, b"demo"))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with self.assertRaises(HTTPError) as caught:
                self._post(f"http://127.0.0.1:{server.server_address[1]}/api/chat", {
                    "session_id": "missing", "message": "hello",
                })
            self.assertEqual(caught.exception.code, 404)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_message_validation(self):
        app = WebDemoApp(lambda trace: FakeAgent())
        session_id = app.create_session()["session_id"]
        with self.assertRaisesRegex(ValueError, "must be a string"):
            app.chat(session_id, None)
        with self.assertRaisesRegex(ValueError, "cannot be empty"):
            app.chat(session_id, "   ")

    @staticmethod
    def _post(url, payload):
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=3) as response:
            return json.loads(response.read().decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
