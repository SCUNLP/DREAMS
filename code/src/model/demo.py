"""Self-contained live HTML trace for MCTS demo runs."""

import copy
import html
import json
import threading
import time
import webbrowser
from pathlib import Path


class DemoTrace:
    def __init__(self, output_path, auto_open=True):
        self.output_path = Path(output_path).resolve() if output_path else None
        self.auto_open = auto_open
        self.events = []
        self._lock = threading.Lock()
        self._opened = False
        self._last_tree = None
        self._last_state = {}

    def record(self, event, **details):
        with self._lock:
            self.events.append({
                "step": len(self.events) + 1,
                "time": time.strftime("%H:%M:%S"),
                "event": event,
                **details,
            })
            self.events = self.events[-2000:]

    def reset(self, **details):
        with self._lock:
            self.events = []
            self._last_tree = None
            self._last_state = {}
        self.record("agent_initialized", **details)

    def render(self, root=None, state=None):
        with self._lock:
            if root is not None:
                self._last_tree = self._tree_data(root)
            if state is not None:
                self._last_state = copy.deepcopy(state)
            payload = self._snapshot_unlocked()
        if self.output_path is None:
            return payload
        document = self._document(payload)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(".tmp")
        temporary_path.write_text(document, encoding="utf-8")
        temporary_path.replace(self.output_path)
        if self.auto_open and not self._opened:
            webbrowser.open(self.output_path.as_uri())
            self._opened = True
        return payload

    def snapshot(self):
        """Return a detached, JSON-compatible view for the web demo API."""
        with self._lock:
            return self._snapshot_unlocked()

    def _snapshot_unlocked(self):
        return copy.deepcopy({
            "events": self.events,
            "state": self._last_state,
            "tree": self._last_tree,
        })

    def _tree_data(self, node):
        return {
            "action": node.action or "ROOT",
            "visits": node.visits,
            "value": round(node.value, 4),
            "average": round(node.value / node.visits, 4) if node.visits else 0,
            "children": [self._tree_data(child) for child in node.children.values()],
        }

    @staticmethod
    def _node_markup(node):
        children = "".join(DemoTrace._node_markup(child) for child in node["children"])
        branch = f"<ul>{children}</ul>" if children else ""
        return (
            "<li><div class='node'>"
            f"<strong>{html.escape(str(node['action']))}</strong>"
            f"<span>visits {node['visits']}</span>"
            f"<span>avg {node['average']}</span>"
            f"<span>value {node['value']}</span>"
            f"</div>{branch}</li>"
        )

    @staticmethod
    def _detail_text(event):
        details = {key: value for key, value in event.items() if key not in {"step", "time", "event"}}
        text = json.dumps(details, ensure_ascii=False, default=str)
        return html.escape(text[:1200])

    @classmethod
    def _document(cls, payload):
        tree = payload["tree"]
        tree_markup = f"<ul class='tree'>{cls._node_markup(tree)}</ul>" if tree else "<p class='empty'>Fast inquiry phase: no search tree is created.</p>"
        rows = "".join(
            "<tr>"
            f"<td>{event['step']}</td>"
            f"<td>{html.escape(event['time'])}</td>"
            f"<td>{html.escape(event['event'])}</td>"
            f"<td><code>{cls._detail_text(event)}</code></td>"
            "</tr>"
            for event in reversed(payload["events"][-100:])
        )
        strategy_rows = "".join(
            "<div class='strategy'>"
            f"<strong>{html.escape(str(event.get('strategy', 'unknown')))}</strong>"
            f"<span>reward {float(event.get('reward', 0)):.4f}</span>"
            f"<code>{html.escape(str(event.get('query', ''))[:240])}</code>"
            "</div>"
            for event in payload["events"]
            if event["event"] == "retrieval_reward"
        ) or "<p class='empty'>No retrieval search yet.</p>"
        state = html.escape(json.dumps(payload["state"], ensure_ascii=False, default=str, indent=2))
        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="2">
<title>DREAMS MCTS Demo</title>
<style>
:root {{ color-scheme: light dark; font-family: ui-sans-serif, system-ui, sans-serif; }}
body {{ margin: 0; padding: 20px; background: #0b1020; color: #e8edf7; }}
h1, h2 {{ margin: 0 0 12px; }}
.layout {{ display: grid; grid-template-columns: minmax(0, 2fr) minmax(280px, 1fr); gap: 18px; }}
.panel {{ border: 1px solid #34415e; background: #121a2d; padding: 16px; overflow: auto; }}
.tree, .tree ul {{ display: flex; justify-content: center; gap: 14px; padding: 18px 0 0; position: relative; }}
.tree ul::before {{ content: ''; position: absolute; top: 0; left: 50%; height: 18px; border-left: 1px solid #607095; }}
.tree li {{ list-style: none; text-align: center; position: relative; min-width: 145px; }}
.tree ul > li::before, .tree ul > li::after {{ content: ''; position: absolute; top: -18px; width: 50%; height: 18px; border-top: 1px solid #607095; }}
.tree ul > li::before {{ right: 50%; }} .tree ul > li::after {{ left: 50%; border-left: 1px solid #607095; }}
.tree ul > li:only-child::before, .tree ul > li:only-child::after {{ display: none; }}
.tree ul > li:first-child::before, .tree ul > li:last-child::after {{ border-top: 0; }}
.node {{ display: grid; gap: 4px; border: 1px solid #607095; background: #19243d; padding: 9px; }}
.node strong {{ color: #82d9ff; }} .node span {{ font-size: 12px; color: #b8c4dc; }}
.strategy {{ display: grid; grid-template-columns: 150px 110px 1fr; gap: 10px; padding: 8px 0; border-bottom: 1px solid #34415e; }}
pre, code {{ white-space: pre-wrap; overflow-wrap: anywhere; font-size: 12px; }}
table {{ width: 100%; border-collapse: collapse; }} th, td {{ border-bottom: 1px solid #34415e; padding: 8px; text-align: left; vertical-align: top; }}
.empty {{ color: #b8c4dc; }}
@media (max-width: 900px) {{ .layout {{ grid-template-columns: 1fr; }} .tree, .tree ul {{ justify-content: flex-start; }} }}
</style>
</head>
<body>
<h1>DREAMS MCTS Demo</h1>
<div class="layout">
  <section class="panel"><h2>Search tree</h2>{tree_markup}</section>
  <section class="panel"><h2>Conversation state</h2><pre>{state}</pre></section>
</div>
<section class="panel" style="margin-top:18px"><h2>Retrieval MCTS strategies</h2>{strategy_rows}</section>
<section class="panel" style="margin-top:18px"><h2>Reward and feedback timeline</h2>
<table><thead><tr><th>#</th><th>Time</th><th>Event</th><th>Details</th></tr></thead><tbody>{rows}</tbody></table>
</section>
</body>
</html>"""
