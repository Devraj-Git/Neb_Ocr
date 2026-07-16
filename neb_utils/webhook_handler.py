"""
Webhook / remote-OCR connection management – extracted from main.py.

``WebhookHandler`` owns the SSE client lifecycle and refreshes the
sidebar status indicator when the connection state changes.
"""

from neb_utils.webhook_client import WebhookClient
from neb_utils.icons import GLOBE_ICON


class WebhookHandler:
    """Manages the webhook/SSE client and its UI indicators."""

    def __init__(self, app):
        self.app = app
        self.client: WebhookClient = None
        self._status = "DISABLED"

    # ── Lifecycle ───────────────────────────────────────────────

    def init_client(self):
        """Create and start the webhook polling client."""
        self.client = WebhookClient(
            status_callback=self._on_status,
            log_callback=self._on_log,
        )
        self.client.start()

    # ── Callbacks (called from WebhookClient's threads) ─────────

    def _on_status(self, status: str):
        """Thread-safe status update → queue UI refresh."""
        self._status = status
        self.app.after(0, self._refresh_ui)

    def _on_log(self, msg: str):
        """Thread-safe log forwarding."""
        self.app.after(0, lambda m=msg: self.app.log_message(m))

    # ── UI refresh ──────────────────────────────────────────────

    def _refresh_ui(self):
        """Update the sidebar status pill and info labels."""
        C = self.app.COLORS
        s = self._status

        if s == "CONNECTED":
            text = "REMOTE: ● ONLINE"
            color = C["success"]
            bg = "#14532D"
        elif s == "DISCONNECTED":
            text = "REMOTE: ○ OFFLINE"
            color = C["danger"]
            bg = "#7F1D1D"
        elif s == "DISABLED":
            text = "REMOTE: ○ DISABLED"
            color = C["text_dim"]
            bg = C["sidebar"]
        elif s == "AUTH_ERROR":
            text = "REMOTE: ● AUTH FAIL"
            color = C["danger"]
            bg = "#7F1D1D"
        elif s == "ERROR":
            text = "REMOTE: ● ERROR"
            color = C["danger"]
            bg = "#7F1D1D"
        elif s.startswith("PROCESSING"):
            text = f"REMOTE: ⏳ {s}"
            color = C["accent"]
            bg = "#0C4A6E"
        else:
            text = f"REMOTE: {s}"
            color = C["text_dim"]
            bg = C["sidebar"]

        self.app.webhook_status_lbl.configure(
            text=text, text_color=color, fg_color=bg,
            image=GLOBE_ICON, compound="left",
        )

        # Info frame labels
        if hasattr(self.app, "webhook_url_lbl"):
            from neb_utils.webhook_client import API_BASE_URL, API_CLIENT_ID

            self.app.webhook_url_lbl.configure(
                text=f"Server: {API_BASE_URL or 'not set'}"
            )
            self.app.webhook_client_id_lbl.configure(
                text=f"Client: {API_CLIENT_ID or 'default'}"
            )
