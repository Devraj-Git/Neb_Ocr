"""
webhook_client.py — SSE-based remote OCR command listener.

Connects to a Django API server via Server-Sent Events (SSE),
keeps a persistent connection open, and processes OCR jobs as
they are pushed by the server in real-time.

Flow:
    1. Start → opens persistent SSE connection to server
    2. Server pushes OCR jobs as SSE events → app processes them
    3. App POSTs OCR results back to the result endpoint
    4. On connection drop → auto-reconnects after a brief delay

No polling. No periodic requests. Events are pushed in real-time.
"""

import base64
import json
import os
import threading
import time
from typing import Optional, Callable

import requests
from dotenv import load_dotenv

from .ollama_pipeline import process_image, save_to_database

load_dotenv()

API_BASE_URL = os.getenv("API_BASE_URL", "").rstrip("/")
API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN", "")
API_CLIENT_ID = os.getenv("API_CLIENT_ID", "")

# How long to wait before reconnecting after a connection drop
RECONNECT_DELAY = 3


class WebhookClient:
    """
    Opens a persistent SSE connection to the Django server and processes
    OCR jobs as they arrive.

    Operates in a daemon thread so it doesn't block app shutdown.

    Args:
        status_callback: Called with status string for UI updates.
        log_callback: Called with log messages for the console.
    """

    def __init__(
        self,
        status_callback: Optional[Callable[[str], None]] = None,
        log_callback: Optional[Callable[[str], None]] = None,
    ):
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._status_callback = status_callback
        self._log_callback = log_callback
        self._current_job_id: Optional[str] = None
        self._auth_headers: dict = {}
        self._sse_url: str = ""
        self._sse_response: Optional[requests.Response] = None

        # Validate configuration on init
        if not API_BASE_URL:
            self._log("⚠️ API_BASE_URL not set in .env — SSE client disabled.")
        if not API_AUTH_TOKEN:
            self._log("⚠️ API_AUTH_TOKEN not set in .env — SSE client disabled.")
        if not API_CLIENT_ID:
            self._log("⚠️ API_CLIENT_ID not set in .env — using 'default'.")
        if API_BASE_URL and API_AUTH_TOKEN:
            self._auth_headers = {
                "Authorization": f"Bearer {API_AUTH_TOKEN}",
                "Accept": "text/event-stream",
            }
            self._sse_url = (
                f"{API_BASE_URL}/api/ocr/events/?client_id="
                f"{API_CLIENT_ID or 'default'}"
            )

    # ── Properties ──────────────────────────────────────────────

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def current_job_id(self) -> Optional[str]:
        return self._current_job_id

    # ── Lifecycle ───────────────────────────────────────────────

    def start(self):
        """Start the SSE listener thread. Safe to call multiple times."""
        if self._running:
            return
        if not API_BASE_URL or not API_AUTH_TOKEN:
            self._set_status("DISABLED")
            return

        self._running = True
        self._thread = threading.Thread(target=self._sse_loop, daemon=True)
        self._thread.start()
        self._log(f"✅ SSE client started. Listening on {self._sse_url}")
        self._set_status("CONNECTING")

    def stop(self):
        """Gracefully stop the SSE listener thread."""
        self._running = False
        # Force-close the SSE connection to unblock iter_lines() immediately
        if self._sse_response:
            try:
                self._sse_response.close()
            except Exception:
                pass
        if self._thread:
            self._thread.join(timeout=10)
        self._log("⏹️ SSE client stopped.")
        self._set_status("DISCONNECTED")

    # ── Internal: SSE Streaming Loop ────────────────────────────

    def _sse_loop(self):
        """
        Main loop: open SSE connection, parse events, reconnect on drop.

        The loop runs indefinitely until stop() is called.
        On any connection error, it waits RECONNECT_DELAY seconds and retries.
        """
        while self._running:
            try:
                self._listen()
            except Exception as e:
                self._log(f"❌ SSE error: {e}")
                self._set_status("ERROR")
                # Brief pause before reconnecting
                if self._running:
                    time.sleep(RECONNECT_DELAY)
            else:
                # Connection closed cleanly (no exception) — likely server shutdown
                if self._running:
                    self._log("🔌 SSE connection closed. Reconnecting...")
                    self._set_status("DISCONNECTED")
                    time.sleep(RECONNECT_DELAY)

    def _listen(self):
        """
        Open a streaming GET request and parse SSE events.

        SSE format:
            data: <json payload>

            : keepalive

        - Lines starting with 'data: ' contain the event payload
        - Lines starting with ':' are comments (keepalive pings)
        - Empty lines delimit events
        """
        self._log("🌐 Opening SSE connection...")
        self._set_status("CONNECTING")

        resp = requests.get(
            self._sse_url,
            headers=self._auth_headers,
            stream=True,
            timeout=None,  # No timeout — connection stays open
        )
        resp.raise_for_status()

        self._log("✅ SSE stream connected.")
        self._set_status("CONNECTED")

        # Store response so stop() can force-close the connection
        self._sse_response = resp

        current_data = ""

        try:
            # Iterate over incoming lines as they arrive
            for raw_line in resp.iter_lines(decode_unicode=True):
                if not self._running:
                    break

                if raw_line is None:
                    continue

                line = raw_line.strip()

                if line.startswith("data: "):
                    current_data = line[6:]  # Strip "data: " prefix

                elif line == "" and current_data:
                    # Empty line = end of event — process the data
                    self._handle_event(current_data)
                    current_data = ""

                # Lines starting with ':' are SSE comments (keepalive pings) — ignore
        finally:
            self._sse_response = None

    # ── Event Handling ──────────────────────────────────────────

    def _handle_event(self, data_str: str):
        """
        Parse an SSE data payload and kick off OCR processing.

        Expected JSON format:
            {"job_id": "...", "image_base64": "...", "meta": {...}}
        """
        try:
            payload = json.loads(data_str)
        except json.JSONDecodeError:
            self._log(f"⚠️ Invalid SSE data (not JSON): {data_str[:80]}")
            return

        job_id = payload.get("job_id")
        if not job_id:
            return

        self._current_job_id = str(job_id)
        self._log(f"📥 Received OCR job via SSE: {job_id}")
        self._set_status(f"PROCESSING #{self._current_job_id[:8]}")

        # Process the job inline (blocks the SSE thread — no other jobs
        # can arrive during processing, which prevents job pile-up)
        self._process_job(payload)

    def _process_job(self, job_data: dict):
        """
        Decode base64 image, run OCR, and submit result.

        Args:
            job_data: Dict from the server containing at minimum
                      {'job_id': str, 'image_base64': str}
                      and optionally 'meta': dict
        """
        job_id = self._current_job_id
        image_base64 = job_data.get("image_base64", "")
        meta = job_data.get("meta", {})

        if not image_base64:
            self._submit_result(job_id, "error", error_message="No image data in job payload")
            return

        temp_path = None
        try:
            # ── Decode base64 image ──
            image_bytes = base64.b64decode(image_base64)

            # ── Save to temp file for VLM processing ──
            temp_dir = "temp"
            os.makedirs(temp_dir, exist_ok=True)
            temp_path = os.path.join(temp_dir, f"sse_{job_id}.png")

            with open(temp_path, "wb") as f:
                f.write(image_bytes)

            # ── Run OCR via VLM pipeline ──
            self._log(f"🔍 Running OCR on job {job_id}...")
            result = process_image(temp_path)

            # ── Save to local database (if meta provided) ──
            if meta and isinstance(meta, dict):
                try:
                    save_to_database(result, temp_path, meta=meta)
                    self._log(f"💾 Saved job {job_id} results to local DB.")
                except Exception as db_err:
                    self._log(f"⚠️ Local DB save warning: {db_err}")

            # ── Submit success result ──
            self._submit_result(job_id, "success", data=result)

        except (ValueError, OSError) as e:
            self._log(f"❌ Job {job_id} I/O error: {e}")
            self._submit_result(job_id, "error", error_message=str(e))
        except Exception as e:
            self._log(f"❌ Job {job_id} OCR error: {e}")
            self._submit_result(job_id, "error", error_message=str(e))
        finally:
            # Clean up temp file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    # ── Result Submission ───────────────────────────────────────

    def _submit_result(
        self,
        job_id: str,
        status: str,
        data: Optional[dict] = None,
        error_message: Optional[str] = None,
    ):
        """
        POST OCR result back to the Django server.

        Args:
            job_id: The original job ID.
            status: 'success' or 'error'.
            data: The full OCR result dict (for success).
            error_message: Error description (for errors).
        """
        payload = {
            "job_id": job_id,
            "client_id": API_CLIENT_ID or "default",
            "status": status,
            "data": data or {},
            "error_message": error_message,
        }

        # Need Content-Type for POST (not needed for SSE GET)
        post_headers = {**self._auth_headers, "Content-Type": "application/json"}
        # Don't send Accept: text/event-stream for a POST
        post_headers.pop("Accept", None)

        try:
            resp = requests.post(
                f"{API_BASE_URL}/api/ocr/submit-result/",
                headers=post_headers,
                json=payload,
                timeout=30,
            )
            resp.raise_for_status()
            self._log(f"📤 Job {job_id} result submitted ({status}).")
        except Exception as e:
            self._log(f"❌ Failed to submit result for {job_id}: {e}")

        # Finalize job state — SSE is still open, ready for next job
        self._current_job_id = None

    # ── Callback Helpers ────────────────────────────────────────

    def _log(self, msg: str):
        """Send a log message to the UI callback (timestamp added by main.py logger)."""
        if self._log_callback:
            self._log_callback(msg)

    def _set_status(self, status: str):
        """Update the connection status indicator via callback."""
        if self._status_callback:
            self._status_callback(status)
