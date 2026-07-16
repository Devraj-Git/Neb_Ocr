# SSE (Server-Sent Events) OCR — Django API Contract

## Overview

The NEB OCR desktop app connects to your Django server via **Server-Sent Events (SSE)** — a persistent HTTP connection where the server pushes OCR jobs in real-time.

**No polling on either side:**
- **App side:** Never polls. Just keeps the SSE connection open.
- **Server side:** Flushes pending jobs on connect, then uses Django **signals** to push new jobs instantly.

**Flow:**
1. OCR app connects → `GET /api/ocr/events/`
2. Server sends **1 pending job** (only one — not all)
3. App processes (may take 10-15 min)
4. App POSTs result back → `POST /api/ocr/submit-result/`
5. Server saves result, then immediately checks for the **next** pending job → sends it
6. Loop continues one job at a time until queue is empty
7. New jobs created via Django admin/signal → pushed one at a time through the same loop

---

## 1. SSE Event Stream — Receive OCR Jobs in Real-Time

**Endpoint:** `GET /api/ocr/events/?client_id=xxx`

**Query Parameters:**

| Parameter   | Type   | Required | Description                                   |
|-------------|--------|----------|-----------------------------------------------|
| `client_id` | string | Yes      | Unique ID identifying this OCR client instance |

**Headers:**

| Header          | Value                        |
|-----------------|------------------------------|
| `Authorization` | `Bearer <API_AUTH_TOKEN>`    |
| `Accept`        | `text/event-stream`          |

**Response:** `Content-Type: text/event-stream` (SSE stream kept open)

**Events format:**

```
data: {"job_id": "abc123-def456", "image_base64": "iVBORw0KGgo...", "meta": {...}}

: keepalive

```

- **`data:`** — Lines starting with `data:` contain the JSON payload of an OCR job.
- **`:`** — Lines starting with `:` are SSE comments (keepalive pings every 30s).
- **Empty line** — Marks the end of an event.

**Job payload JSON:**

```json
{
  "job_id": "abc123-def456",
  "image_base64": "iVBORw0KGgoAAAANSUhEUgAA...",
  "meta": {
    "grade": "12",
    "exam_type": "Regular",
    "exam_year": "2080"
  }
}
```

---

## 2. Submit Result Endpoint — Post OCR Results Back

**Endpoint:** `POST /api/ocr/submit-result/`

**Headers:**

| Header          | Value                        |
|-----------------|------------------------------|
| `Authorization` | `Bearer <API_AUTH_TOKEN>`    |
| `Content-Type`  | `application/json`           |

**Request Body (success):**

```json
{
  "job_id": "abc123-def456",
  "client_id": "neb-ocr-client-1",
  "status": "success",
  "data": {
    "school_code": "2738",
    "school_name": "Example School",
    "page_number": "77",
    "examination_year": "2080",
    "grade": "Twelve",
    "exame_Type": "Regular",
    "students": [
      {
        "symbol_number": "12345678",
        "registration_number": "7890123456",
        "student_name": "Student Name",
        "date_of_birth": "2062-03-15",
        "subjects": [
          {
            "subject_name": "C.ENG",
            "theory": "75",
            "practical": null,
            "total": "75",
            "extra": false
          }
        ],
        "grand_total": "75",
        "remark": "PASS"
      }
    ],
    "handwritten_marginal_notes": []
  },
  "error_message": null
}
```

**Request Body (error):**

```json
{
  "job_id": "abc123-def456",
  "client_id": "neb-ocr-client-1",
  "status": "error",
  "data": {},
  "error_message": "Ollama VLM failed to process image: timeout"
}
```

**Success Response — HTTP 200:**

```json
{
  "status": "accepted",
  "job_id": "abc123-def456"
}
```

---

## 3. Django Implementation Guide (Signal-Based, No Polling)

### Django Model

```python
import uuid
from datetime import timedelta
from django.db import models
from django.utils import timezone

class OCROutboxJob(models.Model):
    """An OCR job waiting to be picked up by a desktop client."""

    STATUS_CHOICES = [
        ("pending", "Pending"),
        ("processing", "Processing"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ]

    job_id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    client_id = models.CharField(max_length=255, blank=True, null=True)
    image_base64 = models.TextField(help_text="Base64-encoded image data")
    meta = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="pending")
    result_data = models.JSONField(null=True, blank=True)
    error_message = models.TextField(blank=True, null=True)
    assigned_at = models.DateTimeField(null=True, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["created_at"]  # Oldest first

    @classmethod
    def release_stuck_jobs(cls, client_id: str, timeout_minutes: int = 15):
        """
        Find jobs stuck in 'processing' for longer than timeout_minutes
        and reset them to 'pending' so they get re-delivered.

        This handles the case where the OCR app crashes mid-processing.
        """
        cutoff = timezone.now() - timedelta(minutes=timeout_minutes)
        stuck = cls.objects.filter(
            status="processing",
            client_id=client_id,
            assigned_at__lt=cutoff,
        )
        count = stuck.update(status="pending", assigned_at=None)
        return count
```

### Signal-Based Event System

This is the core of the no-polling approach. We use Django's `post_save` signal with an in-memory Queue to bridge signal events into the SSE stream.

```python
from queue import Queue, Empty
import threading

# ── Per-client event queues ──
# Maps client_id → Queue. When a new job is created via Django admin/API,
# the signal handler puts it into the queue. The SSE view reads from it.
_client_queues: dict[str, Queue] = {}
_queue_lock = threading.Lock()


def _register_client(client_id: str) -> Queue:
    """Create or get the event queue for a client."""
    with _queue_lock:
        if client_id not in _client_queues:
            _client_queues[client_id] = Queue()
        return _client_queues[client_id]


def _unregister_client(client_id: str):
    """Remove the queue when client disconnects."""
    with _queue_lock:
        _client_queues.pop(client_id, None)


def _serialize_job(job) -> str:
    """Convert a job to an SSE data line."""
    import json
    payload = {
        "job_id": str(job.job_id),
        "image_base64": job.image_base64,
        "meta": job.meta or {},
    }
    return f"data: {json.dumps(payload)}\n\n"
```

### Signal Handler

```python
from django.db.models.signals import post_save
from django.dispatch import receiver

@receiver(post_save, sender=OCROutboxJob)
def on_ocr_job_created(sender, instance, created, **kwargs):
    """
    When a new job is created, push it to the connected SSE client's queue.

    If no client is connected, the job stays 'pending' in the DB
    and will be picked up when the client reconnects.
    """
    if not created:
        return  # Only interested in new jobs
    if instance.status != "pending":
        return

    client_id = instance.client_id
    if not client_id:
        return

    with _queue_lock:
        queue = _client_queues.get(client_id)
        if queue is not None:
            queue.put(instance)
```

### SSE View (No Polling!)

```python
import json
from django.http import StreamingHttpResponse
from django.views.decorators.http import require_GET
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone


@csrf_exempt
@require_GET
def ocr_event_stream(request):
    """
    SSE endpoint for a desktop OCR client.

    Phase 1 — On connect: Immediately flush all pending and stuck jobs.
    Phase 2 — Steady state: Wait on Django signal queue + occasional stuck-job check.
    No polling. No time.sleep().
    """
    client_id = request.GET.get("client_id")
    if not client_id:
        return StreamingHttpResponse(
            "data: {\"error\": \"client_id required\"}\n\n",
            content_type="text/event-stream",
            status=400,
        )

    queue = _register_client(client_id)

    def event_stream():
        try:
            # ═══════════════════════════════════════════════════
            # PHASE 1: Send ONE pending job on connect (not all)
            # ═══════════════════════════════════════════════════

            # 1a. Send the oldest pending job (if any)
            first_pending = OCROutboxJob.objects.filter(
                status="pending", client_id=client_id
            ).order_by("created_at").first()

            if first_pending:
                first_pending.status = "processing"
                first_pending.assigned_at = timezone.now()
                first_pending.save()
                yield _serialize_job(first_pending)
            else:
                # 1b. Check for stuck jobs (15-min timeout)
                stuck_count = OCROutboxJob.release_stuck_jobs(
                    client_id, timeout_minutes=15
                )
                if stuck_count:
                    # Send the first recovered job
                    recovered = OCROutboxJob.objects.filter(
                        status="pending", client_id=client_id
                    ).order_by("created_at").first()
                    if recovered:
                        recovered.status = "processing"
                        recovered.assigned_at = timezone.now()
                        recovered.save()
                        yield _serialize_job(recovered)

            # ═══════════════════════════════════════════════════
            # PHASE 2: Wait for signals — one job at a time
            # ═══════════════════════════════════════════════════

            while True:
                # Block until a job arrives in the queue.
                # Jobs arrive here from TWO sources:
                #   a) submit_ocr_result → after a job completes, checks for next pending
                #   b) post_save signal → when admin creates a new job
                # 30-second timeout to check for stuck jobs as safety net.
                try:
                    job = queue.get(timeout=30)
                except Empty:
                    # Safety net: check for stuck jobs every 30s
                    stuck = OCROutboxJob.release_stuck_jobs(
                        client_id, timeout_minutes=15
                    )
                    if stuck:
                        recovered = OCROutboxJob.objects.filter(
                            status="pending", client_id=client_id
                        ).order_by("created_at").first()
                        if recovered:
                            recovered.status = "processing"
                            recovered.assigned_at = timezone.now()
                            recovered.save()
                            yield _serialize_job(recovered)
                    else:
                        # Keepalive ping to prevent proxy timeout
                        yield ": keepalive\n\n"
                    continue

                # A job arrived — send it immediately (one at a time)
                job.status = "processing"
                job.assigned_at = timezone.now()
                job.save()
                yield _serialize_job(job)

        except GeneratorExit:
            pass  # Client disconnected — nothing special to do
        finally:
            _unregister_client(client_id)

    response = StreamingHttpResponse(
        event_stream(), content_type="text/event-stream"
    )
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"   # Disable nginx buffering
    return response
```

### Result Submission View (Auto-Queues Next Job)

```python
from django.db import transaction
from django.http import JsonResponse
from django.views.decorators.http import require_POST
from django.views.decorators.csrf import csrf_exempt
from django.utils import timezone
import json


@csrf_exempt
@require_POST
def submit_ocr_result(request):
    """
    Receive OCR result from the desktop app.

    After saving the result, this view atomically grabs the NEXT pending
    job, marks it as 'processing', and queues it for the SSE stream.

    The atomic lock prevents the signal handler from also grabbing
    the same job — avoiding duplicate processing.
    """
    body = json.loads(request.body)
    job_id = body.get("job_id")
    client_id = body.get("client_id")
    status = body.get("status")

    try:
        job = OCROutboxJob.objects.get(job_id=job_id)
    except OCROutboxJob.DoesNotExist:
        return JsonResponse({"error": "Job not found"}, status=404)

    job.status = "completed" if status == "success" else "failed"
    job.result_data = body.get("data")
    job.error_message = body.get("error_message")
    job.completed_at = timezone.now()
    job.save()

    # ── Queue the next pending job (one at a time) ──
    # Use select_for_update + atomic to prevent race conditions
    # where the signal handler might also try to grab this job.
    next_job = None
    if client_id:
        with transaction.atomic():
            next_job = OCROutboxJob.objects.filter(
                status="pending", client_id=client_id
            ).select_for_update(skip_locked=True).order_by("created_at").first()

            if next_job:
                next_job.status = "processing"
                next_job.assigned_at = timezone.now()
                next_job.save()

        if next_job:
            with _queue_lock:
                queue = _client_queues.get(client_id)
                if queue is not None:
                    queue.put(next_job)

    return JsonResponse({"status": "accepted", "job_id": job_id})
```

### URL Configuration

```python
from django.urls import path

urlpatterns = [
    path("api/ocr/events/", ocr_event_stream, name="ocr-event-stream"),
    path("api/ocr/submit-result/", submit_ocr_result, name="ocr-submit-result"),
]
```

---

## 4. What Happens If OCR App Is Off?

```
User creates job at 10:00 AM (app offline)
  ↓  Job saved to DB with status="pending"
  ↓
App opens next day at 9:00 AM
  ↓  SSE connects → Phase 1 runs
  ↓
Pending job found → sent immediately
  ↓
App processes and submits result
```

**The 15-minute timeout handles edge cases:**
- App crashes mid-OCR → job stuck at "processing"
- On next reconnect → 15-min timeout resets it to "pending"
- Job gets re-delivered and processed again

---

## 5. Environment Variables (on OCR Desktop App `.env`)

```ini
# SSE / Remote OCR Configuration
API_BASE_URL=http://your-django-server.com
API_AUTH_TOKEN=your-auth-token-here
API_CLIENT_ID=neb-ocr-client-1
```

---

## 6. Multi-Worker Note

The in-memory `_client_queues` dict works with:
- `python manage.py runserver` ✅
- Single gunicorn worker ✅
- **Multiple workers** — each worker has its own memory space, so the signal may fire in a different worker than the SSE view. For multi-worker setups, replace the in-memory Queue with **Redis pub/sub** or **channels layers**.
