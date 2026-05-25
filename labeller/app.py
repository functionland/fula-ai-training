"""19.2 — minimal labelling UI.

One-operator FastAPI app that lists raw transcripts and accepts
labels. Designed to run on the operator's laptop pointing at the
synced corpus.

Label decision keys (saved into corpus/labelled/<id>.labelled.json):
  - label_decision: "accept" | "partial" | "reject"
  - verdict_correct: bool
  - actions_correct: bool
  - root_cause_correct: bool
  - notes: free-text
  - labelled_at: ISO timestamp

NOTE: this is intentionally simple. Per Phase 19 sub-plan, the labelling
UI is a focused tool, not a product. A production labelling deployment
might add: multi-reviewer agreement tracking, reviewer auth, audit
log of label changes. Out of scope for v1.

Run:
    BLOX_AI_RAW_DIR=corpus/raw \\
    BLOX_AI_LABELLED_DIR=corpus/labelled \\
    uvicorn labeller.app:app --host 127.0.0.1 --port 8765
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field


RAW_DIR = Path(os.environ.get("BLOX_AI_RAW_DIR", "corpus/raw")).resolve()
LABELLED_DIR = Path(
    os.environ.get("BLOX_AI_LABELLED_DIR", "corpus/labelled")
).resolve()

app = FastAPI(title="fula-ai-training labeller", version="0.1.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Tiny single-page UI. No external assets so it works offline."""
    return _INDEX_HTML


@app.get("/transcripts")
def list_transcripts() -> JSONResponse:
    """Return a list of unlabelled transcripts."""
    if not RAW_DIR.is_dir():
        return JSONResponse(
            status_code=500,
            content={"error": "raw_dir_missing", "path": str(RAW_DIR)},
        )
    labelled_ids = _labelled_ids()
    items = []
    for p in sorted(RAW_DIR.rglob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        uid = data.get("upload_id")
        if not uid or uid in labelled_ids:
            continue
        items.append({
            "upload_id": uid,
            "rating": data.get("user_rating"),
            "comment": (data.get("user_comment") or "")[:200],
            "n_events": len(data.get("events") or []),
            "path": str(p.relative_to(RAW_DIR)),
        })
    return JSONResponse({"unlabelled": items, "total_labelled": len(labelled_ids)})


@app.get("/transcripts/{upload_id}")
def get_transcript(upload_id: str) -> JSONResponse:
    """Return the full transcript for review."""
    for p in RAW_DIR.rglob(f"{upload_id}.json"):
        return JSONResponse(json.loads(p.read_text(encoding="utf-8")))
    return JSONResponse(status_code=404, content={"error": "not_found"})


class LabelRequest(BaseModel):
    model_config = {"extra": "forbid"}
    upload_id: str = Field(min_length=1, max_length=128)
    label_decision: str = Field(pattern="^(accept|partial|reject)$")
    verdict_correct: bool
    actions_correct: bool
    root_cause_correct: bool
    notes: str = Field(default="", max_length=2000)


@app.post("/label")
def post_label(req: LabelRequest) -> JSONResponse:
    LABELLED_DIR.mkdir(parents=True, exist_ok=True)
    # Find the raw transcript so we can embed it in the labelled file
    raw_data = None
    for p in RAW_DIR.rglob(f"{req.upload_id}.json"):
        raw_data = json.loads(p.read_text(encoding="utf-8"))
        break
    if raw_data is None:
        return JSONResponse(status_code=404, content={"error": "not_found"})
    record = {
        "upload_id": req.upload_id,
        "label_decision": req.label_decision,
        "verdict_correct": req.verdict_correct,
        "actions_correct": req.actions_correct,
        "root_cause_correct": req.root_cause_correct,
        "notes": req.notes,
        "labelled_at": datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "transcript": raw_data,
    }
    out = LABELLED_DIR / f"{req.upload_id}.labelled.json"
    out.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return JSONResponse({"ok": True, "path": str(out.relative_to(LABELLED_DIR))})


def _labelled_ids() -> set[str]:
    if not LABELLED_DIR.is_dir():
        return set()
    return {p.stem.replace(".labelled", "") for p in LABELLED_DIR.glob("*.labelled.json")}


# ---------------------------------------------------------------------------
# Minimal HTML (no external deps; works offline)
# ---------------------------------------------------------------------------

_INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Blox AI labeller</title>
<style>
  body{font-family:system-ui,sans-serif;max-width:900px;margin:1em auto;padding:1em}
  pre{background:#f4f4f4;padding:.5em;overflow:auto;max-height:60vh}
  button{margin:.2em;padding:.5em 1em}
  label{display:block;margin:.4em 0}
  textarea{width:100%;height:4em}
</style></head><body>
<h1>Blox AI labeller</h1>
<div id="status">loading...</div>
<div id="transcript"></div>
<form id="label" style="display:none">
  <h3>Label this transcript</h3>
  <label>Decision:
    <select name="label_decision">
      <option value="accept">accept (use for training)</option>
      <option value="partial">partial (some right, some wrong)</option>
      <option value="reject">reject (model behaviour wrong; don't train on)</option>
    </select>
  </label>
  <label><input type="checkbox" name="verdict_correct"> verdict correct</label>
  <label><input type="checkbox" name="actions_correct"> recommended actions correct</label>
  <label><input type="checkbox" name="root_cause_correct"> root cause correct</label>
  <label>Notes: <textarea name="notes"></textarea></label>
  <button type="submit">Save label + next</button>
</form>
<script>
let queue = [];
let idx = 0;
async function refresh(){
  const r = await fetch('/transcripts');
  const j = await r.json();
  queue = j.unlabelled;
  document.getElementById('status').textContent =
    `${queue.length} unlabelled / ${j.total_labelled} labelled so far`;
  if(queue.length === 0){
    document.getElementById('transcript').innerHTML = '<p>no transcripts to label.</p>';
    document.getElementById('label').style.display = 'none';
    return;
  }
  show(0);
}
async function show(i){
  if(i >= queue.length){ await refresh(); return; }
  idx = i;
  const t = queue[i];
  const r = await fetch('/transcripts/' + t.upload_id);
  const full = await r.json();
  document.getElementById('transcript').innerHTML =
    `<h3>${t.upload_id}</h3>
     <p>rating: ${t.rating} | comment: ${t.comment}</p>
     <pre>${JSON.stringify(full.events, null, 2)}</pre>`;
  document.getElementById('label').style.display = 'block';
  document.getElementById('label').reset();
  document.getElementById('label').upload_id = t.upload_id;
}
document.getElementById('label').addEventListener('submit', async e => {
  e.preventDefault();
  const f = e.target;
  const body = {
    upload_id: queue[idx].upload_id,
    label_decision: f.label_decision.value,
    verdict_correct: f.verdict_correct.checked,
    actions_correct: f.actions_correct.checked,
    root_cause_correct: f.root_cause_correct.checked,
    notes: f.notes.value,
  };
  const r = await fetch('/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  });
  if(r.ok){ show(idx+1); } else { alert('save failed'); }
});
refresh();
</script>
</body></html>
"""
