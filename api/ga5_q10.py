# ga5 q10 - a2a invoice agent
import asyncio
import hashlib
import json
import os
import re
import tempfile
import uuid
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import Response

router = APIRouter()

BASE_URL = "https://tds-ga0-one.vercel.app/a2a/"
A2A_MEDIA = "application/a2a+json"
BATCH_MODE = "application/vnd.ga5.invoice-claim-batch+json"
PROPOSALS_MODE = "application/vnd.ga5.invoice-action-proposals+json"
RECEIPTS_MODE = "application/vnd.ga5.invoice-action-receipts+json"
RESULTS_MODE = "application/vnd.ga5.invoice-action-results+json"

ACTIONS = [
    "settle_invoice",
    "request_approval",
    "hold_invoice",
    "reject_duplicate",
    "open_exception",
]

# module level store. mirrored to /tmp so a warm lambda keeps state
STORE = {"tasks": {}, "msgs": {}, "decisions": {}}
STORE_PATH = os.path.join(tempfile.gettempdir(), "ga5_q10_store.json")
_last_mtime = [0.0]

# locks are made lazily inside the running loop. making them at import time binds
# them to the wrong loop on older pythons and every await blows up
_LOCKS = {}
_LOOP_ID = [None]


def _lock(name):
    try:
        lid = id(asyncio.get_event_loop())
    except Exception:
        lid = 0
    if _LOOP_ID[0] != lid:
        _LOOP_ID[0] = lid
        _LOCKS.clear()
    lk = _LOCKS.get(name)
    if lk is None:
        if len(_LOCKS) > 800:
            for kk in [k for k in list(_LOCKS) if k != "state" and not _LOCKS[k].locked()][:400]:
                _LOCKS.pop(kk, None)
        lk = asyncio.Lock()
        _LOCKS[name] = lk
    return lk


def now_iso():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha_of(obj):
    return hashlib.sha256(canon(obj).encode("utf-8")).hexdigest()


# ---------------- persistence (best effort, /tmp is the only writable spot) ----------------

def load_store():
    try:
        st = os.stat(STORE_PATH)
    except Exception:
        return
    if st.st_mtime <= _last_mtime[0]:
        return
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            for k in ("tasks", "msgs", "decisions"):
                if isinstance(data.get(k), dict):
                    # merge, memory wins for keys we already have
                    for kk, vv in data[k].items():
                        STORE[k].setdefault(kk, vv)
        _last_mtime[0] = st.st_mtime
    except Exception:
        pass


def save_store():
    try:
        tmp = STORE_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(STORE, f, separators=(",", ":"), ensure_ascii=False)
        os.replace(tmp, STORE_PATH)
        _last_mtime[0] = os.stat(STORE_PATH).st_mtime
    except Exception:
        pass  # read only fs, whatever, memory still works


# ---------------- request plumbing ----------------

def jr(payload, status=200, media=A2A_MEDIA):
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return Response(
        content=body,
        status_code=status,
        media_type=media,
        headers={"Cache-Control": "no-store", "A2A-Version": "1.0"},
    )


def err(status, code, message="request rejected"):
    return jr({"error": {"code": code, "message": message}, "code": code, "message": message}, status)


def principal_of(request):
    h = request.headers.get("authorization") or request.headers.get("Authorization") or ""
    bits = h.split(None, 1)
    if len(bits) != 2:
        return None
    if bits[0].strip().lower() != "bearer":
        return None
    tok = bits[1].strip()
    if not tok:
        return None
    return tok


def bad_version(request):
    v = request.headers.get("a2a-version")
    if v is None:
        return False  # be lenient when its missing, only a wrong one is a 400
    v = v.strip()
    return v not in ("1.0", "1", "1.0.0")


def bad_media(request):
    ct = (request.headers.get("content-type") or "").split(";")[0].strip().lower()
    if not ct:
        return False
    if "a2a+json" in ct or ct == "application/json" or ct.endswith("+json"):
        return False
    return True


async def guard(request, want_body=False):
    """returns (error_response, principal). None error means ok"""
    p = principal_of(request)
    if p is None:
        return err(401, "UNAUTHENTICATED", "missing or malformed bearer credentials"), None
    if bad_version(request):
        return err(400, "UNSUPPORTED_VERSION", "A2A-Version 1.0 is required"), None
    if want_body and bad_media(request):
        return err(415, "UNSUPPORTED_MEDIA_TYPE", "application/a2a+json is required"), None
    return None, p


# ---------------- text / fact digging ----------------

REF_RE = re.compile(r"\[([^\[\]\n]{2,80})\]")
DECOY_WORDS = (
    "cover sheet", "coversheet", "archive", "archived", "prior year", "previous year",
    "for reference only", "training", "example only", "sample only", "superseded",
    "historical", "do not use", "illustrative", "legacy note", "old example",
)


def walk_strings(obj, out, depth=0):
    if depth > 8:
        return
    if isinstance(obj, str):
        if len(obj.strip()) >= 25:
            out.append(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            walk_strings(v, out, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            walk_strings(v, out, depth + 1)


def paragraphs_of(pkg):
    chunks = []
    walk_strings(pkg, chunks)
    paras = []
    for c in chunks:
        for piece in re.split(r"\n\s*\n", c):
            piece = piece.strip()
            if len(piece) >= 25:
                paras.append(piece)
    # dedupe but keep order
    seen = set()
    outp = []
    for p in paras:
        if p not in seen:
            seen.add(p)
            outp.append(p)
    return outp


def refs_in(text):
    out = []
    for m in REF_RE.finditer(text):
        r = m.group(1).strip()
        if r and r not in out:
            out.append(r)
    return out


def norm_key(k):
    return re.sub(r"[^a-z0-9]", "", str(k).lower())


FACT_KEYS = {
    "vendorName": ["vendorname", "vendorlegalname", "vendor", "suppliername", "supplier",
                   "payeename", "payee", "billedby", "merchantname", "sellername", "counterparty"],
    "invoiceNumber": ["invoicenumber", "invoiceno", "invoicenum", "invoiceid", "invoicereference",
                      "invoiceref", "billnumber", "documentnumber", "docnumber"],
    "amountMinor": ["amountminor", "totalminor", "grandtotalminor", "invoiceamountminor",
                    "totalamountminor", "netamountminor", "amountminorunits"],
    "currency": ["currency", "currencycode", "isocurrency", "curr"],
}

SYMBOLS = {"₹": "INR", "$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}


def deep_find(obj, wanted, depth=0):
    # breadth-ish search for the first key that looks right
    if depth > 8:
        return None
    if isinstance(obj, dict):
        for k, v in obj.items():
            if norm_key(k) in wanted and isinstance(v, (str, int, float)) and str(v).strip():
                return v
        for v in obj.values():
            got = deep_find(v, wanted, depth + 1)
            if got is not None:
                return got
    elif isinstance(obj, list):
        for v in obj:
            got = deep_find(v, wanted, depth + 1)
            if got is not None:
                return got
    return None


def to_minor(raw):
    try:
        if isinstance(raw, bool):
            return None
        if isinstance(raw, int):
            return int(raw)
        if isinstance(raw, float):
            return int(round(raw))
        s = str(raw).strip().replace(",", "")
        s = re.sub(r"[^0-9.\-]", "", s)
        if not s:
            return None
        if "." in s:
            return int(round(float(s) * 100))
        return int(s)
    except Exception:
        return None


def extract_facts(pkg, text):
    facts = {}

    vend = deep_find(pkg, set(FACT_KEYS["vendorName"]))
    inv = deep_find(pkg, set(FACT_KEYS["invoiceNumber"]))
    amt = deep_find(pkg, set(FACT_KEYS["amountMinor"]))
    cur = deep_find(pkg, set(FACT_KEYS["currency"]))

    if amt is None:
        # maybe a plain amount / total field in major units
        loose = deep_find(pkg, {"amount", "total", "grandtotal", "totalamount", "invoiceamount",
                                "amountdue", "netamount", "amountpayable", "gross", "grossamount"})
        if loose is not None:
            amt = to_minor(loose)
            if amt is not None and isinstance(loose, (int, float)) and float(loose) == int(loose) \
                    and abs(int(loose)) < 100000:
                amt = int(loose) * 100
    else:
        amt = to_minor(amt)

    if not vend:
        m = re.search(r"(?:vendor|supplier|billed\s+by|payee)\s*(?:name)?\s*[:\-]\s*([^\n\r,;]{2,80})",
                      text, re.I)
        if m:
            vend = m.group(1).strip()
    if not inv:
        m = re.search(r"invoice\s*(?:no\.?|number|#|id|ref(?:erence)?)\s*[:\-#]?\s*([A-Za-z0-9][A-Za-z0-9\-\/_]{2,40})",
                      text, re.I)
        if m:
            inv = m.group(1).strip()
    if not cur:
        m = re.search(r"\b(INR|USD|EUR|GBP|AED|SGD|JPY|AUD|CAD|CHF|SEK|ZAR)\b", text)
        if m:
            cur = m.group(1)
        else:
            for sym, code in SYMBOLS.items():
                if sym in text:
                    cur = code
                    break
    if amt is None:
        m = re.search(r"(?:total|amount\s*(?:due|payable)?|grand\s*total)\D{0,15}([0-9][0-9,]*(?:\.[0-9]{1,2})?)",
                      text, re.I)
        if m:
            amt = to_minor(m.group(1))
            if amt is not None and "." not in m.group(1):
                amt = amt * 100

    facts["vendorName"] = str(vend).strip() if vend else "unknown vendor"
    facts["invoiceNumber"] = str(inv).strip() if inv else "unknown"
    facts["amountMinor"] = int(amt) if isinstance(amt, int) else 0
    facts["currency"] = str(cur).strip().upper()[:3] if cur else "INR"
    return facts


# ---------------- rule based decision (fallback / no llm) ----------------

CUES = {
    "reject_duplicate": ["duplicate", "already paid", "already settled", "previously paid",
                         "second submission", "resubmitted the same invoice", "same invoice number was paid",
                         "paid in full earlier", "double billed", "re-billed"],
    "open_exception": ["conflict", "conflicting", "mismatch", "does not match", "do not match",
                       "discrepanc", "contradict", "inconsisten", "irreconcil", "cannot be reconciled",
                       "exception workflow", "records disagree", "material difference"],
    "hold_invoice": ["hold", "pause payment", "pending verification", "awaiting verification",
                     "verification required", "unverified", "bank detail", "bank account change",
                     "remittance change", "until confirmed", "suspend payment", "sanctions screening",
                     "goods receipt is missing", "pending inspection"],
    "request_approval": ["exceeds", "above the", "beyond the", "outside delegated authority",
                         "delegated authority", "approval threshold", "requires approval",
                         "escalate for approval", "over the limit", "authority limit", "sign-off required"],
    "settle_invoice": ["three-way match", "three way match", "fully reconciled", "matches the purchase order",
                       "within the autonomous", "within delegated authority", "clear to pay", "approved for payment",
                       "no discrepancies", "goods receipt confirms"],
}

NEG_RE = re.compile(r"\b(not|no|never|isn'?t|aren'?t|without|nor)\b", re.I)


def para_is_decoy(p):
    low = p.lower()
    return any(w in low for w in DECOY_WORDS)


def score_para(p):
    low = p.lower()
    scores = {a: 0.0 for a in ACTIONS}
    for act, words in CUES.items():
        for w in words:
            idx = low.find(w)
            while idx >= 0:
                window = low[max(0, idx - 45):idx]
                hit = 0.6 if NEG_RE.search(window) else 1.0
                scores[act] += hit
                idx = low.find(w, idx + len(w))
    return scores


def pick_paragraph(paras):
    """find the paragraph that actually decides. prefers a 3 ref paragraph."""
    best = None
    best_val = -1.0
    for i, p in enumerate(paras):
        rs = refs_in(p)
        if not rs:
            continue
        sc = score_para(p)
        top = max(sc.values()) if sc else 0.0
        val = top
        if len(rs) == 3:
            val += 2.5
        if para_is_decoy(p):
            val -= 6.0
        if i == 0:
            val -= 1.2  # first block is usually the cover sheet
        if val > best_val:
            best_val = val
            best = (i, p, rs, sc)
    return best


def rule_decide(pkg):
    paras = paragraphs_of(pkg)
    text = "\n\n".join(paras)
    chosen = pick_paragraph(paras)
    all_refs = refs_in(text)
    if chosen is None:
        refs = all_refs[:3]
        action = "hold_invoice"
        why = "no decisive paragraph could be isolated so payment is paused for manual verification"
    else:
        _, p, rs, sc = chosen
        refs = rs
        ranked = sorted(sc.items(), key=lambda kv: -kv[1])
        action = ranked[0][0] if ranked and ranked[0][1] > 0 else "hold_invoice"
        why = "the decisive paragraph states the condition that drives this action"
    refs = fix_refs(refs, all_refs)
    return {"action": action, "evidenceRefs": refs, "reason": why,
            "facts": extract_facts(pkg, text)}


def fix_refs(refs, all_refs):
    out = []
    for r in refs:
        if r not in out:
            out.append(r)
    for r in all_refs:
        if len(out) >= 3:
            break
        if r not in out:
            out.append(r)
    return out[:3]


# ---------------- llm batch decision ----------------

SYS_PROMPT = (
    "You are an accounts payable control agent. For every invoice package pick exactly one action "
    "and identify the ONE paragraph that decides it.\n"
    "Actions:\n"
    "settle_invoice = valid, reconciled and inside the autonomous payment authority.\n"
    "request_approval = commercially valid but outside the delegated authority (amount/limit/threshold).\n"
    "hold_invoice = payment must pause until a stated verification completes.\n"
    "reject_duplicate = the same commercial invoice was already paid.\n"
    "open_exception = material records conflict and an exception workflow is needed.\n"
    "The documents contain cover sheets, archived prior examples, negations and irrelevant action words. "
    "Ignore those. The decisive paragraph is the one whose facts force the action; it normally carries "
    "exactly three bracketed references.\n"
    'Reply with JSON only: {"decisions":[{"packageId":"...","paragraph":"P3","action":"settle_invoice",'
    '"reason":"one or two sentences of concrete reasoning"}]}'
)


def build_pkg_block(pid, paras, budget):
    lines = ["PACKAGE %s" % pid]
    used = 0
    for i, p in enumerate(paras):
        rs = refs_in(p)
        tag = "P%d" % (i + 1)
        body = p if len(p) <= 1400 else (p[:1400] + " ...")
        chunk = "%s refs=%s :: %s" % (tag, ",".join(rs) if rs else "-", body)
        if used + len(chunk) > budget:
            break
        lines.append(chunk)
        used += len(chunk)
    return "\n".join(lines)


async def llm_decide(pending):
    """pending: list of (pid, pkg, paras). returns dict pid -> {action, paragraph, reason}"""
    token = os.environ.get("AIPIPE_TOKEN") or os.environ.get("AIPIPE_KEY") or ""
    if not token or not pending:
        return {}
    budget = max(1500, int(85000 / max(1, len(pending))))
    blocks = []
    for pid, _pkg, paras in pending:
        blocks.append(build_pkg_block(pid, paras, budget))
    user = "\n\n=====\n\n".join(blocks)
    body = {
        "model": "gpt-4.1-nano",
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0,
    }
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(28.0, connect=8.0)) as cli:
            r = await cli.post(
                "https://aipipe.org/openai/v1/chat/completions",
                headers={"Authorization": "Bearer " + token,
                         "Content-Type": "application/json"},
                json=body,
            )
        if r.status_code >= 400:
            return {}
        txt = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(txt)
    except Exception:
        return {}
    out = {}
    decs = parsed.get("decisions") if isinstance(parsed, dict) else None
    if not isinstance(decs, list):
        return {}
    for d in decs:
        if not isinstance(d, dict):
            continue
        pid = str(d.get("packageId") or "").strip()
        act = str(d.get("action") or "").strip()
        if pid and act in ACTIONS:
            out[pid] = {"action": act,
                        "paragraph": str(d.get("paragraph") or "").strip(),
                        "reason": str(d.get("reason") or "").strip()}
    return out


def build_rationale(action, facts, refs, reason):
    reason = re.sub(r"\s+", " ", reason or "").strip()
    if len(reason) > 800:
        reason = reason[:800].rstrip() + "."
    cited = ", ".join(refs) if refs else "the decisive paragraph"
    txt = ("Action %s for invoice %s from %s (%d %s). %s Decisive evidence: %s."
           % (action, facts.get("invoiceNumber"), facts.get("vendorName"),
              facts.get("amountMinor", 0), facts.get("currency"), reason, cited))
    if len(txt) < 60:
        txt = txt + " The cited references in the decisive paragraph are the only basis for this action."
    if len(txt) > 1500:
        txt = txt[:1490].rstrip() + "."
    return txt


async def decide_batch(packages):
    """one model call for the whole batch, cached by canonical package content"""
    load_store()
    results = {}
    pending = []
    for idx, pkg in enumerate(packages):
        pid = pkg_id(pkg, idx)
        chash = sha_of(pkg)
        cached = STORE["decisions"].get(chash)
        if cached:
            results[pid] = dict(cached)
            continue
        paras = paragraphs_of(pkg)
        results[pid] = rule_decide(pkg)  # deterministic baseline, upgraded if the model answers
        pending.append((pid, pkg, paras))

    if pending:
        try:
            llm = await llm_decide(pending)
        except Exception:
            llm = {}
        for pid, pkg, paras in pending:
            got = llm.get(pid)
            base = results[pid]
            if got:
                base["action"] = got["action"]
                pnum = re.match(r"[Pp]?(\d+)", got.get("paragraph") or "")
                if pnum:
                    i = int(pnum.group(1)) - 1
                    if 0 <= i < len(paras):
                        rs = refs_in(paras[i])
                        if rs:
                            base["evidenceRefs"] = fix_refs(rs, refs_in("\n\n".join(paras)))
                if got.get("reason"):
                    base["reason"] = got["reason"]
            STORE["decisions"][sha_of(pkg)] = dict(base)
    return results


def pkg_id(pkg, idx):
    if isinstance(pkg, dict):
        for k in ("packageId", "package_id", "id", "packageID"):
            v = pkg.get(k)
            if isinstance(v, (str, int)) and str(v).strip():
                return str(v).strip()
    return "pkg-%d" % (idx + 1)


# ---------------- task helpers ----------------

def new_id(prefix):
    return prefix + uuid.uuid4().hex


def action_id_for(batch_id, pid, chash):
    raw = "%s|%s|%s" % (batch_id, pid, chash)
    return "act-" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:28]


def first_part_data(message):
    parts = message.get("parts") if isinstance(message, dict) else None
    if not isinstance(parts, list):
        return None, None
    for p in parts:
        if not isinstance(p, dict):
            continue
        mt = p.get("mediaType") or p.get("media_type") or p.get("mimeType") or ""
        data = p.get("data")
        if isinstance(data, dict) and isinstance(data.get("data"), dict) and "batchId" not in data:
            data = data["data"]  # protojson DataPart nesting, just in case
        if isinstance(data, dict):
            return str(mt), data
    return None, None


def clean_message(msg, task_id, context_id):
    out = {
        "messageId": str(msg.get("messageId") or msg.get("message_id") or new_id("msg-")),
        "role": msg.get("role") or "ROLE_USER",
        "parts": msg.get("parts") if isinstance(msg.get("parts"), list) else [],
        "taskId": task_id,
        "contextId": context_id,
        "kind": "message",
    }
    return out


def agent_note(text, task_id, context_id):
    return {
        "messageId": new_id("msg-agent-"),
        "role": "ROLE_AGENT",
        "parts": [{"mediaType": "text/plain", "text": text}],
        "taskId": task_id,
        "contextId": context_id,
        "kind": "message",
    }


def task_view(rec, history_length=None):
    t = json.loads(json.dumps(rec["task"]))
    if isinstance(history_length, int) and history_length >= 0:
        t["history"] = t.get("history", [])[-history_length:] if history_length else []
    body = json.dumps({"task": t}, separators=(",", ":"), ensure_ascii=False)
    if len(body.encode("utf-8")) > 490000:
        # too fat for the 512 KiB ceiling, shrink the echoed input message
        for m in t.get("history", []):
            for p in m.get("parts", []):
                if isinstance(p, dict) and isinstance(p.get("data"), dict):
                    d = p["data"]
                    if isinstance(d.get("packages"), list):
                        d["packages"] = [{"packageId": pkg_id(x, i)} for i, x in enumerate(d["packages"])]
    return t


# ---------------- routes ----------------

AGENT_CARD = {
    "protocolVersion": "1.0",
    "name": "GA5 Invoice Action Agent",
    "description": "A2A 1.0 agent that reads invoice claim batches, proposes one typed business action "
                   "per package with exact document evidence, and executes only accepted proposals "
                   "after the caller returns signed receipts.",
    "version": "1.0.0",
    "url": BASE_URL,
    "preferredTransport": "HTTP+JSON",
    "provider": {"organization": "TDS GA5 Student Agent", "url": "https://tds-ga0-one.vercel.app/"},
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
        "stateTransitionHistory": True,
        "extensions": [],
    },
    "defaultInputModes": [BATCH_MODE, RESULTS_MODE, "application/json"],
    "defaultOutputModes": [PROPOSALS_MODE, RECEIPTS_MODE, "application/json"],
    "supportedInterfaces": [
        {"url": BASE_URL, "protocolBinding": "HTTP+JSON", "protocolVersion": "1.0"}
    ],
    "securitySchemes": {
        "bearerAuth": {"type": "http", "scheme": "bearer", "description": "Bearer token per principal"}
    },
    "security": [{"bearerAuth": []}],
    "skills": [
        {
            "id": "invoice_action_agent",
            "name": "invoice_action_agent",
            "description": "Reads an invoice claim batch, chooses exactly one of settle_invoice, "
                           "request_approval, hold_invoice, reject_duplicate or open_exception per package, "
                           "cites the decisive document references, and completes the task from grader receipts.",
            "tags": ["invoice", "accounts-payable", "reconciliation", "a2a", "invoice_action_agent"],
            "examples": ["Propose one action for each invoice package in a claim batch"],
            "inputModes": [BATCH_MODE, RESULTS_MODE],
            "outputModes": [PROPOSALS_MODE, RECEIPTS_MODE],
        }
    ],
}


async def handle_card(request: Request):
    return jr(AGENT_CARD, 200, "application/json")


async def handle_send(request: Request):
    try:
        e, principal = await guard(request, want_body=True)
        if e is not None:
            return e
        try:
            raw = await request.body()
            payload = json.loads(raw.decode("utf-8")) if raw else {}
        except Exception:
            return err(400, "INVALID_ARGUMENT", "body must be valid JSON")
        if not isinstance(payload, dict):
            return err(400, "INVALID_ARGUMENT", "body must be a JSON object")

        message = payload.get("message")
        if not isinstance(message, dict):
            message = payload.get("request", {}).get("message") if isinstance(payload.get("request"), dict) else None
        if not isinstance(message, dict):
            return err(400, "INVALID_ARGUMENT", "message is required")

        msg_id = str(message.get("messageId") or message.get("message_id") or "").strip()
        if not msg_id:
            return err(400, "INVALID_ARGUMENT", "messageId is required")

        cfg = payload.get("configuration") if isinstance(payload.get("configuration"), dict) else {}
        hist_len = cfg.get("historyLength")

        mhash = sha_of(message)  # config is deliberately excluded
        dkey = principal + "|" + msg_id

        async with _lock("msg:" + dkey):
            load_store()
            prev = STORE["msgs"].get(dkey)
            if prev:
                if prev.get("hash") != mhash:
                    return err(409, "IDEMPOTENCY_CONFLICT",
                               "message id reused with different content")
                rec = STORE["tasks"].get(prev.get("taskId"))
                if rec and rec.get("principal") == principal:
                    return jr({"task": task_view(rec, hist_len)}, 200)
                return err(409, "IDEMPOTENCY_CONFLICT", "message id reused with different content")

            mt, data = first_part_data(message)
            mt = (mt or "").lower()
            if not isinstance(data, dict):
                return err(400, "INVALID_ARGUMENT", "a data part is required")

            is_results = ("results" in mt) or isinstance(data.get("results"), list)
            if is_results:
                return await do_continuation(principal, message, data, mhash, dkey, hist_len)
            return await do_initial(principal, message, data, mhash, dkey, hist_len)
    except Exception:
        return err(400, "INVALID_ARGUMENT", "request could not be processed")


async def do_initial(principal, message, data, mhash, dkey, hist_len):
    batch_id = str(data.get("batchId") or data.get("batch_id") or "").strip()
    packages = data.get("packages")
    if not isinstance(packages, list) or not packages:
        return err(400, "INVALID_ARGUMENT", "packages are required")

    decisions = await decide_batch(packages)  # the only model call

    task_id = str(message.get("taskId") or "").strip() or new_id("task-")
    context_id = str(message.get("contextId") or "").strip() or new_id("ctx-")

    proposals = []
    stored = {}
    seen_pid = set()
    seen_act = set()
    for idx, pkg in enumerate(packages):
        pid = pkg_id(pkg, idx)
        if pid in seen_pid:
            pid = "%s-%d" % (pid, idx + 1)
        seen_pid.add(pid)
        d = decisions.get(pkg_id(pkg, idx)) or rule_decide(pkg)
        chash = sha_of(pkg)
        aid = action_id_for(batch_id or task_id, pid, chash)
        while aid in seen_act:
            aid = "act-" + hashlib.sha256((aid + "x").encode()).hexdigest()[:28]
        seen_act.add(aid)
        facts = d.get("facts") or {}
        refs = d.get("evidenceRefs") or []
        act = d.get("action") if d.get("action") in ACTIONS else "hold_invoice"
        prop = {
            "packageId": pid,
            "actionId": aid,
            "action": act,
            "facts": {
                "vendorName": str(facts.get("vendorName") or "unknown vendor"),
                "invoiceNumber": str(facts.get("invoiceNumber") or "unknown"),
                "amountMinor": int(facts.get("amountMinor") or 0),
                "currency": str(facts.get("currency") or "INR"),
            },
            "evidenceRefs": [str(r) for r in refs][:3],
            "rationale": build_rationale(act, facts, [str(r) for r in refs][:3], d.get("reason")),
        }
        proposals.append(prop)
        stored[pid] = prop

    art_data = {"batchId": batch_id, "proposals": proposals}
    task = {
        "id": task_id,
        "contextId": context_id,
        "kind": "task",
        "status": {"state": "TASK_STATE_INPUT_REQUIRED", "timestamp": now_iso()},
        "artifacts": [{
            "artifactId": new_id("art-prop-"),
            "name": "invoice-action-proposals",
            "description": "one proposed action per invoice package",
            "parts": [{"mediaType": PROPOSALS_MODE, "data": art_data}],
        }],
        "history": [
            clean_message(message, task_id, context_id),
            agent_note("Proposed %d actions for batch %s. Awaiting results before any execution."
                       % (len(proposals), batch_id or "-"), task_id, context_id),
        ],
        "metadata": {"batchId": batch_id, "policyRevision": str(data.get("policyRevision") or "")},
    }

    async with _lock("state"):
        STORE["tasks"][task_id] = {
            "principal": principal, "task": task, "batchId": batch_id,
            "proposals": stored, "resultsHash": None,
        }
        STORE["msgs"][dkey] = {"hash": mhash, "taskId": task_id}
        save_store()
    return jr({"task": task_view(STORE["tasks"][task_id], hist_len)}, 200)


def results_signature(results):
    sig = []
    for r in results:
        if isinstance(r, dict):
            sig.append({"packageId": str(r.get("packageId") or ""),
                        "actionId": str(r.get("actionId") or ""),
                        "action": str(r.get("action") or ""),
                        "outcome": str(r.get("outcome") or ""),
                        "receiptNonce": str(r.get("receiptNonce") or "")})
    sig.sort(key=lambda x: (x["packageId"], x["actionId"]))
    return sha_of(sig)


async def do_continuation(principal, message, data, mhash, dkey, hist_len):
    task_id = str(message.get("taskId") or message.get("task_id") or "").strip()
    context_id = str(message.get("contextId") or message.get("context_id") or "").strip()
    if not task_id:
        return err(400, "INVALID_ARGUMENT", "taskId is required for a result continuation")

    async with _lock("state"):
        rec = STORE["tasks"].get(task_id)
        if rec is None or rec.get("principal") != principal:
            return err(404, "TASK_NOT_FOUND", "task not found")  # generic on purpose
        task = rec["task"]
        if context_id and context_id != task.get("contextId"):
            return err(400, "INVALID_CONTINUATION", "context does not match the stored task")

        batch_id = str(data.get("batchId") or data.get("batch_id") or "").strip()
        if rec.get("batchId") and batch_id and batch_id != rec["batchId"]:
            return err(400, "INVALID_CONTINUATION", "batch does not match the stored task")

        results = data.get("results")
        if not isinstance(results, list) or not results:
            return err(400, "INVALID_CONTINUATION", "results are required")

        sig = results_signature(results)
        state = task["status"]["state"]
        if state == "TASK_STATE_CANCELED":
            return err(409, "TASK_TERMINAL", "task is already terminal")
        if state == "TASK_STATE_COMPLETED":
            if rec.get("resultsHash") == sig:
                STORE["msgs"][dkey] = {"hash": mhash, "taskId": task_id}
                save_store()
                return jr({"task": task_view(rec, hist_len)}, 200)
            return err(409, "TASK_TERMINAL", "task is already terminal")

        executions = []
        seen = set()
        for r in results:
            if not isinstance(r, dict):
                return err(400, "INVALID_CONTINUATION", "malformed result entry")
            pid = str(r.get("packageId") or "").strip()
            aid = str(r.get("actionId") or "").strip()
            act = str(r.get("action") or "").strip()
            outcome = str(r.get("outcome") or "").strip().upper()
            nonce = r.get("receiptNonce")
            prop = rec["proposals"].get(pid)
            if prop is None:
                return err(400, "INVALID_CONTINUATION", "unknown package in results")
            if aid != prop["actionId"] or act != prop["action"]:
                return err(400, "INVALID_CONTINUATION", "result does not match the stored proposal")
            if outcome not in ("ACCEPTED", "REJECTED"):
                return err(400, "INVALID_CONTINUATION", "unknown outcome")
            if pid in seen:
                return err(400, "INVALID_CONTINUATION", "duplicate package in results")
            seen.add(pid)
            if outcome == "ACCEPTED":
                if not isinstance(nonce, str) or not nonce.strip():
                    return err(400, "INVALID_CONTINUATION", "accepted result needs a receipt nonce")
                executions.append({
                    "packageId": pid,
                    "actionId": aid,
                    "action": act,
                    "receiptNonce": nonce,
                    "facts": dict(prop["facts"]),
                    "evidenceRefs": list(prop["evidenceRefs"]),
                })

        receipt_data = {"batchId": rec.get("batchId") or batch_id, "executions": executions}
        task["artifacts"].append({
            "artifactId": new_id("art-rcpt-"),
            "name": "invoice-action-receipts",
            "description": "executions for accepted proposals only",
            "parts": [{"mediaType": RECEIPTS_MODE, "data": receipt_data}],
        })
        task["history"].append(clean_message(message, task_id, task.get("contextId")))
        task["history"].append(agent_note(
            "Executed %d accepted proposal(s); rejected proposals were not executed."
            % len(executions), task_id, task.get("contextId")))
        task["status"] = {"state": "TASK_STATE_COMPLETED", "timestamp": now_iso()}
        rec["resultsHash"] = sig
        STORE["msgs"][dkey] = {"hash": mhash, "taskId": task_id}
        save_store()
        return jr({"task": task_view(rec, hist_len)}, 200)


async def handle_get_task(request: Request, task_id: str):
    try:
        e, principal = await guard(request)
        if e is not None:
            return e
        load_store()
        rec = STORE["tasks"].get(task_id)
        if rec is None or rec.get("principal") != principal:
            return err(404, "TASK_NOT_FOUND", "task not found")
        hl = request.query_params.get("historyLength")
        try:
            hl = int(hl) if hl is not None else None
        except Exception:
            hl = None
        return jr(task_view(rec, hl), 200)
    except Exception:
        return err(404, "TASK_NOT_FOUND", "task not found")


async def handle_list_tasks(request: Request):
    try:
        e, principal = await guard(request)
        if e is not None:
            return e
        load_store()
        out = []
        for tid, rec in STORE["tasks"].items():
            if rec.get("principal") == principal:
                out.append(task_view(rec))
        return jr({"tasks": out}, 200)
    except Exception:
        return jr({"tasks": []}, 200)


async def handle_cancel(request: Request, task_id: str):
    try:
        e, principal = await guard(request)
        if e is not None:
            return e
        async with _lock("state"):
            load_store()
            rec = STORE["tasks"].get(task_id)
            if rec is None or rec.get("principal") != principal:
                return err(404, "TASK_NOT_FOUND", "task not found")
            state = rec["task"]["status"]["state"]
            if state in ("TASK_STATE_COMPLETED", "TASK_STATE_CANCELED", "TASK_STATE_FAILED",
                         "TASK_STATE_REJECTED"):
                return err(409, "TASK_NOT_CANCELABLE", "task is already terminal")
            rec["task"]["status"] = {"state": "TASK_STATE_CANCELED", "timestamp": now_iso()}
            rec["task"]["history"].append(
                agent_note("Task canceled before any receipt arrived; nothing was executed.",
                           rec["task"]["id"], rec["task"].get("contextId")))
            save_store()
            return jr(task_view(rec), 200)
    except Exception:
        return err(409, "TASK_NOT_CANCELABLE", "task could not be canceled")


# register. the colon paths are literal, starlette handles them fine
for _p in ("/.well-known/agent-card.json", "/a2a/.well-known/agent-card.json",
           "/.well-known/agent.json", "/a2a/agent-card.json"):
    router.add_api_route(_p, handle_card, methods=["GET"], include_in_schema=False)

for _p in ("/a2a/message:send", "/a2a//message:send", "/a2a/v1/message:send"):
    router.add_api_route(_p, handle_send, methods=["POST"], include_in_schema=False)

for _p in ("/a2a/tasks", "/a2a/tasks/", "/a2a/v1/tasks"):
    router.add_api_route(_p, handle_list_tasks, methods=["GET"], include_in_schema=False)

for _p in ("/a2a/tasks/{task_id}", "/a2a/v1/tasks/{task_id}"):
    router.add_api_route(_p, handle_get_task, methods=["GET"], include_in_schema=False)

for _p in ("/a2a/tasks/{task_id}:cancel", "/a2a/v1/tasks/{task_id}:cancel"):
    router.add_api_route(_p, handle_cancel, methods=["POST"], include_in_schema=False)
