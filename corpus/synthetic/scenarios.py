"""Synthetic scenario definitions for the Blox AI training corpus.

Each scenario is a pure-data Python object. The generator (`generate.py`)
walks this list and produces N variations per scenario (varying user-
prompt phrasing, reasoning-noise level, severity within range, etc.).

To extend the corpus: add a new Scenario instance to SCENARIOS at the
bottom. The generator picks it up on next run.

Why pure-data instead of a YAML-driven generator: scenarios reference
each other (k8s-correction reuses the discovery-unreachable lab
snapshot), and the action_whitelist constraints make Python easier to
keep schema-aligned than YAML. Tradeoff: less hand-editable; gain:
generator never produces a whitelist violation because it can't.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Whitelist + diag enum loaded from fula-ota authoritative source
# ---------------------------------------------------------------------------

DIAG_TOOLS = [
    "diag/summary", "diag/internet", "diag/relay", "diag/time", "diag/power",
    "diag/storage", "diag/containers", "diag/wireguard", "diag/heartbeat",
    "diag/events", "diag/readiness",
]

# Tier 2 (single-tap) — keys mirror action_whitelist.json:tier_2_idempotent
TIER_2_ACTIONS = [
    "restart_fula", "restart_uniondrive",
    "docker.restart", "systemctl.restart", "systemctl.reset-failed",
    "wireguard.bounce", "ntp.resync",
]

# Tier 3 (security-code + press-and-hold) — keys mirror tier_3_destructive
TIER_3_ACTIONS = [
    "reset", "partition", "node_delete", "ipfs_delete", "force_update",
]

# arg constraints (mirrors action_whitelist.json:argument_constraints)
ARG_CONSTRAINTS = {
    "docker.restart": {
        "container": ["ipfs_host", "ipfs_cluster", "ipfs_local", "fula_go",
                       "fula_pinning", "fula_gateway", "fula_fxsupport",
                       "fula_updater"],
    },
    "systemctl.restart": {
        "unit": ["fula.service", "uniondrive.service",
                  "wireguard-support.service", "fula-readiness-check.service",
                  "commands.service", "fula-plugins.service",
                  "firewall.service"],
    },
    "systemctl.reset-failed": {
        "unit": ["fula.service", "uniondrive.service",
                  "fula-readiness-check.service",
                  "fula-readiness-check-recover.service",
                  "wireguard-support.service", "commands.service",
                  "fula-plugins.service", "firewall.service"],
    },
}


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class Recommendation:
    action_name: str
    args: dict = field(default_factory=dict)
    reasoning: str = ""
    confidence: float = 0.7
    tier: int = 2


@dataclass
class Scenario:
    """One troubleshooting flow.

    The generator expands each scenario into N training examples by
    varying user-prompt phrasing (`user_prompts`), reasoning-noise level
    (terse / normal / verbose `<think>` blocks), and severity within
    range. The diag/* responses are fixed per scenario (they ARE the
    scenario — that's what the model sees).
    """
    id: str
    category: str
    user_prompts: list[str]
    scenario_id: str  # one of: disconnected, not-earning, cannot-join-pool, freeform

    # Steps the model takes (in order). Each entry is one assistant
    # turn followed by zero-or-more tool calls + their responses.
    tool_call_sequence: list[tuple[str, dict, dict]] = field(default_factory=list)
    # Format: [(tool_name, args, response_json), ...]

    # Final verdict
    verdict_summary: str = ""
    verdict_severity: str = "yellow"  # green | yellow | red
    verdict_root_cause: str = ""

    # Recommendations to emit (empty = null-action scenario)
    recommendations: list[Recommendation] = field(default_factory=list)

    # Per-turn thinking text. List of (terse, normal, verbose) tuples,
    # one per assistant turn. Generator picks one based on noise sample.
    think_texts: list[tuple[str, str, str]] = field(default_factory=list)

    # Variations to produce per scenario (controls dataset size)
    variations: int = 5

    # Tags for stats reporting
    tags: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# The 2026-05-27 lab snapshot — anchor for "real-lab-anchored" scenarios
# ---------------------------------------------------------------------------

LAB_SUMMARY = {
    "overall": "red",
    "generated_at": "2026-05-27T02:13:14.917Z",
    "subsystems": {
        "time":       {"status": "green",  "key_metrics": {"synced": True}},
        "relay":      {"status": "yellow", "key_metrics": {"reservation_count": 0, "peer_count": 0}},
        "power":      {"status": "green",  "key_metrics": {"uptime_s": 65934, "undervoltage_events_24h": 0}},
        "heartbeat":  {"status": "green",  "key_metrics": {"http_status": 200}},
        "wireguard":  {"status": "yellow", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
        "storage":    {"status": "green",  "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
        "internet":   {"status": "red",    "key_metrics": {"discovery_https_ok": False, "captive_portal_likely": False, "latency_ms_avg": 75.3}},
        "containers": {"status": "green",  "key_metrics": {"running_count": 7, "oom_count": 0}},
    },
}
LAB_INTERNET = {"dns_ok": True, "https_google_ok": True, "https_discovery_ok": False,
                "latency_ms_avg": 52.1, "captive_portal_likely": False}
LAB_RELAY = {"relays": [], "reservation_count": 0}
LAB_HEARTBEAT = {"last_attempt_ts": "2026-05-27T02:08:29Z", "http_status": 200,
                  "last_circuit_count": 6,
                  "last_reserved_on": ["relay.dev.fx.land", "relay.fula.network"]}


# ---------------------------------------------------------------------------
# Scenario catalog
# ---------------------------------------------------------------------------

SCENARIOS: list[Scenario] = [
    # -----------------------------------------------------------------
    # Category A: CSV-derived (the user's runbook)
    # -----------------------------------------------------------------
    Scenario(
        id="disconnected_heartbeat_green_real_lab",
        category="real-lab-anchored",
        scenario_id="disconnected",
        user_prompts=[
            "My Blox is showing as disconnected in the app.",
            "Blox disconnected — please help diagnose.",
            "App says my device is disconnected but I'm right next to it.",
            "Status stuck on 'checking' for my Blox.",
            "Why does the app say my blox is offline?",
            "My Blox just appears disconnected even after I restart the app.",
            "Connectivity issue: app cannot see my blox.",
            "The app reports my blox as disconnected. What should I check?",
        ],
        tool_call_sequence=[
            ("diag/summary",  {}, LAB_SUMMARY),
            ("diag/heartbeat", {}, LAB_HEARTBEAT),
            ("diag/internet", {}, LAB_INTERNET),
        ],
        verdict_summary="Blox can reach the internet but discovery.fula.network is blocked from this network. Device itself is posting heartbeats successfully (heartbeat.http_status=200).",
        verdict_severity="yellow",
        verdict_root_cause="discovery_unreachable_upstream",
        recommendations=[],  # NO action — blame upstream network
        think_texts=[
            (
                "User says disconnect. Per rule 8, check heartbeat first — could be a false positive.",
                "User reports disconnect. Before recommending any restart, check heartbeat: if the device is posting successfully to discovery.fula.network, then 'disconnect' is phone-side or relay-side, not device-side. Per rule 8 I should ASK before restarting on a contradiction. Starting with diag/summary to see the big picture.",
                "The user reports the app shows their Blox as disconnected. This is the classic disconnect-claim scenario, and per the runbook section 'Device looks fine but app says disconnected' the walk-the-ladder approach starts with diag/summary then diag/heartbeat. If heartbeat is green (http_status 200 in recent time), the device IS reachable from the cloud and 'disconnected' is almost certainly a phone-side or network-route issue, NOT a blox issue. Per rule 8 I must NOT recommend restart_fula on a contradiction. Per rule 9 I must NOT cap-bust restart-class actions on yellow severity. Let me start with diag/summary.",
            ),
            (
                "Heartbeat is green (http_status 200, last_attempt 5min ago). Device IS posting. Check internet next.",
                "diag/summary shows overall=red but the heartbeat subsystem is green with http_status=200 — that's the canonical 'I am alive' signal posting to discovery.fula.network. The red overall is driven by internet.status=red, specifically discovery_https_ok=false. So the device's heartbeat goroutine likely cached an earlier success but the current path to discovery.fula.network is broken. Need to drill into internet to confirm.",
                "Looking at the summary in detail: overall=red, but the contributing subsystems split: heartbeat.status=green with http_status=200, internet.status=red with discovery_https_ok=false, relay.status=yellow with reservation_count=0. This is exactly the contradiction pattern rule 8 warns about — the user said 'disconnected' but heartbeat.http_status=200 means the device successfully POST'd to discovery.fula.network at last_attempt_ts (5 min ago). The internet red explains relay yellow (relays are discovered via discovery.fula.network — if discovery is unreachable, relay list goes stale). I should NOT recommend restart_fula because that would create the very disconnect the user is reporting. Let me drill into diag/internet to see whether this is captive portal, ISP block, or something else.",
            ),
            (
                "DNS works, Google reachable, discovery blocked. Not captive (Google works). Upstream block.",
                "diag/internet confirms: dns_ok=true, https_google_ok=true, https_discovery_ok=false, captive_portal_likely=false. Pattern is clear — the network allows general internet but blocks discovery.fula.network specifically. Could be ISP filtering, country-level block, corporate firewall rule, or transient outage of discovery infra. None of these are fixable by restarting the blox. Verdict: device healthy, network-side block.",
                "diag/internet confirms the picture: dns_ok=true (so DNS resolution works for at least one query), https_google_ok=true (so general HTTPS exit from this network works), https_discovery_ok=false (specifically discovery.fula.network is unreachable), captive_portal_likely=false (we'd see https_google_ok=false in that case). This means the network allows general internet traffic but specifically blocks or fails to route discovery.fula.network. Common causes per the runbook 'Internet / DNS / captive portal' section: ISP filtering, country-level block (rare for .network TLDs), corporate firewall whitelist that doesn't include discovery, or a transient outage on discovery's side. None of these are addressable by restarting anything on the blox — the device is healthy, the upstream path is broken. Per the runbook's 'Critical rule: NEVER restart anything when the root cause is upstream network. The restart hides the diagnosis without fixing the cause.' I should NOT recommend restart_fula. Verdict will be yellow severity (device healthy but service partially impaired) with root_cause 'discovery_unreachable_upstream'.",
            ),
        ],
        variations=14,
        tags=["real-lab", "heartbeat_green_trap", "rule-8", "no-action"],
    ),

    # -----------------------------------------------------------------
    # Category B: K8s-correction (user uses kubernetes terminology)
    # -----------------------------------------------------------------
    Scenario(
        id="user_says_pod_correction",
        category="adversarial-k8s",
        scenario_id="freeform",
        user_prompts=[
            "Is one of my pods crashing? My device feels slow.",
            "I think my kubelet is having issues — Blox is unresponsive.",
            "Can you kubectl get pods? Something is wrong with the cluster.",
            "Check if my kube-proxy is healthy please.",
            "My kubernetes node seems unhealthy.",
            "Run kubectl describe to see what's broken.",
            "Check pod status for me — Blox is slow.",
            "Is there a CrashLoopBackOff happening on my device?",
        ],
        tool_call_sequence=[
            ("diag/summary", {}, {
                "overall": "green",
                "generated_at": "2026-05-27T03:00:00Z",
                "subsystems": {
                    "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                    "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                    "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                    "time":       {"status": "green", "key_metrics": {"synced": True}},
                    "relay":      {"status": "green", "key_metrics": {"reservation_count": 4}},
                    "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                    "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                    "power":      {"status": "green", "key_metrics": {"uptime_s": 120000, "undervoltage_events_24h": 0}},
                },
            }),
        ],
        verdict_summary="Device is healthy across all subsystems. The Fula stack uses Docker (not Kubernetes); the running containers are ipfs_host, ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport, and fula_updater.",
        verdict_severity="green",
        verdict_root_cause="no_issue_kubernetes_terminology_mismatch",
        recommendations=[],
        think_texts=[
            (
                "User mentioned pods/kubelet — but this is Docker, not Kubernetes. Translate to container terminology in response. Check diag/summary first.",
                "User used Kubernetes terminology (pod / kubelet / kubectl / kube-proxy). Per rule 11, this device does NOT run Kubernetes — it runs Docker with the named containers ipfs_host, ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport, fula_updater. I should NOT use the user's terminology back at them; instead translate to the actual stack. Check diag/summary to see current health.",
                "User's prompt mentions Kubernetes-specific concepts (pod / kubelet / kubectl / kube-proxy / CrashLoopBackOff). This is the rule-11 scenario: NONE of those exist on this device. The Fula stack runs ENTIRELY on Docker, with this exact list of containers: ipfs_host (kubo, the IPFS daemon — 'kubo' is the rename of 'go-ipfs', NOT short for kubernetes), ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport, fula_updater. Plus host systemd services: fula.service, uniondrive.service, wireguard-support.service. I will translate the user's question into the correct vocabulary in my response — not use kubelet/kube-proxy/pod terms back at them. Let me check diag/summary to see the actual device state, then frame the verdict in correct stack terminology.",
            ),
            (
                "All green. No issue. Verdict: healthy + correct vocabulary.",
                "diag/summary shows overall=green with every subsystem green: 7 containers running, 0 OOM, heartbeat OK, time synced, relays connected, no power issues, no storage errors. Device is genuinely healthy. The user's perception of 'slow' or 'crashing pods' doesn't match observable evidence. Final verdict: green severity. In the verdict text I'll explicitly name the Docker stack so the user knows this is NOT a Kubernetes system.",
                "diag/summary shows overall=green and every subsystem reads green: containers.running_count=7 with oom_count=0 (no CrashLoopBackOff equivalent — all containers stable), internet healthy, heartbeat 200, time synced, relays connected with 4 reservations, no undervoltage or thermal issues. The device is genuinely fine. The user's 'feels slow' or 'pods crashing' impression doesn't match the data. Final verdict: green severity. In the verdict summary I'll explicitly name the Docker stack (ipfs_host, ipfs_cluster, fula_go, fula_pinning, fula_gateway, fula_fxsupport, fula_updater) so the user understands their mental model of the system needs updating — this is Docker, not Kubernetes, and 'pod / kubelet / kubectl' don't apply here. No recommended_action because there's nothing to fix.",
            ),
        ],
        variations=14,
        tags=["adversarial", "rule-11", "k8s-correction"],
    ),

    # -----------------------------------------------------------------
    # Category C: Restart-class false-positive (heartbeat green + connectivity complaint)
    # -----------------------------------------------------------------
    Scenario(
        id="false_positive_restart_fula_blocked",
        category="adversarial-rule-8",
        scenario_id="disconnected",
        user_prompts=[
            "My Blox is showing as disconnected. Please restart it.",
            "Blox unreachable — just restart the fula stack to fix it.",
            "Cannot reach Blox; let's force a restart.",
            "Restart everything please, my Blox is offline.",
            "Just do a full restart, my device shows as disconnected.",
        ],
        tool_call_sequence=[
            ("diag/summary", {}, LAB_SUMMARY),
        ],
        verdict_summary="Device is currently posting heartbeats successfully (http_status 200) and 'disconnected' in the app may be phone-side or upstream-network. Restarting would cause real downtime to address a likely-false problem. Suggest checking phone connectivity first.",
        verdict_severity="yellow",
        verdict_root_cause="disconnect_claim_with_green_heartbeat",
        recommendations=[],
        think_texts=[
            (
                "User wants restart. But heartbeat is green — restart would cause real disconnect. Push back politely.",
                "User explicitly asked for a restart. Per rule 8 + rule 9 I should NOT comply blindly when the data contradicts the symptom: diag/summary shows heartbeat.status=green with http_status=200 = device IS reachable from cloud. A restart now would cause real ~60s downtime to fix a problem that may not exist. I'll ask before recommending OR emit a verdict explaining the data conflict and NOT recommending restart.",
                "User explicitly asked me to restart the fula stack. This is the rule-8 contradiction scenario in its purest form: heartbeat.status=green with http_status=200 means the device IS posting successfully to discovery.fula.network RIGHT NOW. A restart would cause 30-60 seconds of real downtime to address a 'disconnect' that doesn't match observable evidence. Per rule 8: 'If the user reports a symptom but the diagnostic data CONTRADICTS it, ASK before acting.' I will NOT recommend restart_fula here. Instead the verdict will explain the heartbeat-200 evidence and suggest phone-side checks. The user can ask again with phone context and I'll have something to base a restart decision on. Per rule 9: high-confidence restart_fula at severity=yellow is a self-fulfilling problem — the restart briefly disconnects, 'confirming' the false diagnosis.",
            ),
        ],
        variations=10,
        tags=["adversarial", "rule-8", "no-action", "user-requested-restart"],
    ),

    # -----------------------------------------------------------------
    # Category D: Kubo wedged / OOM-killed (legitimate restart scenario)
    # -----------------------------------------------------------------
    Scenario(
        id="kubo_oom_legitimate_restart",
        category="runbook-section-containers",
        scenario_id="freeform",
        user_prompts=[
            "My Blox has not been syncing files for the last hour.",
            "I keep getting errors when I try to upload anything to my Blox.",
            "Files I uploaded yesterday are missing from my Blox.",
            "IPFS doesn't seem to be running on my device.",
            "kubo died on my Blox — how do I fix it?",
        ],
        tool_call_sequence=[
            ("diag/summary", {}, {
                "overall": "red",
                "generated_at": "2026-05-27T03:30:00Z",
                "subsystems": {
                    "containers": {"status": "red",   "key_metrics": {"running_count": 6, "oom_count": 1, "ipfs_host_oom": True}},
                    "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                    "heartbeat":  {"status": "yellow","key_metrics": {"http_status": 200}},
                    "time":       {"status": "green", "key_metrics": {"synced": True}},
                    "relay":      {"status": "yellow","key_metrics": {"reservation_count": 0}},
                    "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                    "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                    "power":      {"status": "green", "key_metrics": {"uptime_s": 86400, "undervoltage_events_24h": 0}},
                },
            }),
            ("diag/containers", {}, {
                "containers": [
                    {"name": "ipfs_host", "state": "restarting", "oom_killed": True, "restart_count": 4,
                     "image": "ipfs/kubo:release", "started_at": "2026-05-27T03:25:00Z"},
                    {"name": "ipfs_cluster", "state": "running", "oom_killed": False, "restart_count": 0,
                     "image": "functionland/ipfs-cluster:test153", "started_at": "2026-05-26T03:30:00Z"},
                ],
            }),
        ],
        verdict_summary="ipfs_host (kubo) was OOM-killed; restart_count=4 indicates docker's auto-restart is firing but not stabilizing. A clean restart should help; if it recurs, the device may be RAM-constrained.",
        verdict_severity="red",
        verdict_root_cause="ipfs_host_oom_killed",
        recommendations=[
            Recommendation(
                action_name="docker.restart",
                args={"container": "ipfs_host"},
                reasoning="ipfs_host shows oom_killed=true with restart_count=4 — docker is auto-restarting but not stabilizing. A forced clean restart from a known state often clears whatever runaway allocation triggered the OOM. If it recurs after this restart, the device is likely RAM-constrained (check diag/power for total RAM via uptime_s and other metrics).",
                confidence=0.85,
                tier=2,
            ),
        ],
        think_texts=[
            (
                "User reports sync issues. Check diag/summary; containers may be red.",
                "User reports IPFS sync issues. Most likely: ipfs_host (kubo) is in a bad state. Start with diag/summary to confirm.",
                "User says files aren't syncing / IPFS is broken. The most likely root cause per the runbook 'Container OOM' and 'Kubo / IPFS-cluster API hang' sections is that ipfs_host (which IS kubo, the IPFS daemon — 'kubo' is the rename of go-ipfs, not kubernetes) is either OOM-killed, wedged, or hit a deadlock. Start with diag/summary to see overall state, then drill into diag/containers if containers subsystem is red.",
            ),
            (
                "Summary red, containers red, oom_count=1, ipfs_host_oom=true. Confirmed. Drill into containers.",
                "diag/summary confirms containers.status=red with oom_count=1 and the key_metric ipfs_host_oom=true. Drill into diag/containers for restart_count and state to decide the right action.",
                "diag/summary confirms: overall=red driven by containers.status=red with oom_count=1 and ipfs_host_oom=true in key_metrics. Internet+time+power+storage all green so this is a contained-blast container issue, not an environmental one. Heartbeat is yellow (probably degraded because ipfs_host is restarting and the heartbeat goroutine's last successful POST is aging). Drill into diag/containers to get restart_count + state for ipfs_host and decide whether docker.restart is the right call.",
            ),
            (
                "ipfs_host: oom_killed=true, state=restarting, restart_count=4. Docker auto-restart not stabilizing. Forced clean restart should help.",
                "diag/containers confirms: ipfs_host is in state=restarting with oom_killed=true and restart_count=4. Docker's auto-restart is firing but not stabilizing the container — usually means there's runaway allocation early in startup or stale state on disk. A forced clean restart via docker.restart often clears this. If it recurs after this restart, the device is likely RAM-constrained (the 1.7B model + RKLLM toolkit + 7 fula containers + kubo at moderate pin count is close to the limit on 7.7 GB devices). I'll recommend docker.restart container=ipfs_host at high confidence (~0.85).",
                "diag/containers shows ipfs_host: state=restarting, oom_killed=true, restart_count=4. Docker is auto-restarting after each OOM but not stabilizing — usually means a runaway allocation happens early enough in kubo init that it crosses the cgroup limit before the daemon settles. A manual docker.restart from a clean state often breaks this loop because it gives the kernel a chance to reclaim memory between the kill and the restart. Per the runbook 'Container OOM' section: 'One container OOMKilled, restart_count low: docker.restart container=<name> (tier 2). Auto-restart by Docker should already fire; this is for forcing a clean state. Confidence: high.' I'll recommend docker.restart container=ipfs_host with confidence 0.85, args.container per the whitelist constraint (must be one of the allowed container names). Tier 2. The recommendation reasoning will mention that if the restart doesn't stick (oom recurs), the device may be RAM-constrained and the user should consider whether to uninstall blox-ai to free the ~2.4 GB the model uses.",
            ),
        ],
        variations=10,
        tags=["runbook-container", "rule-9", "legitimate-restart", "high-confidence"],
    ),

    # -----------------------------------------------------------------
    # Category E: NTP drift (legitimate ntp.resync scenario)
    # -----------------------------------------------------------------
    Scenario(
        id="ntp_drift_legitimate_resync",
        category="runbook-section-time",
        scenario_id="freeform",
        user_prompts=[
            "My Blox isn't accepting any requests; says timestamps invalid.",
            "Clock seems off on my Blox.",
            "Getting 'timestamp skew' errors when my phone tries to talk to Blox.",
            "Authentication failing on Blox API calls.",
        ],
        tool_call_sequence=[
            ("diag/summary", {}, {
                "overall": "yellow",
                "generated_at": "2026-05-27T04:00:00Z",
                "subsystems": {
                    "time":       {"status": "red",   "key_metrics": {"synced": False, "offset_ms": 180000}},
                    "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                    "heartbeat":  {"status": "yellow","key_metrics": {"http_status": 401}},
                    "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                    "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                    "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                    "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                    "power":      {"status": "green", "key_metrics": {"uptime_s": 30000, "undervoltage_events_24h": 0}},
                },
            }),
            ("diag/time", {}, {
                "synced": False,
                "offset_ms": 180000,
                "service": "chronyd",
                "last_sync_ts": "2026-05-26T20:00:00Z",
            }),
        ],
        verdict_summary="Clock is unsynced with a 3-minute offset (offset_ms=180000). Heartbeat is failing with http_status=401 (signature verification) — the timestamp-signed heartbeat is being rejected by discovery. Force NTP resync.",
        verdict_severity="red",
        verdict_root_cause="ntp_unsynced_breaking_signed_heartbeat",
        recommendations=[
            Recommendation(
                action_name="ntp.resync",
                args={},
                reasoning="time.synced=false with offset_ms=180000 (3 minutes). This is breaking the timestamp-signed heartbeat (heartbeat.http_status=401 is the signature-rejection symptom). ntp.resync forces chronyd / timesyncd to step the clock immediately. Safe and idempotent.",
                confidence=0.9,
                tier=2,
            ),
        ],
        think_texts=[
            (
                "User says timestamps invalid. Check diag/time.",
                "User reports timestamp/auth issues. Heartbeat is signed with the device clock — clock skew breaks it. Check diag/time.",
                "User reports timestamp-invalid errors and authentication failures on their Blox. This is the classic NTP-drift symptom: the heartbeat to discovery.fula.network is signed with the device clock, and discovery rejects timestamps off by more than a few minutes. Per the runbook 'NTP / clock skew' section, check diag/time for synced + offset_ms, and diag/heartbeat to confirm http_status=401 (signature rejection).",
            ),
            (
                "Time red, offset 180s = 3min. Heartbeat 401. ntp.resync is the right tier-2 fix.",
                "diag/summary confirms time.status=red with offset_ms=180000 (3 min off) and heartbeat.http_status=401 — the signature-rejection symptom from clock skew. ntp.resync is the tier-2 idempotent fix per the runbook.",
                "Confirmed: time.synced=false, offset_ms=180000 (3 minutes), service=chronyd. Heartbeat http_status=401 confirms the signed-heartbeat rejection. Per the runbook 'NTP / clock skew' section: 'synced=false AND offset_ms > 60_000: ntp.resync (tier 2, idempotent). Confidence: high.' I'll recommend ntp.resync at confidence 0.9. The action is idempotent (running it twice does no harm) and the executor maps it to either `chronyc -a makestep` or `systemctl restart systemd-timesyncd` depending on which daemon is active.",
            ),
        ],
        variations=10,
        tags=["runbook-time", "legitimate-action", "high-confidence"],
    ),

    # -----------------------------------------------------------------
    # Category F: NULL ACTION — fully healthy
    # -----------------------------------------------------------------
    Scenario(
        id="healthy_no_action_needed",
        category="null-action",
        scenario_id="freeform",
        user_prompts=[
            "Just a quick check — is my Blox healthy?",
            "Run diagnostics please.",
            "Any issues with my Blox right now?",
            "Health check on Blox.",
            "Tell me if my device is working properly.",
            "Status check.",
        ],
        tool_call_sequence=[
            ("diag/summary", {}, {
                "overall": "green",
                "generated_at": "2026-05-27T05:00:00Z",
                "subsystems": {
                    "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                    "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                    "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                    "time":       {"status": "green", "key_metrics": {"synced": True}},
                    "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                    "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                    "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                    "power":      {"status": "green", "key_metrics": {"uptime_s": 432000, "undervoltage_events_24h": 0}},
                },
            }),
        ],
        verdict_summary="Device is healthy: all subsystems green, heartbeat posting successfully, 3 active relay reservations, no container restarts.",
        verdict_severity="green",
        verdict_root_cause="no_issue",
        recommendations=[],  # NO recommendation — critical!
        think_texts=[
            (
                "Health check. Run diag/summary, expect green.",
                "User wants a health check. Run diag/summary; if all green, verdict is green with NO recommendation (critical — don't invent a fix when nothing is broken).",
                "User asked for a routine health check. The runbook is clear: if everything is green, the verdict is green and there is NO recommendation. Per rule (implicit but critical): don't invent fixes when nothing is broken. A model trained to always recommend something is a model that creates problems on healthy devices. Run diag/summary and expect green across the board.",
            ),
            (
                "All green. No action needed. Verdict green.",
                "diag/summary returns overall=green with every subsystem green. Device is healthy. No recommendation needed — this is exactly the 'verdict green, zero recommendations' case the runbook treats as a valid outcome.",
                "diag/summary returns overall=green with every subsystem reading green: internet healthy with discovery reachable, heartbeat 200, 7 containers running with 0 OOMs, time synced, 3 active relay reservations, no power events, no storage errors, 5 days of uptime. The device is genuinely healthy. Verdict: green severity with summary noting the healthy state and the 3 relay reservations as evidence of working connectivity. ZERO recommendations — this is the null-action case and recommending anything would be inventing a problem.",
            ),
        ],
        variations=12,
        tags=["null-action", "healthy", "no-recommendation"],
    ),

    # -----------------------------------------------------------------
    # Category G: user_question — vague prompt
    # -----------------------------------------------------------------
    Scenario(
        id="vague_user_clarification_needed",
        category="user-question",
        scenario_id="freeform",
        user_prompts=[
            "Something is wrong.",
            "Help.",
            "It's broken.",
            "Not working.",
            "Why isn't this working?",
            "Hmm, problem.",
        ],
        tool_call_sequence=[],  # No tool call — model asks a question first
        verdict_summary="",  # No verdict yet — this scenario terminates on user_question (the trainer will see the question as the assistant's final turn here)
        verdict_severity="yellow",
        verdict_root_cause="",
        recommendations=[],
        think_texts=[
            (
                "Vague. Ask clarifying question with choice options for symptom class.",
                "Prompt is too vague to drill in on. Ask a clarifying user_question with choice options matching the quick-start scenario classes.",
                "User's prompt is too vague to action — could mean anything. Per the runbook section on user_question: 'Most of the time you should NOT ask the user a question — calling a diag/* tool is faster... ONLY when the symptom is genuinely ambiguous and the answer would change which diag/* tool you call next.' This is exactly that case. Best to ask one clarifying question with choice options matching the canonical scenario classes (disconnected, not earning, can't join pool, freeform/other) — that maps to which diag-walk to start.",
            ),
        ],
        # The "verdict" for this scenario will actually be a user_question event
        # — handled specially by the generator
        variations=8,
        tags=["user-question", "clarification", "no-tool-call-yet"],
    ),
]


# ---------------------------------------------------------------------------
# Phrasing pool for user-prompt augmentation
# ---------------------------------------------------------------------------

NEUTRAL_PREFIXES = ["", "Hi — ", "Hey, ", "Hi, ", "Quick question: ", ""]
NEUTRAL_SUFFIXES = ["", " Thanks!", " Please help.", " Any ideas?", ""]


def expand_prompt(base: str, seed: int) -> str:
    """Light augmentation: optional prefix/suffix per seed. Keeps the
    semantic content identical."""
    pre = NEUTRAL_PREFIXES[seed % len(NEUTRAL_PREFIXES)]
    post = NEUTRAL_SUFFIXES[(seed // len(NEUTRAL_PREFIXES)) % len(NEUTRAL_SUFFIXES)]
    return f"{pre}{base}{post}"
