"""Extended scenarios — supplements scenarios.SCENARIOS with broader
coverage of the runbook sections + adversarial cases. Imported by
generate.py via scenarios.SCENARIOS.extend().

Organized by runbook section, then by adversarial category.
"""
from __future__ import annotations

from corpus.synthetic.scenarios import Recommendation, Scenario


EXTENDED: list[Scenario] = []


# ===========================================================================
# Runbook: WireGuard handshake stale (legitimate bounce)
# ===========================================================================
EXTENDED.append(Scenario(
    id="wg_handshake_stale_bounce",
    category="runbook-section-wireguard",
    scenario_id="freeform",
    user_prompts=[
        "My WG tunnel seems dead.",
        "Wireguard not connecting from my Blox anymore.",
        "Lost connection through the wireguard tunnel.",
        "WG handshake is failing on my device.",
        "VPN to the Blox is broken.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "yellow",
            "generated_at": "2026-05-27T06:00:00Z",
            "subsystems": {
                "wireguard":  {"status": "red",   "key_metrics": {"installed": True, "active": True, "last_handshake_age_sec": 600}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 86400, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/wireguard", {}, {
            "installed": True, "registered": True, "active": True,
            "last_handshake_age_sec": 600, "rx_bytes": 12000000, "tx_bytes": 8000000,
            "persistent_keepalive_sec": 25,
        }),
        ("diag/internet", {}, {
            "dns_ok": True, "https_google_ok": True, "https_discovery_ok": True,
            "latency_ms_avg": 35.0, "captive_portal_likely": False,
        }),
    ],
    verdict_summary="WireGuard handshake is 600 seconds old (well past 3 * keepalive of 75s); tunnel is dead at protocol level even though the service reports active. Underlying internet is fine.",
    verdict_severity="yellow",
    verdict_root_cause="wg_handshake_stale_tunnel_dead",
    recommendations=[
        Recommendation(
            action_name="wireguard.bounce",
            args={},
            reasoning="last_handshake_age_sec=600 with keepalive=25s means we're past 3x keepalive and the tunnel has dropped at protocol level. internet.https_google_ok=true confirms upstream WAN is fine, so bouncing the tunnel should re-handshake cleanly.",
            confidence=0.85,
            tier=2,
        ),
    ],
    think_texts=[
        (
            "WG complaint. Check diag/summary.",
            "User reports WG tunnel issue. Check diag/summary for wireguard subsystem status.",
            "User says WG tunnel is dead. Per the runbook 'WireGuard handshake stale / tunnel dead' section, the first check is diag/wireguard for last_handshake_age_sec, and diag/internet to confirm underlying WAN works (don't bounce a tunnel when WAN is dead). Start with diag/summary.",
        ),
        (
            "WG red, handshake age 600s. Check WG details and internet.",
            "diag/summary shows wireguard.status=red with last_handshake_age_sec=600 in key_metrics. Drill into diag/wireguard for keepalive + transfer counts, and diag/internet to confirm WAN.",
            "diag/summary confirms wireguard.status=red with handshake age 600s. Per the runbook the criterion for bounce is last_handshake_age_sec > 3 * keepalive AND underlying internet OK. Check diag/wireguard for keepalive value + diag/internet to confirm WAN is fine before recommending a bounce.",
        ),
        (
            "Handshake 600s, keepalive 25s — 24x threshold. Internet OK. Bounce.",
            "diag/wireguard confirms keepalive=25s, so 600s = 24x threshold (3x = 75s). Internet is fine (https_google_ok=true). Bounce is safe.",
            "diag/wireguard confirms keepalive=25s, so the 600s handshake age is 24x the threshold (3 * 25 = 75s). diag/internet confirms underlying WAN is fine (https_google_ok=true, latency_ms_avg=35). Per the runbook: 'last_handshake_age_sec > 270 AND diag/internet.https_google_ok=true: wireguard.bounce (tier 2). Confidence: high.' Recommending wireguard.bounce at confidence 0.85.",
        ),
        (
            "After bounce, verify with another diag/wireguard.",
            "After the bounce executes the model can verify by re-running diag/wireguard.",
            "After the bounce executes, follow-up plan is to re-run diag/wireguard and check that last_handshake_age_sec dropped to a small number (typically <30s for a healthy tunnel with 25s keepalive). The recommendation reasoning will mention this verify step.",
        ),
    ],
    variations=10,
    tags=["runbook-wireguard", "legitimate-action", "high-confidence"],
))


# ===========================================================================
# Runbook: ext4 errors — NO action (refer to user)
# ===========================================================================
EXTENDED.append(Scenario(
    id="ext4_errors_refer_user_no_action",
    category="runbook-section-storage",
    scenario_id="freeform",
    user_prompts=[
        "My device feels slow and I see disk errors in dmesg.",
        "I'm getting filesystem errors on my Blox.",
        "Storage seems corrupted on the Blox.",
        "ext4 errors keep showing up.",
        "Disk I/O is failing on my Blox.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "red",
            "generated_at": "2026-05-27T07:00:00Z",
            "subsystems": {
                "storage":    {"status": "red",   "key_metrics": {"ext4_errors": 12, "dmesg_io_errors_1h": 45}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "yellow","key_metrics": {"http_status": 200}},
                "containers": {"status": "yellow","key_metrics": {"running_count": 6, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 2}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 200000, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/storage", {}, {
            "df": {"/uniondrive": {"size_bytes": 1000000000000, "used_bytes": 600000000000, "avail_bytes": 400000000000}},
            "ext4_errors_count": 12,
            "dmesg_io_errors_1h": 45,
            "smartctl_health": "FAILED",
        }),
    ],
    verdict_summary="Storage media is failing: 12 ext4 errors logged + 45 I/O errors in the last hour + smartctl reports FAILED. The Blox needs the storage media replaced — software repair cannot fix bad sectors.",
    verdict_severity="red",
    verdict_root_cause="storage_media_failure",
    recommendations=[],  # NO action — refer to user per runbook
    think_texts=[
        (
            "Disk errors. Check diag/storage. Likely media failure — no software fix.",
            "User reports disk errors. Per runbook ext4 section, this is a media-replacement scenario, not a software repair. Check diag/storage to confirm.",
            "User mentions disk errors and slowness. Per the runbook 'ext4 / disk errors' section: the diagnostic order is diag/storage for errors_count and io_errors_1h. The critical rule is: 'errors_count > 0 OR io_errors_1h > 10: NO automated repair. Tell user to back up and prepare to replace storage media. Confidence: high.' AND: 'NEVER propose node_delete or ipfs_delete when disk errors are present. The next write will hit the same bad sectors.' Check diag/storage to confirm + verdict will recommend backup + replacement.",
        ),
        (
            "Confirmed: 12 ext4 errors, 45 I/O errors, smartctl FAILED. Media dying. No action.",
            "Confirmed: ext4_errors=12, dmesg_io_errors_1h=45, smartctl_health=FAILED. SD card / SATA SSD is dying. NO software repair will fix this — must replace media. Verdict red with no recommendation.",
            "diag/storage confirms: ext4_errors_count=12, dmesg_io_errors_1h=45, smartctl_health=FAILED. The media is genuinely failing. Per the runbook critical rule: NO automated repair, refer the user to back up and replace storage. Specifically warn against node_delete or ipfs_delete which would attempt writes that hit bad sectors. Verdict: red severity, root_cause=storage_media_failure. NO recommendation — software cannot fix hardware failure.",
        ),
    ],
    variations=10,
    tags=["runbook-storage", "no-action", "hardware-issue"],
))


# ===========================================================================
# Runbook: Captive portal (https_google_ok=true, https_discovery_ok=false)
# ===========================================================================
EXTENDED.append(Scenario(
    id="captive_portal_no_action",
    category="runbook-section-internet",
    scenario_id="disconnected",
    user_prompts=[
        "Blox stopped working when I moved to a new wifi.",
        "Just connected my Blox to hotel wifi — won't work.",
        "Coffee shop wifi seems to break my Blox.",
        "My Blox can't reach anything from the conference wifi.",
        "Public wifi is interfering with my Blox.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "red",
            "generated_at": "2026-05-27T08:00:00Z",
            "subsystems": {
                "internet":   {"status": "red", "key_metrics": {"discovery_https_ok": False, "captive_portal_likely": True, "latency_ms_avg": 15}},
                "heartbeat":  {"status": "yellow", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "red", "key_metrics": {"reservation_count": 0}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 7200, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/internet", {}, {
            "dns_ok": True,
            "https_google_ok": False,
            "https_discovery_ok": False,
            "latency_ms_avg": 15.0,
            "captive_portal_likely": True,
        }),
    ],
    verdict_summary="Network has a captive portal active (HTTPS to public sites is intercepted). User needs to sign into the wifi captive portal from a browser to unblock the network for the Blox.",
    verdict_severity="red",
    verdict_root_cause="captive_portal_unauthenticated",
    recommendations=[],  # NO action — user must auth at portal
    think_texts=[
        (
            "New wifi + cannot reach. Check diag/internet for captive_portal_likely.",
            "Recent wifi change. Strong captive-portal signature. Check diag/internet.",
            "User just changed networks (hotel / coffee / conference wifi). Classic captive-portal scenario. Per the runbook 'Internet / DNS / captive portal' section, captive portal signature is dns_ok=true + https_google_ok=false + (sometimes) captive_portal_likely=true. Check diag/internet.",
        ),
        (
            "Captive portal confirmed. User needs to sign in via browser.",
            "diag/internet: dns_ok=true, https_google_ok=false, captive_portal_likely=true. Captive portal confirmed. NO blox action — user must auth in browser.",
            "diag/internet confirms: dns_ok=true (DNS works because portals typically allow DNS to drive their auth flow), https_google_ok=false (HTTPS to public sites blocked), captive_portal_likely=true. Per the runbook: 'Captive portal signature: NO blox action. Tell user their network requires sign-in. Confidence: high.' The Blox can't sign into a captive portal — the user must do it from a phone/laptop browser. After they auth, the network unblocks and the Blox starts working again. No recommendation; verdict explains the situation.",
        ),
    ],
    variations=10,
    tags=["runbook-internet", "no-action", "captive-portal"],
))


# ===========================================================================
# Runbook: Undervoltage (NO action — hardware issue)
# ===========================================================================
EXTENDED.append(Scenario(
    id="undervoltage_refer_user_no_action",
    category="runbook-section-power",
    scenario_id="freeform",
    user_prompts=[
        "My Blox keeps rebooting unexpectedly.",
        "Random reboots on Blox lately.",
        "Device shows undervoltage warnings.",
        "Power issues with my Blox?",
        "Blox restarts itself every few hours.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "red",
            "generated_at": "2026-05-27T09:00:00Z",
            "subsystems": {
                "power":      {"status": "red",   "key_metrics": {"undervoltage_events_24h": 8, "recent_reboots": 6}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "yellow","key_metrics": {"http_status": 200}},
                "containers": {"status": "yellow","key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 2}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
            },
        }),
        ("diag/power", {}, {
            "uptime_s": 1800,
            "undervoltage_events_24h": 8,
            "recent_reboots": 6,
            "max_temp_c": 65.0,
            "soc_voltage_ratio": 0.92,
        }),
    ],
    verdict_summary="Device has 8 undervoltage events in 24h and 6 recent reboots — strong signal of a bad PSU, bad cable, or insufficient power supply. Hardware issue; not fixable in software.",
    verdict_severity="red",
    verdict_root_cause="power_supply_inadequate",
    recommendations=[],  # NO action — runbook says hardware issue
    think_texts=[
        (
            "Reboots. Check diag/power for UV events.",
            "User reports random reboots. Check diag/power for undervoltage events and recent_reboots.",
            "User reports random reboots. Per the runbook 'Power / undervoltage / brownout' section: the diagnostic order is diag/power for undervoltage_events_24h and recent_reboots. If undervoltage_events > 5 OR recent_reboots > 5: NO repair action; tell user to check power cable, swap PSU, move to different outlet. This is almost always physical — AI must not propose a software fix.",
        ),
        (
            "8 UV events + 6 reboots. Hardware. Refer user.",
            "diag/power confirms: undervoltage_events_24h=8, recent_reboots=6. Both over the 5-event threshold. Hardware power issue. NO action. Tell user about PSU/cable.",
            "diag/power confirms 8 undervoltage events in 24h and 6 recent reboots — both well over the runbook's threshold of 5. Per the runbook: 'If UV_events > 5 OR recent_reboots > 5: NO repair action. Tell user to check power cable, swap PSU, move to different outlet. Confidence: high.' Verdict: red severity, root_cause=power_supply_inadequate. NO recommendation. The verdict message will explicitly suggest: check the official 5V/4A power supply, try a different cable, plug into a different outlet (avoid powered USB hubs).",
        ),
    ],
    variations=10,
    tags=["runbook-power", "no-action", "hardware-issue"],
))


# ===========================================================================
# Runbook: Relay reachability — zero reservations, recommend restart_fula
# ===========================================================================
EXTENDED.append(Scenario(
    id="relay_zero_reservations_restart_fula",
    category="runbook-section-relay",
    scenario_id="not-earning",
    user_prompts=[
        "My Blox isn't earning rewards.",
        "Rewards stopped increasing on Blox.",
        "Blox not pinning anything for the pool.",
        "I should be earning but I'm not.",
        "Rewards page hasn't updated in a day.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "yellow",
            "generated_at": "2026-05-27T10:00:00Z",
            "subsystems": {
                "relay":      {"status": "red",   "key_metrics": {"reservation_count": 0, "peer_count": 0}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 86400, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/relay", {}, {
            "relays": [
                {"addr": "relay.fula.network", "swarm_connect_ok": False, "has_circuit_reservation": False, "latency_ms": 0},
                {"addr": "relay.dev.fx.land", "swarm_connect_ok": False, "has_circuit_reservation": False, "latency_ms": 0},
            ],
            "reservation_count": 0,
        }),
        ("diag/internet", {}, {
            "dns_ok": True,
            "https_google_ok": True,
            "https_discovery_ok": True,
            "latency_ms_avg": 60.0,
            "captive_portal_likely": False,
        }),
    ],
    verdict_summary="Blox has zero circuit reservations on relays but underlying internet + discovery are healthy. A restart of the fula stack typically re-registers with relays cleanly.",
    verdict_severity="yellow",
    verdict_root_cause="relay_registration_stale_no_circuits",
    recommendations=[
        Recommendation(
            action_name="restart_fula",
            args={},
            reasoning="Zero circuit reservations (relay.reservation_count=0) but underlying internet (https_discovery_ok=true) is fine. The fula stack's relay-registration goroutine sometimes gets stuck after a long uptime; restart_fula forces a clean re-registration. Tier-2 idempotent.",
            confidence=0.6,
            tier=2,
        ),
    ],
    think_texts=[
        (
            "Not-earning + relay=red. Check internet first.",
            "User reports not-earning. Per runbook, check relay reservations and confirm underlying internet works.",
            "User reports not-earning. Per the runbook 'Relay reachability' section and the user's CSV, the first sanity check is whether the device is reachable via libp2p relays. Zero reservations = blox is unreachable from outside the LAN. Run diag/summary, then drill into diag/relay and diag/internet (run internet diag first to confirm WAN works — otherwise it's not a relay issue).",
        ),
        (
            "Zero reservations, internet OK. Restart_fula to re-register.",
            "Confirmed: 0 reservations, internet is fine, discovery reachable. Per runbook, restart_fula tier-2 to force re-registration. Medium confidence (0.6) — runbook caps confidence here.",
            "Confirmed: relay.reservation_count=0 with internet.https_discovery_ok=true. Per the runbook 'Relay reachability' section: 'Zero circuit reservations + diag/internet.https_discovery_ok=true: restart_fula (tier 2). Forces re-registration with relays. Confidence: medium.' I'll recommend restart_fula at confidence 0.6 (capped per runbook, even though my reasoning is high-conviction — yellow severity should not produce high-confidence restart-class recommendations per rule 9).",
        ),
    ],
    variations=10,
    tags=["runbook-relay", "csv-not-earning", "medium-confidence"],
))


# ===========================================================================
# Adversarial: field confusion (model must cite the field name correctly)
# ===========================================================================
EXTENDED.append(Scenario(
    id="adversarial_cite_field_name",
    category="adversarial-rule-6",
    scenario_id="freeform",
    user_prompts=[
        "Check my Blox's clock and internet please.",
        "Run a diagnostic on time and network.",
        "Latency seems high on my Blox.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "green",
            "generated_at": "2026-05-27T11:00:00Z",
            "subsystems": {
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True, "latency_ms_avg": 93.0}},
                "time":       {"status": "green", "key_metrics": {"synced": True, "offset_ms": 0}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 120000, "undervoltage_events_24h": 0}},
            },
        }),
    ],
    verdict_summary="Device is healthy. internet.latency_ms_avg=93 (network latency) and time.synced=true with offset_ms=0 (clock is correct). The 93 is network latency, not a clock offset — distinct fields.",
    verdict_severity="green",
    verdict_root_cause="no_issue_field_naming_clarified",
    recommendations=[],
    think_texts=[
        (
            "Check both. Cite distinct field names — they're not the same.",
            "User wants clock + internet check. Per rule 6, cite the actual field names: internet.latency_ms_avg is NETWORK latency, time.offset_ms is CLOCK offset. They are not interchangeable.",
            "User asks about clock and internet. Per rule 6, I must cite the actual JSON field names so the verdict is unambiguous. The diag/summary surfaces internet.latency_ms_avg (network round-trip latency) and time.offset_ms (clock skew vs NTP) as DISTINCT metrics. A common confusion (and a lab-observed AI hallucination) is treating 'latency_ms_avg' as 'clock offset of 93ms' — these are completely different concepts. The verdict will explicitly name each field and what it measures.",
        ),
        (
            "All green. internet.latency_ms_avg=93 (network), time.offset_ms=0 (clock). Different fields, both fine.",
            "All green. Distinct fields: internet.latency_ms_avg=93ms is network latency, time.offset_ms=0 is clock skew. Both fine. Verdict explains the distinction.",
            "diag/summary returns all green. internet.latency_ms_avg=93ms (this is the average round-trip latency to discovery + Google over recent probes — NETWORK metric) and time.offset_ms=0 with time.synced=true (this is the system clock's offset vs NTP — CLOCK metric). They are not the same; a model that confuses them would say things like 'your clock is off by 93ms because the latency is 93ms' which is wrong on multiple levels. Verdict: green. The summary will explicitly cite both field names and what they measure to demonstrate rule-6 compliance.",
        ),
    ],
    variations=8,
    tags=["adversarial", "rule-6", "field-citation"],
))


# ===========================================================================
# Adversarial: relay yellow but everything else green (don't over-react)
# ===========================================================================
EXTENDED.append(Scenario(
    id="adversarial_relay_yellow_no_overreact",
    category="adversarial-rule-10",
    scenario_id="freeform",
    user_prompts=[
        "Health check on Blox please.",
        "Diagnose my Blox.",
        "How is my Blox doing?",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "yellow",
            "generated_at": "2026-05-27T12:00:00Z",
            "subsystems": {
                "relay":      {"status": "yellow","key_metrics": {"reservation_count": 0}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 200000, "undervoltage_events_24h": 0}},
            },
        }),
    ],
    verdict_summary="Device is fundamentally healthy: heartbeat green (cloud sees the blox), internet green, all containers up. relay.reservation_count=0 is yellow but informational — only matters if the user is being reached from outside the LAN. No action needed.",
    verdict_severity="green",
    verdict_root_cause="no_issue_relay_yellow_is_informational",
    recommendations=[],
    think_texts=[
        (
            "Yellow overall but only relay is yellow. Per rule 10, relay=0 is informational unless user reports unreachability.",
            "Health check. diag/summary shows yellow overall but only relay is yellow. Per rule 10, relay.reservation_count=0 is informational unless user reports unreachability. Heartbeat green = cloud sees the blox. Verdict green, no action.",
            "Routine health check. diag/summary shows overall=yellow but the contributing subsystem is ONLY relay (reservation_count=0). Per rule 10 of the system prompt: 'relay.reservation_count=0 is NOT a problem on its own — it only matters if the user is trying to be reached from outside their LAN.' The user just asked a generic health check, not 'app can't see my blox'. Heartbeat is green = the cloud sees the blox via discovery POST. Internet is green. All containers up. This is fundamentally healthy. Verdict: green severity (not yellow — the overall=yellow flag isn't a problem the user needs to act on). NO recommendation.",
        ),
    ],
    variations=8,
    tags=["adversarial", "rule-10", "yellow-is-informational"],
))


# ===========================================================================
# Multi-turn drill-in (3 tool calls before verdict)
# ===========================================================================
EXTENDED.append(Scenario(
    id="multi_turn_drill_in_cluster_pebble",
    category="runbook-section-containers",
    scenario_id="freeform",
    user_prompts=[
        "Cluster keeps crashing.",
        "ipfs_cluster won't stay up on Blox.",
        "Cluster container restarting in a loop.",
        "Blox cluster service is unstable.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "red",
            "generated_at": "2026-05-27T13:00:00Z",
            "subsystems": {
                "containers": {"status": "red",   "key_metrics": {"running_count": 6, "oom_count": 0}},
                "storage":    {"status": "yellow","key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "yellow","key_metrics": {"http_status": 200}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "yellow","key_metrics": {"reservation_count": 1}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 86400, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/containers", {}, {
            "containers": [
                {"name": "ipfs_cluster", "state": "restarting", "oom_killed": False, "restart_count": 12,
                 "image": "functionland/ipfs-cluster:test153", "started_at": "2026-05-27T12:55:00Z"},
                {"name": "ipfs_host", "state": "running", "oom_killed": False, "restart_count": 0,
                 "image": "ipfs/kubo:release", "started_at": "2026-05-26T13:00:00Z"},
            ],
        }),
        ("diag/events", {}, {
            "events": [
                {"ts": "2026-05-27T12:30:00Z", "category": "container_restart", "detail": "ipfs_cluster exited 1"},
                {"ts": "2026-05-27T12:40:00Z", "category": "container_restart", "detail": "ipfs_cluster exited 1"},
                {"ts": "2026-05-27T12:50:00Z", "category": "container_restart", "detail": "ipfs_cluster exited 1"},
            ],
        }),
    ],
    verdict_summary="ipfs_cluster is in a tight restart loop (12 restarts, event log shows repeated exits with code 1). docker.restart hasn't broken the loop — likely cluster pebble DB corruption from an unclean shutdown. Suggest docker.restart first as cheap retry; if it doesn't stick, tier-3 node_delete may be needed.",
    verdict_severity="red",
    verdict_root_cause="ipfs_cluster_restart_loop_likely_pebble_corruption",
    recommendations=[
        Recommendation(
            action_name="docker.restart",
            args={"container": "ipfs_cluster"},
            reasoning="ipfs_cluster restart_count=12 with consistent exit-1 in event log. Docker's auto-restart isn't stabilizing — try a clean manual restart first. If it still restarts in a loop after this, escalate to tier-3 node_delete (purges cluster pebble DB). This is the cheaper option first.",
            confidence=0.65,
            tier=2,
        ),
    ],
    think_texts=[
        (
            "Cluster issues. Check summary, containers, events.",
            "User reports cluster issues. Walk: diag/summary → diag/containers for restart counts → diag/events for the pattern over time.",
            "User reports ipfs_cluster instability. Per the runbook 'Kubo / IPFS-cluster API hang' section, the drill-in is: diag/containers for restart_count + state, diag/storage to rule out disk errors, and diag/events for recent restart pattern. Three tool calls before verdict.",
        ),
        (
            "ipfs_cluster restart_count=12. Tight loop. Check events for pattern.",
            "diag/containers confirms ipfs_cluster restart_count=12 in state=restarting. Drill into diag/events for the timeline.",
            "diag/containers confirms the symptom: ipfs_cluster is in state=restarting with restart_count=12. Docker is firing auto-restart repeatedly but not stabilizing — strong signal of a startup-side error. Drill into diag/events to see the restart pattern and any error categories logged.",
        ),
        (
            "Events show consistent exit-1 every 10min. Pebble corruption likely. Try docker.restart first.",
            "diag/events shows ipfs_cluster exit-1 every 10 minutes. Per runbook, this pattern fits cluster pebble DB corruption from unclean shutdown. Try docker.restart first (cheap retry). If it doesn't stick, escalate to tier-3 node_delete.",
            "diag/events confirms the restart cadence: ipfs_cluster exit-1 every ~10 minutes, no other contributing categories. Per the runbook 'Kubo / IPFS-cluster API hang' section: 'Pebble DB corruption signature in cluster logs: tier 3 node_delete (purges chain DB; forces resync over hours). Get explicit user approval. Confidence: medium.' But the runbook also says 'docker.restart container=ipfs_host first, then docker.restart container=ipfs_cluster if still wedged' — so let me try the cheaper docker.restart on ipfs_cluster first. If it doesn't stick (the model can see this on next session), the user can escalate to tier-3. I'll recommend docker.restart container=ipfs_cluster at confidence 0.65 with reasoning that explains the escalation path.",
        ),
        (
            "Recommendation: docker.restart first, then escalate if it loops again.",
            "After the docker.restart executes, follow-up is to wait, then re-check diag/containers — if ipfs_cluster restart_count is still incrementing, escalate to tier-3 node_delete.",
            "After docker.restart container=ipfs_cluster executes, the follow-up plan (for the next session if the user comes back) is: re-check diag/containers, and if restart_count is still incrementing, escalate to tier-3 node_delete. The recommendation reasoning will surface this escalation path so the user knows what to expect.",
        ),
    ],
    variations=10,
    tags=["runbook-container", "multi-turn", "drill-in", "medium-confidence"],
))


# ===========================================================================
# Cannot join pool — CSV-derived "not earning" path
# ===========================================================================
EXTENDED.append(Scenario(
    id="cannot_join_pool_config_missing",
    category="csv-not-earning",
    scenario_id="cannot-join-pool",
    user_prompts=[
        "I can't join the pool from my Blox.",
        "Tried to join pool but nothing happens.",
        "Pool join keeps failing.",
        "How do I make my Blox join the pool?",
        "Blox not on the pool yet.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "green",
            "generated_at": "2026-05-27T14:00:00Z",
            "subsystems": {
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "containers": {"status": "green", "key_metrics": {"running_count": 7, "oom_count": 0}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 100000, "undervoltage_events_24h": 0}},
            },
        }),
    ],
    verdict_summary="Device is healthy across all subsystems — there is no Blox-side issue blocking pool join. Pool membership is set via the app's Settings > Pools flow, which writes pool_id into config.yaml on the device. If the user expected to be joined but isn't, they likely haven't completed the in-app pool-selection step.",
    verdict_severity="green",
    verdict_root_cause="no_blox_issue_user_must_join_via_app",
    recommendations=[],
    think_texts=[
        (
            "Pool-join question. Check diag/summary. Pool membership is app-side, not blox-side.",
            "User asks about pool join. Per the user's CSV runbook: 'if pool_id is not set to 1, then the user is not joined the pool and they need to manually in the app go to settings > pools and join the pool'. Check diag/summary to confirm device is healthy, then explain pool membership flow.",
            "User asks about pool join — this is the 'cannot-join-pool' canonical scenario from the user's CSV. The CSV runbook specifically says: 'if pool_id is not set to 1, then the user is not joined the pool and they need to manually in the app go to settings > pools and join the pool'. So pool join is an APP-side action (user taps a button in Settings > Pools), and the result is that pool_id gets written into /home/pi/.internal/config.yaml on the device. There is no diag/* tool that reads config.yaml directly (current tool set is health-focused, not config-focused). Best response: check diag/summary to confirm the device is healthy (no infrastructure issue blocking pool join), then verdict explaining the user must complete the pool-selection in the app.",
        ),
        (
            "All green. Verdict explains the app-side flow.",
            "All green: device is healthy. Verdict explains pool membership is set via app Settings > Pools.",
            "diag/summary confirms green across the board — there's no Blox-side issue. Verdict: green severity, root_cause='no_blox_issue_user_must_join_via_app'. The summary text will explicitly mention the in-app flow: 'Open the app → Settings → Pools → tap your pool to join'. NO recommendation — there's no diag/* failure to act on, and no whitelisted action covers 'remind the user to tap a button in the app'.",
        ),
    ],
    variations=10,
    tags=["csv-cannot-join-pool", "null-action", "user-side-action"],
))


# ===========================================================================
# Cannot join pool — also a "container check" example since CSV says check ipfs_cluster
# ===========================================================================
EXTENDED.append(Scenario(
    id="cannot_earn_check_ipfs_cluster_first",
    category="csv-not-earning",
    scenario_id="not-earning",
    user_prompts=[
        "Blox connected but not earning.",
        "Pinning isn't happening on my Blox.",
        "ipfs_cluster looks off — my rewards aren't moving.",
    ],
    tool_call_sequence=[
        ("diag/summary", {}, {
            "overall": "red",
            "generated_at": "2026-05-27T15:00:00Z",
            "subsystems": {
                "containers": {"status": "red",   "key_metrics": {"running_count": 6, "oom_count": 0}},
                "internet":   {"status": "green", "key_metrics": {"discovery_https_ok": True}},
                "heartbeat":  {"status": "green", "key_metrics": {"http_status": 200}},
                "time":       {"status": "green", "key_metrics": {"synced": True}},
                "relay":      {"status": "green", "key_metrics": {"reservation_count": 3}},
                "wireguard":  {"status": "green", "key_metrics": {"active": False, "last_handshake_age_sec": -1}},
                "storage":    {"status": "green", "key_metrics": {"ext4_errors": 0, "dmesg_io_errors_1h": 0}},
                "power":      {"status": "green", "key_metrics": {"uptime_s": 100000, "undervoltage_events_24h": 0}},
            },
        }),
        ("diag/containers", {}, {
            "containers": [
                {"name": "ipfs_cluster", "state": "exited", "oom_killed": False, "restart_count": 3,
                 "image": "functionland/ipfs-cluster:test153", "started_at": "2026-05-27T10:00:00Z"},
                {"name": "ipfs_host", "state": "running", "oom_killed": False, "restart_count": 0,
                 "image": "ipfs/kubo:release", "started_at": "2026-05-26T15:00:00Z"},
                {"name": "fula_go", "state": "running", "oom_killed": False, "restart_count": 0,
                 "image": "functionland/go-fula:test153", "started_at": "2026-05-26T15:00:00Z"},
            ],
        }),
    ],
    verdict_summary="ipfs_cluster is in state=exited (not running) which prevents pinning + earning. A docker.restart should bring it back; if it crashes again the cluster pebble DB may need cleanup.",
    verdict_severity="red",
    verdict_root_cause="ipfs_cluster_not_running",
    recommendations=[
        Recommendation(
            action_name="docker.restart",
            args={"container": "ipfs_cluster"},
            reasoning="Per the user's CSV runbook for not-earning: 'check if ipfs_cluster docker which runs cluster and related to earning is running'. ipfs_cluster.state=exited matches that diagnostic. docker.restart container=ipfs_cluster brings it back from a clean state.",
            confidence=0.85,
            tier=2,
        ),
    ],
    think_texts=[
        (
            "Not-earning. CSV says check ipfs_cluster first.",
            "User reports not-earning. Per the CSV runbook: 'check if ipfs_cluster docker which runs cluster and related to earning is running'. Drill into diag/containers.",
            "User reports not-earning. The CSV runbook's check sequence for not-earning specifically calls out ipfs_cluster: 'check if ipfs_cluster docker which runs cluster and related to earning is running'. That's because cluster runs the pinning + reward attestation. Start with diag/summary, then drill into diag/containers if anything's amiss.",
        ),
        (
            "ipfs_cluster exited. Restart it.",
            "ipfs_cluster state=exited. That matches the CSV check. Restart it via docker.restart.",
            "diag/containers confirms ipfs_cluster.state=exited with restart_count=3. Per the user's CSV runbook this is exactly the 'ipfs_cluster not running' case for not-earning. Recommendation: docker.restart container=ipfs_cluster at confidence 0.85. The action is whitelisted and the args.container is within the allowed enum. Tier 2.",
        ),
    ],
    variations=10,
    tags=["csv-not-earning", "rule-9", "legitimate-restart"],
))
