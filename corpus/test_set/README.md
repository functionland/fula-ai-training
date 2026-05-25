# Held-out test scenarios (sub-phase 19.3)

Hand-authored canonical scenarios that the eval gate (`training/eval_held_out.py`) runs against any new model. These are NOT user transcripts — they're synthesized adversarial cases that test specific behaviors.

Each scenario is one YAML file:

```yaml
id: kubo_hang_001
description: Kubo container is wedged; container restart should fix.

user_prompt: |
  My device says it's connected but the app can't reach any files. It's been like this for an hour.

# Canned tool responses the eval harness will return when the model
# calls each tool. Lets the harness deterministically score the model's
# verdict + recommendation against the same evidence the real system
# would have shown.
canned_diag_responses:
  diag/summary:
    overall: red
    subsystems:
      containers: {status: red, key_metrics: {oom_count: 0, running_count: 5}}
      internet:   {status: green}
      time:       {status: green}
  diag/containers:
    containers:
      - {name: ipfs_host, state: running, oom_killed: false, restart_count: 7}
      - {name: ipfs_cluster, state: running, oom_killed: false, restart_count: 0}

# Expected behaviors. The scorer measures each independently.
expected:
  severity: red
  root_cause_keywords: ["kubo", "ipfs_host", "wedge", "hang", "stuck"]
  recommended_actions:
    - action_name: docker.restart
      args: {container: ipfs_host}
      tier: 2

# Hard rules — model failures here cause the gate to reject the model.
hard_rules:
  - max_tool_calls: 4
  - no_whitelist_violations: true
```

## Adding a new scenario

1. Reproduce the failure on the lab device or in production audit logs.
2. Capture the diag/* responses that surface the failure.
3. Author the YAML with `id` matching the failure pattern (e.g., `wifi_offline_002`).
4. Decide the expected verdict severity + root_cause_keywords + recommended_actions.
5. Set hard_rules conservatively.
6. Commit. Eval gate picks it up on the next run.

## Spec invariants

- `id` MUST be unique (eval reports key on it)
- `severity` MUST be one of green/yellow/red
- `recommended_actions[*].action_name` MUST exist in `fula-ota/.../action_whitelist.json` (eval cross-validates)
- `canned_diag_responses` MUST cover every diag/* the model might call (else eval returns a synthesized "unknown" response which the model can't reason about)

## Current scenarios (15 shipping in v1)

| File | id | Category |
|---|---|---|
| `kubo_hang_001.yaml` | kubo_hang_001 | Container wedge |
| `wifi_offline_001.yaml` | wifi_offline_001 | Phone offline (root cause is client-side, NOT blox) |
| `captive_portal_001.yaml` | captive_portal_001 | Public WiFi intercept |
| `ntp_drift_001.yaml` | ntp_drift_001 | Clock unsynced |
| `wireguard_stale_001.yaml` | wireguard_stale_001 | WG handshake aged out |
| `discovery_unreachable_001.yaml` | discovery_unreachable_001 | discovery.fula.network blocked |
| `ext4_corrupt_001.yaml` | ext4_corrupt_001 | Storage errors |
| `oom_kubo_001.yaml` | oom_kubo_001 | OOM-killed container |
| `undervoltage_001.yaml` | undervoltage_001 | RK3588 power issue |
| `multi_red_001.yaml` | multi_red_001 | Several subsystems red simultaneously |
| `all_green_001.yaml` | all_green_001 | Truly healthy device (verdict should be green, no action) |
| `slow_sync_001.yaml` | slow_sync_001 | Discovery slow, not unreachable |
| `relay_dead_001.yaml` | relay_dead_001 | Zero circuit reservations |
| `partial_data_001.yaml` | partial_data_001 | Some diag/* return error — model must still finalize |
| `vague_user_001.yaml` | vague_user_001 | User says "it's broken"; model must drill in |
