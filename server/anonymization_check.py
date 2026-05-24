"""Phase 20 — defense-in-depth anonymization scanner.

The phone-side anonymizer is the FIRST line of defense (Phase 21 — strips
peerIds, IPs, paths, SSIDs, BSSIDs, timestamps before upload). This module
is the SECOND line: even if the phone-side anonymizer has a bug or an
attacker-modified build slips through, the server refuses to persist any
transcript containing the PII patterns this scanner detects.

By design this is LOSSY in the safe direction: it errs toward false-positive
(reject a transcript that happens to contain a string matching an IP-shaped
substring inside otherwise-anonymized text). Better to reject 1% of valid
transcripts than to persist 1% with leaked PII.

Patterns scanned:
- IPv4 literals (e.g. 192.168.1.1)
- IPv6 literals (compressed or full)
- Base58 peerId-like strings (12D3 / QmX prefixes — libp2p + IPFS standards)
- macOS / Linux home paths (/home/X/, /Users/X/)
- Common SSID indicators (the anonymizer should have replaced these with
  '<SSID>' or stripped them entirely)
- Wall-clock timestamps in payloads (anonymizer must convert to relative offsets)

The scanner walks all string values in the payload recursively.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional


# IPv4: 4 dot-separated octets, each 0-255. Tight bounds so we don't false-
# positive on version strings like "1.2.3.4-beta". We accept SOME false
# positives (e.g. "192.168.1.1" embedded inside otherwise-anonymized text);
# better to reject than to persist.
_IPV4_RE = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\b"
)

# IPv6: loose — any 2+ colon-separated hex groups of length 1-4. False
# positives possible on hex-style identifiers, but the anonymizer should
# have stripped those too. RFC-5952-correct parsing isn't worth the
# complexity here.
_IPV6_RE = re.compile(r"\b(?:[A-Fa-f0-9]{1,4}:){2,}[A-Fa-f0-9:]{1,}\b")

# libp2p peerIds start with 12D3KooW (Ed25519, post-2020). IPFS legacy CIDs
# starting with Qm are base58 multihashes — also peerId-shaped. Both are
# device-identifying.
_PEERID_LIBP2P_RE = re.compile(r"\b12D3KooW[A-HJ-NP-Za-km-z1-9]{40,}\b")
_PEERID_LEGACY_RE = re.compile(r"\bQm[A-HJ-NP-Za-km-z1-9]{40,46}\b")

# Linux/macOS user home paths. Anonymizer should normalize to '/home/<USER>/'
# or '/Users/<USER>/'. If we see a literal username path, it didn't run.
_HOMEDIR_RE = re.compile(
    r"/(?:home|Users)/(?!<USER>/)[A-Za-z0-9_.-]{1,32}/"
)

# Wall-clock timestamps the anonymizer should have stripped to relative
# offsets. Match ISO-8601-ish anything year-2000-or-later — tight enough
# not to confuse with version numbers or runbook references.
_ISO_TS_RE = re.compile(
    r"\b20\d{2}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\d"
    r"(?:\.\d+)?"
    r"(?:Z|[+-](?:[01]\d|2[0-3]):?[0-5]\d)\b"
)

# Common WiFi SSID indicator strings — the anonymizer should have replaced
# values; if we still see them as field VALUES they leaked. We don't try to
# detect arbitrary SSIDs (impossible), only the most common pattern: the
# anonymizer's expected sentinel is '<SSID>' so anything that looks like a
# plausible SSID alongside an explicit hint is suspicious.
_SSID_HINT_RE = re.compile(r'\bssid["\']?\s*[:=]\s*["\']?(?!<SSID>)[A-Za-z0-9_\- ]{2,32}["\']?', re.IGNORECASE)


# All scanners. (name, compiled regex) — name is what we return so callers
# can log the cause without echoing the matched substring.
_SCANNERS = (
    ("ipv4_literal",    _IPV4_RE),
    ("ipv6_literal",    _IPV6_RE),
    ("peerid_libp2p",   _PEERID_LIBP2P_RE),
    ("peerid_legacy",   _PEERID_LEGACY_RE),
    ("home_directory",  _HOMEDIR_RE),
    ("wallclock_ts",    _ISO_TS_RE),
    ("ssid_hint",       _SSID_HINT_RE),
)


def _walk_strings(obj) -> Iterator[str]:
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_strings(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_strings(v)
    # other JSON-scalar types (int, float, bool, None) are not text-bearing


def find_pii(payload) -> Optional[str]:
    """Return the name of the first scanner that matches anywhere in payload,
    or None if all scanners come up clean. We return only the scanner NAME,
    never the matched substring, so callers can't accidentally log the PII
    they just rejected."""
    for s in _walk_strings(payload):
        for name, rx in _SCANNERS:
            if rx.search(s):
                return name
    return None
