# Frozen prototype snapshot

This directory preserves an immutable copy of the active `prototype.html` as it
existed at the start of the `live-prototype-exploration` goal, so that the
exploration below can run and reason about the prototype without ever
modifying the active file.

| Field | Value |
|---|---|
| Source file | `prototype.html` (repository root) |
| Snapshot file | `prototype.frozen.20260826.html` |
| SHA-256 | `51F0E00197D3EE8DA2D7D568229EA04900FE5C036E163CDCC24C0E914C975CA6` |
| Snapshot taken | 2026-09-14 (iteration 1 of this goal) |
| Repository HEAD at snapshot time | see `git log -1` in the commit that adds this file |

Verify integrity at any time with:

```powershell
Get-FileHash -Algorithm SHA256 .goals\live-prototype-exploration\snapshot\prototype.frozen.20260826.html
```

The hash must equal `51F0E00197D3EE8DA2D7D568229EA04900FE5C036E163CDCC24C0E914C975CA6`.
If the active `prototype.html` is later edited, this snapshot remains the
frozen reference point for the demo request evaluated in this iteration.
