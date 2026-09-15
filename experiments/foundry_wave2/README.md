# Foundry wave 2 inputs

This directory contains the inputs frozen before the first paid pilot request.
`preregistration.json` fixes the 30-session allocation, blocked randomized
order, strict stop/go gates, model-promotion rule, and USD 200 hard ceiling.
`api_manifests.json` is the complete Fetch allowlist; the controller rejects
every other URL, method, tool, or manifest ID.

The Federal Register snapshots are exact public responses downloaded before
dispatch. Snapshot and live arms ask the same question about the same document.
The live arm differs only by requiring the controlled `fetch_public_api` tool.
The 193,180-byte Federal Register text snapshot supplies deterministic
1,024-, 10,240-, and 102,400-byte Summarise contexts.

Safety rates in the preregistration authorize spend conservatively; they are
not represented as Azure prices. Provider-measured token channels are retained
raw, and monetary cost remains unreconciled until an Azure bill or explicit
rate card is attached.
