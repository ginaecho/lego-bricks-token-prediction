# Registering a DOI with Zenodo

OpenHarness ships the metadata Zenodo needs (`.zenodo.json`, `CITATION.cff`) and
a release workflow. Minting an actual DOI requires one manual, one-time consent
step on Zenodo's side — a repository owner has to link the GitHub repo, because
Zenodo mints the DOI in *your* name, not ours. Here is the whole flow.

## One-time setup

1. Sign in at <https://zenodo.org> with the **same GitHub account that owns the
   repository** (`ginaecho`). Use *Log in with GitHub* and authorize Zenodo.
2. Go to **Settings → GitHub** (<https://zenodo.org/account/settings/github/>).
3. Find **`ginaecho/lego-bricks-token-prediction`** in the list and flip its toggle **On**.
   (If it isn't listed yet, click *Sync now* — Zenodo re-reads your repos.)

That's it. Zenodo is now watching the repo for releases.

## Mint the DOI

Cutting a release is what actually produces the DOI:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The `Release` workflow (`.github/workflows/release.yml`) runs the tests and
creates a GitHub Release for the tag. Zenodo receives the release webhook,
archives the tagged snapshot, reads `.zenodo.json` for title/authors/keywords,
and mints:

- a **version DOI** — points at `v0.1.0` specifically;
- a **concept DOI** — a permanent identifier that always resolves to the latest
  version. This is the one to cite.

You can also create the release from the GitHub UI (*Releases → Draft a new
release*) instead of pushing a tag.

## After the first release

1. Copy the **concept DOI** from your Zenodo record
   (`https://doi.org/10.5281/zenodo.XXXXXXX`).
2. Replace the placeholder in two places:
   - `CITATION.cff` → `identifiers[0].value`
   - `README.md` → the DOI badge at the top
3. Commit. Every future `git push origin vX.Y.Z` mints a fresh version DOI under
   the same concept DOI automatically — no further setup.

## Why the manual step can't be skipped

A DOI is a claim of authorship and archival responsibility. Zenodo deliberately
requires the repository owner to authorize the link through their own account,
so no third party (including this tooling) can register a DOI on your behalf.
Everything that *can* be automated — metadata, release cutting, re-minting on
each tag — already is; only the initial consent toggle is left to you.
