# Documentation archive

Point-in-time snapshots of the pilot's key pages, so prior versions are preserved as the live pages
evolve. **Git history is the authoritative, complete archive** (every commit preserves every page) —
these folders are browsable convenience snapshots taken at notable milestones.

| snapshot | taken at | git HEAD | contents |
|---|---|---|---|
| [snapshot-2026-10-26](snapshot-2026-10-26) | 2026-10-26, before the README doc-integration restructure | `889ddd6` | README, THESIS_SUMMARY, methodology-playbook, standalone-kernel-optimization-playbook, kernel-pilot-performance, e2e-wall-time |

Filenames use `__` for the original path separator (e.g. `docs__e2e-wall-time.md` = `docs/e2e-wall-time.md`).

To recover any page at any past commit directly from git:
`git show <commit>:docs/<page>.md` or `git log --follow -- docs/<page>.md`.
