"""Local 4B model running on the user's machine to produce ambient
micro-feedback for the pet — small things too frequent/cheap to round-trip
to Sonnet:

- Idle nudges ("怎么了" / "走神啦?") when user has been quiet
- Expression switches not driven by Aria's main agent state
- Time-of-day flavored ambient acknowledgements

The main Sonnet-backed Aria agent remains the source of all substantive
conversation. The microbrain is intentionally narrow: a small, local
voice for the *pet* as an artifact, not as a chat partner.

See: docs/superpowers/specs/ for the design rationale (TODO: write spec
once Phase 1 confirms 4B can carry the IM register).
"""
