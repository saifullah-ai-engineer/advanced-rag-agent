# Contributing

Thanks for helping improve advanced-rag-agent. A few ground rules keep
contributions easy to review.

## How to contribute

1. Fork the repo and create a branch from `main`.
2. Keep changes small and focused. One fix or one feature per pull request.
3. Add or update tests for the behaviour you change.
4. Run the test suite before opening a pull request:
   `python -m pytest`
5. Run the demo to make sure the end to end flow still works:
   `rag-agent demo`
6. Write in plain, direct English. No invented benchmarks or claims.

## What we accept

- Bug fixes with a failing test first.
- Retrieval quality improvements that come with an eval result from
  `rag-agent eval` on the shipped eval set, so the change is measurable.
- New loaders and chunkers, as long as they stay local and dependency light.

## What we do not accept

- Changes that require a paid API or a paid model.
- Changes that remove the offline fallback paths. The system must keep
  working with no API key and no model download.
- Large new dependencies without discussion in an issue first.

## AI assistance disclosure

If you used AI assistance to write a contribution, say so in one line in
the pull request description. A human must own the change and be able to
explain it.
