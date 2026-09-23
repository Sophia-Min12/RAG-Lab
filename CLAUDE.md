# RAG-Lab — working agreements

## Commits
- **Never add `Co-Authored-By: Claude ...` trailers.** Commits are authored by the repo
  owner alone, so GitHub shows a single avatar.
- No "Generated with Claude Code" footers in commit messages or PR descriptions.

## README roadmap
- **Do not add dates** to roadmap entries. Checking the box (`- [x]`) is the only
  completion marker; the git history already carries the timestamps.

## Language
- All repo content — README, docstrings, comments, commit messages — is written in English.

## Conventions
- Every day folder `NN_topic/dayNN_name/` is self-contained: helpers from earlier days
  are copied forward with a `# reused from dayNN` comment.
- Randomness is always seeded. A test that cannot be re-run to the same number is not
  a test.
- Prefer testing against a **closed-form or independently-known answer** over testing
  a function against its own output.
