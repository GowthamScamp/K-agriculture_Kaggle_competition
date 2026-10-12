# Local GitHub publishing checklist

This file is a checklist only. It does not configure a remote, create a GitHub
repository, commit, push, or upload anything.

## Completed repository hygiene

- A standalone local Git repository exists at this project root.
- `.gitignore` excludes virtual environments, test output, local benchmark
  caches, credentials, replay data, experimental controls, and generated
  variants.
- The release keeps the reviewed Apache-2.0 tape chassis required to reproduce
  the optional tape artifact; other downloaded public-agent material is not
  included.
- `LICENSE` and `LICENSES/THIRD_PARTY.md` describe the repository and upstream
  terms.
- `README.md`, `CONTRIBUTING.md`, `SECURITY.md`, `.editorconfig`,
  `.gitattributes`, and a GitHub Actions test workflow are present.
- Public documentation contains no workstation path or credential value.

## Before creating a remote

Run these commands locally and inspect their output:

```powershell
git status --short --ignored
git add -n .
git diff --check
```

Confirm that no ignored file was force-added and that the only third-party
source to be redistributed is covered by `LICENSES/THIRD_PARTY.md`.

## When you decide to publish

1. Create an empty GitHub repository with a name and visibility you choose.
2. Review the staged diff locally before making the first commit.
3. Add the remote and push only after that review.

Keep GitHub Actions enabled so every push and pull request runs
`.github/workflows/tests.yml`. Do not add Kaggle credentials, replays, or local
benchmark downloads to the repository.
