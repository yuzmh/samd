# Contributing to SAMD

Contributions are welcome. Use this guide to prepare changes for review.

## Contribution rules

- For bugs or feature ideas, [open an issue](https://github.com/yuzmh/samd/issues)
  first for substantial changes; small fixes can go directly to a pull request.
- Keep each change focused and avoid unrelated edits.
- Add or update tests when behavior changes. Keep integration tests in the
  crate's `tests/` directory.
- Document public API changes and non-obvious behavior.
- Explain every nontrivial `unsafe` block with an accurate `// SAFETY:` comment.
- Do not add dependencies or public API without a clear need.

## Commit messages

Use the Conventional Commits format:

```text
<type>(<scope>): <short summary>
```

The scope is optional. Use a lowercase type and an imperative summary without a
period. Keep the subject concise. Add a `!` after the type or scope for a
breaking change and explain it in the commit body.

Common types include `feat`, `fix`, `docs`, `test`, `refactor`, `perf`, `build`,
and `chore`.

Examples:

```text
feat(core): add a chunk reduction
fix(x86): handle empty masks
docs: clarify contribution checks
```

## Validation

Run these commands from the workspace root before opening a pull request:

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets --all-features -- \
    -D warnings \
    -W clippy::pedantic \
    -W clippy::nursery \
    -W clippy::cargo
cargo test --workspace
cargo check --workspace
cargo check -p samd-core --no-default-features
cargo doc --workspace --no-deps
```

## Pull requests

Describe what changed, why it changed, and which validation commands passed.
Keep the pull request focused and ready for review.
