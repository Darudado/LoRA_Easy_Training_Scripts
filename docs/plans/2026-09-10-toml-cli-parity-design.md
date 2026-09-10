# TOML CLI Parity Design

## Understanding summary

- The saved-TOML CLI must accept the same combined TOML files that the UI saves without requiring manual edits.
- CLI-launched training must receive the same normalized arguments and dataset configuration as UI-launched training.
- The failures to prevent include UI-only subset keys, nested argument tables, missing network modules, and unqualified custom optimizer names.
- Validation must happen before expensive model loading and report actionable configuration errors.
- Existing UI TOML structure and UI-launched training behavior must remain compatible.
- Standard LoRA, SDXL, Flux, Anima, textual inversion, LyCORIS, custom optimizers, schedulers, and derived warmup settings are in scope.
- Existing backend/submodule work must be preserved while the feature branch is fixed and merged into `refresh`.

## Assumptions and non-functional requirements

- Backend validation behavior is authoritative.
- Saved source TOMLs are not modified by the CLI.
- Canonicalization is local, deterministic, and inexpensive relative to training.
- The CLI remains usable without launching the UI or HTTP backend.
- Pure transformations should be shared or independently testable; filesystem validation and other side effects stay at the boundary.
- Errors return before subprocess launch and identify the invalid field or unsupported combination.

## Architecture

The CLI loader first reconstructs the grouped `args` and `dataset` mappings used by the UI/backend validation request. A canonicalization layer then applies the UI's semantic transformations: omit empty/false optional arguments, serialize nested network/optimizer/scheduler mappings as `key=value` arrays, derive the network module, qualify registered custom optimizers, calculate derived scheduler values, and remove UI-only dataset metadata. The resulting flat training config and sd-scripts dataset config are written only after successful normalization.

Pure conversion helpers live at the top-level CLI boundary so unit tests do not import the heavyweight backend runtime. Rules that must match backend behavior are expressed as narrow helpers with representative parity tests. CLI errors wrap conversion failures with concise messages and never start training when normalization fails.

## Error handling and edge cases

- Reject malformed combined TOMLs and non-mapping argument sections.
- Reject conflicting model-family selections and CLI flags.
- Ignore known presentation-only subset fields such as `name`; preserve valid backend subset fields.
- Reject unknown subset fields before the trainer schema does.
- Preserve list values and encode booleans in nested `key=value` arguments consistently with the UI.
- Resolve known custom optimizers without importing the entire optimizer package; preserve already-qualified optimizer paths.
- Ensure Anima LoRA and LyCORIS choose the correct module.
- Preserve resolution schedules and Unicode paths.

## Decision log

- Work on `feat/toml-training-cli`, then merge it into `refresh`.
- Preserve the newer dirty backend submodule state already present in the worktree.
- Keep saved TOMLs immutable and generate normalized runtime files.
- Prefer a pure canonicalization boundary over invoking the HTTP backend or duplicating the whole validation service.
- Cover observed failures with end-to-end fixture tests and cover individual transformations with focused unit tests.
- Do not create commits unless explicitly requested.

