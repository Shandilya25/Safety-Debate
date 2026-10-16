# Original implementation (unmodified)

These files are the original `safe-debates` sources, moved here byte-for-byte so the
refactor in `safe_debates/` can be checked against them (`tests/test_legacy_parity.py`,
`benchmarks/`). They were written for PyTorch 0.4 and **do not run on current PyTorch**
(`SparseMNIST.__getitem__` and `Agent.apply_action` both raise). See the top-level README,
"Changes from the original implementation", for the list of defects.

They can be deleted once the refactor has been reviewed.
