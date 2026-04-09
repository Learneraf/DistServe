# Local Large Files

These local files were intentionally not added to Git because they are too large
for a normal GitHub push or are machine-local generated artifacts.

Current large local files on this machine:

- `simdistserve/dataset/raw/ShareGPT_V3_unfiltered_cleaned_split.json` (`641.67 MB`)
- `simdistserve/dataset/sharegpt.ds` (`198.01 MB`)
- `simdistserve/dataset/raw/longbench.zip` (`108.65 MB`)
- `SwiftTransformer/build/src/csrc/kernel/libxformers_autogen_impl.a` (`82.00 MB`)
- `simdistserve/dataset/longbench.ds` (`54.77 MB`)

Recommended handling:

- Keep them local and regenerate from scripts when possible.
- If they must be versioned, use releases, external object storage, or Git LFS.
- For Conda, store environment export files instead of the full environment directory.
