## [0.1.0](https://github.com/AzHicham/dataloader-rs/compare/0.0.1...0.1.0) (2026-04-05)


### Features

* **bench:** replace ad-hoc benchmarks with Criterion and pytest-benchmark suites ([8b9fac3](https://github.com/AzHicham/dataloader-rs/commit/8b9fac3d60b63dbc659d016054bb86e9fcca6a8f))
* implement core DataLoader with Dataset, Collator, and Sampler abstractions ([f78de99](https://github.com/AzHicham/dataloader-rs/commit/f78de990af7fa2c8fa7c9d8b563d35dfb8022695))
* **python:** add parallel worker path and expose collator and epoch_chunks APIs ([1c540c7](https://github.com/AzHicham/dataloader-rs/commit/1c540c75c1e7e6d3e5a536ad01fcbc3f9088824c))
* **python:** add PyO3 bindings with PyDataloader, PyDataset, and collate_fn support ([f941db4](https://github.com/AzHicham/dataloader-rs/commit/f941db40538f45c6175eb2b3cb0005de3ed8bb0e))


### Bug Fixes

* **ci:** add maturin to pyproject.toml dev dependencies ([80f7327](https://github.com/AzHicham/dataloader-rs/commit/80f7327876284fa62b7a7da24ed02d85f0d899e1))
* **ci:** fix audit workflow permissions and publish workflow setup ([e74179c](https://github.com/AzHicham/dataloader-rs/commit/e74179ca7dfe5a7c8f43771805bda2d61e9f4251))
* **ci:** fix publish workflow and pin pytest and torch versions ([5ce9468](https://github.com/AzHicham/dataloader-rs/commit/5ce946843495029ebb3d979e6b368f167bfa3e8c))
* **ci:** restrict pre-commit to commit stage and fix Criterion benchmark group names ([3478137](https://github.com/AzHicham/dataloader-rs/commit/347813795ce4bac785f0126dc0486dc5fb4ccf57))
* **release:** bump pyproject.toml version and include it in release commit assets ([9c57561](https://github.com/AzHicham/dataloader-rs/commit/9c57561cbb524809519566180a0028067f250373))
* **release:** change tag format ([53cb600](https://github.com/AzHicham/dataloader-rs/commit/53cb60084cd0ab6feab204681576afd4fba1077f))


### Performance Improvements

* **python:** optimize collator with lazy PyBatch to defer GIL acquisition ([26ca05c](https://github.com/AzHicham/dataloader-rs/commit/26ca05ca280d72960e2d355cdcaa9f83dbfb67f3))


### Documentation

* add README and rewrite bench.md with benchmark analysis and PyTorch comparison ([534d431](https://github.com/AzHicham/dataloader-rs/commit/534d43143df66025b1f1409ea1a994aa30417190))


### CI/CD

* add GitHub Actions workflows, pre-commit config, and semantic release setup ([6635e60](https://github.com/AzHicham/dataloader-rs/commit/6635e60330f97b8fabf5cb1b3ac3d16d99b68991))


### Miscellaneous Chores

* add Cargo.lock and pin dependency versions ([5900b27](https://github.com/AzHicham/dataloader-rs/commit/5900b2705dcf45591ca2a0cf8928418bfc2eaabf))
* **deps:** update actions/checkout action to v6 ([b8f38ac](https://github.com/AzHicham/dataloader-rs/commit/b8f38acb9a12a94ed6cdbf98e6fedd827da8f53f))
* **deps:** update astral-sh/setup-uv action to v8 ([#7](https://github.com/AzHicham/dataloader-rs/issues/7)) ([285defe](https://github.com/AzHicham/dataloader-rs/commit/285defeff0ca4fa5d90c9b4499f38bdb1916dce8))
* **deps:** update dependency pytest to v9.0.2 ([#2](https://github.com/AzHicham/dataloader-rs/issues/2)) ([bac8d44](https://github.com/AzHicham/dataloader-rs/commit/bac8d441ba2a3a10b2780df2a2c64bb367b8721a))
* **deps:** update dependency pytest-benchmark to v5.2.3 ([#3](https://github.com/AzHicham/dataloader-rs/issues/3)) ([487aefa](https://github.com/AzHicham/dataloader-rs/commit/487aefadf16dd47a28cf6f38d79bfce5fa3c5206))
* **deps:** update github artifact actions ([6507deb](https://github.com/AzHicham/dataloader-rs/commit/6507debec514f62dc30c9c4477d7bfe2b36ed72b))
* **deps:** update github-actions ([#4](https://github.com/AzHicham/dataloader-rs/issues/4)) ([5b2a566](https://github.com/AzHicham/dataloader-rs/commit/5b2a56649f577e673372f685626277879d7f70ca))
* **deps:** update j178/prek-action action to v2 ([#9](https://github.com/AzHicham/dataloader-rs/issues/9)) ([feaa34e](https://github.com/AzHicham/dataloader-rs/commit/feaa34e3247fabf4a5ecce1d9c2e8d34637fdffa))
* **deps:** update pre-commit hook astral-sh/ruff-pre-commit to v0.15.9 ([#5](https://github.com/AzHicham/dataloader-rs/issues/5)) ([863272b](https://github.com/AzHicham/dataloader-rs/commit/863272b7189d0ac56edf113cbd83d2beada1243f))
* **deps:** update pre-commit hook pre-commit/pre-commit-hooks to v6 ([#10](https://github.com/AzHicham/dataloader-rs/issues/10)) ([e23174d](https://github.com/AzHicham/dataloader-rs/commit/e23174d49d2fe4749beb59c4912067f239f95507))

# Changelog

All notable changes to this project will be documented in this file.
This file is maintained automatically by [semantic-release](https://github.com/semantic-release/semantic-release).
