## [0.2.0](https://github.com/AzHicham/dataloader-rs/compare/0.1.0...0.2.0) (2026-10-26)

### Features

* **python:** add .pyi stub file for IDE type hints and inline documentation ([6792398](https://github.com/AzHicham/dataloader-rs/commit/6792398e971b3a53ac0da20ea6b56410f646c3c6))
* **python:** add seed= for reproducible shuffling ([7f7655a](https://github.com/AzHicham/dataloader-rs/commit/7f7655a682d0a0c230761c3432d8ffe5bce07267))
* **python:** fetch the items of a batch concurrently with parallel_items ([237a03b](https://github.com/AzHicham/dataloader-rs/commit/237a03b397f45a62b9a7bcfa5790e78edac8c1f2))
* **python:** publish free-threaded 3.15t and Linux arm64 wheels ([5d9fdba](https://github.com/AzHicham/dataloader-rs/commit/5d9fdbab959f0bb26b3d8a4efb739ab5a67adadd))
* **python:** publish wheels for the GIL builds of CPython 3.13 to 3.15 ([c6597d5](https://github.com/AzHicham/dataloader-rs/commit/c6597d5220e5b569569b024c7bf8fd4f3040a590))
* **python:** support batched fetching via __getitems__ ([f4608c1](https://github.com/AzHicham/dataloader-rs/commit/f4608c1ac6e75d47210be592eb76b6d9011eeb99))
* replace intra_workers with parallel_items and num_threads ([751768d](https://github.com/AzHicham/dataloader-rs/commit/751768d9e467aec7a0897731aff48c468dc6f951))
* two parallelism settings, num_workers and items_in_parallel ([6d3ba30](https://github.com/AzHicham/dataloader-rs/commit/6d3ba30fb28dfde06039b46a90f107fd5995dfe8))

### Bug Fixes

* a huge prefetch_depth neither aborts nor wraps around ([869077d](https://github.com/AzHicham/dataloader-rs/commit/869077d5c30fb3317b84608d9846de0b2a766499))
* an epoch used in a fork()ed child raises instead of hanging ([cbeeda5](https://github.com/AzHicham/dataloader-rs/commit/cbeeda5fed42c264c98bf8e16549df968167259b))
* **collator:** pin batches for the CUDA or MPS device, not for the CPU ([679a383](https://github.com/AzHicham/dataloader-rs/commit/679a383cf8ddf4aecf8ed4db2f977a138d923afc))
* count batches from the sampler, not the dataset ([790d8ef](https://github.com/AzHicham/dataloader-rs/commit/790d8ef8bae2ab41a897dc086466ff2c7b6342c3))
* **deps:** add numpy to the dev group for the real-world benchmarks ([4900e5d](https://github.com/AzHicham/dataloader-rs/commit/4900e5da94b173f9c9ecbdf6882a85d60d044cf7))
* **deps:** update cargo ([d0ac623](https://github.com/AzHicham/dataloader-rs/commit/d0ac62344106368c4486330737ad2834c5f6d107))
* **deps:** update pyo3 to 0.30 ([aabcf53](https://github.com/AzHicham/dataloader-rs/commit/aabcf53557fde2b8bc651075cf10f484f4df389f))
* **deps:** update Python dev dependencies ([21da8dc](https://github.com/AzHicham/dataloader-rs/commit/21da8dc92806eb4bd6923bb91f262838763f9435))
* **deps:** update Rust dependencies within their semver ranges ([118964e](https://github.com/AzHicham/dataloader-rs/commit/118964ea38207ad8c78801294c38eff865820ed4))
* **deps:** update tch to 0.26 (libtorch 2.13) ([ed6f138](https://github.com/AzHicham/dataloader-rs/commit/ed6f1385ed43741850bd1d4926f90906e228321b))
* items_in_parallel never calls get_batch, even for one item ([c73f00e](https://github.com/AzHicham/dataloader-rs/commit/c73f00e0299c4a245b1893f66d9ecc95f5444810))
* pool threads get a main thread's 8 MiB stack ([dfb5c49](https://github.com/AzHicham/dataloader-rs/commit/dfb5c49a9c8abb3b580b3b07381cb91f32068b63))
* **pre-commit:** update ruff to v0.17.2 ([634564f](https://github.com/AzHicham/dataloader-rs/commit/634564fc2e6296b7f66b97da969cc3a2deaeb1de))
* **python:** an exception no longer ends an epoch with workers ([b20805d](https://github.com/AzHicham/dataloader-rs/commit/b20805d67eb5d0ab269dd3aeceb47718455139ba))
* **python:** Ctrl-C interrupts a wait for the next batch ([ecbde64](https://github.com/AzHicham/dataloader-rs/commit/ecbde6485070b934e3a2473140e26ed6ec5b8af6))
* **python:** Ctrl-C stops a sampler that never ends; samplers must be finite ([59f2904](https://github.com/AzHicham/dataloader-rs/commit/59f29048dd50cff415e0ab53c65bb3608d90c301))
* **python:** keep thread pools alive while the interpreter shuts down ([6f113a4](https://github.com/AzHicham/dataloader-rs/commit/6f113a472656eb928f05fa26e7df646f51f63b33))
* **python:** loaders and iterators work on any thread ([f84930d](https://github.com/AzHicham/dataloader-rs/commit/f84930d111b974d08d5222ffdcaaa74b9e63e81f))
* **python:** raise from len(loader) instead of guessing the sampler length ([9de80d3](https://github.com/AzHicham/dataloader-rs/commit/9de80d366f960009bd131a5384d8a25fb8cc952b))
* **python:** the garbage collector frees loaders caught in reference cycles ([4e7e1f6](https://github.com/AzHicham/dataloader-rs/commit/4e7e1f6f28366869b475437a59858584a442e48f))
* **python:** treat a falsy __getitems__ as not defined ([1798eb1](https://github.com/AzHicham/dataloader-rs/commit/1798eb1cb6ab50a6f5486b67ba2c324d6279ce6d))
* raise a failing dataset __len__ instead of panicking ([ed1032e](https://github.com/AzHicham/dataloader-rs/commit/ed1032ee6c80e74fbac261b30a7b8af94672b7a1))
* raise Python sampler exceptions instead of yielding an empty epoch ([acc3df1](https://github.com/AzHicham/dataloader-rs/commit/acc3df1d6cd641f95667663186a89c72e1df7279))
* re-raise the original Python exception type ([2613e2a](https://github.com/AzHicham/dataloader-rs/commit/2613e2ad60e14e6c528d0a8ebccb90807a7c1e51))
* re-raise worker panics instead of silently ending the epoch ([2397701](https://github.com/AzHicham/dataloader-rs/commit/23977015d3a0bbc0afc7833bc3fde07149797a81))
* rebuild the worker pool in a forked child ([f80f999](https://github.com/AzHicham/dataloader-rs/commit/f80f9994bfb96d85d6398831b54a14733e2fcc6b))
* **sampler:** seed RandomSampler::from_entropy from the OS, as documented ([ed23f84](https://github.com/AzHicham/dataloader-rs/commit/ed23f8421714cf245b41ba6f319315922f0f5b89))
* share dataset/collator via Arc and remove all unsafe code ([91cb838](https://github.com/AzHicham/dataloader-rs/commit/91cb838cff2aaabe075f068f2832935dbdfd8dcd))

### Performance Improvements

* keep out-of-order batches in a ring, not a HashMap ([fd56ced](https://github.com/AzHicham/dataloader-rs/commit/fd56ced18978c3150e9a48bc70d6902d6a3efd38))
* **python:** keep one thread state per pool thread ([b915410](https://github.com/AzHicham/dataloader-rs/commit/b915410d3de4fa4065a629f6c816a0124aa36ca5))
* **python:** keep the interpreter when the next batch is already there ([21e59cc](https://github.com/AzHicham/dataloader-rs/commit/21e59cc2ecefdae3e12a5a4f1f42006ff9c8e309))
* run batches and items on one rayon pool ([4d9194f](https://github.com/AzHicham/dataloader-rs/commit/4d9194f61bf13ac8639761ee939d55c7a1a92ba5))

### Documentation

* add a guide per module, included in the rustdoc ([2655df4](https://github.com/AzHicham/dataloader-rs/commit/2655df4b6a6082bcd059754239b0d53ff9f1fc60))
* add ARCHITECTURE.md ([718dbae](https://github.com/AzHicham/dataloader-rs/commit/718dbaeb5384f785fd55c8f70b4f4a60316aee9b))
* add the MIT LICENSE file and crates.io metadata ([4d10508](https://github.com/AzHicham/dataloader-rs/commit/4d10508c0432d4efbab217d6650f9b8a09018312))
* **bench:** re-measure the real-world workloads on dev ([bd3cd1d](https://github.com/AzHicham/dataloader-rs/commit/bd3cd1d56260fc3ee511491348c07133560e7edd))
* **bench:** turn the benchmark results into a structured BENCH.md ([4cd385f](https://github.com/AzHicham/dataloader-rs/commit/4cd385fb122ce83b3305bd1b6a9736d0a7eb35b6))
* describe the single pool and the new pool options ([0a16962](https://github.com/AzHicham/dataloader-rs/commit/0a16962269f48c054ac12a974a20316ae750124f)), closes [#50](https://github.com/AzHicham/dataloader-rs/issues/50)
* describe the single worker pool ([30f8a94](https://github.com/AzHicham/dataloader-rs/commit/30f8a944e933b8cb01b3ff72dd717bc13c0110d9))
* **python:** add PyPI README with quickstart, API reference, and PyTorch comparison ([eb83a32](https://github.com/AzHicham/dataloader-rs/commit/eb83a325a21f90159cac0c88bf5d3bf437235c0a))
* **python:** restructure the Python README and add PyPI metadata ([25ffca3](https://github.com/AzHicham/dataloader-rs/commit/25ffca3472347555b31f76634c7b73fb2788d614))
* re-measure BENCH.md on the optimised loader ([3a0f56e](https://github.com/AzHicham/dataloader-rs/commit/3a0f56edc355b4765fab0675692941de1d5be519))
* restructure the README ([fec8304](https://github.com/AzHicham/dataloader-rs/commit/fec83049242b4c80679dd7353f2e3d6e0ff11b83))
* rewrite BENCH.md from a run calibrated for 4-vCPU runners ([ebe5c6e](https://github.com/AzHicham/dataloader-rs/commit/ebe5c6eba38e193535521ab11d91b01c9929ce80))
* root causes and allocator settings; re-measure with the allocator pinned ([17685c4](https://github.com/AzHicham/dataloader-rs/commit/17685c4f074fb525e9c757d4b72fb34fb4d29040))
* say plainly when PyTorch is faster, and why ([77877b7](https://github.com/AzHicham/dataloader-rs/commit/77877b7574bcb3f93d54fcdfe77f99897a9d0783))
* threads, Ctrl-C, the garbage collector and worker stacks ([bb78b23](https://github.com/AzHicham/dataloader-rs/commit/bb78b239d96c23a82ebb4fe3d12460b1bb803586))

### CI/CD

* build release wheels and the crate without restoring caches ([a63daba](https://github.com/AzHicham/dataloader-rs/commit/a63dabafaffd9c3c1ecd7adedc199cc3ab75e89a))
* build the Linux arm64 wheels on ubuntu-26.04-arm ([08321ed](https://github.com/AzHicham/dataloader-rs/commit/08321eda98c4d6512875ada67dc88677c989e8fb))
* build the wheels on pull requests that change packaging ([476a631](https://github.com/AzHicham/dataloader-rs/commit/476a6311fbf58ed84c485076c469782ea172eb6d))
* clear stale Criterion baselines before Rust benchmarks ([0d9e7a8](https://github.com/AzHicham/dataloader-rs/commit/0d9e7a8d2404ac3098433d8ac00410366734b59e))
* drop Python 3.13t from the CI, bench and wheel matrices ([89394d9](https://github.com/AzHicham/dataloader-rs/commit/89394d92341a39be592dd9191cc0716a71bcf499))
* give CI, Publish and Benchmarks jobs a read-only token ([8757611](https://github.com/AzHicham/dataloader-rs/commit/87576116df4cbc859b3ade4d98fb55b6e6f95245))
* pin Python for the Rust test job ([9891f8c](https://github.com/AzHicham/dataloader-rs/commit/9891f8c036c0a79bccbb9c9fcb65278468dbb086))
* pin the conventionalcommits preset to 9.x for the release notes ([88961bf](https://github.com/AzHicham/dataloader-rs/commit/88961bfd9b536cb3cc392a97991d2763b38ce8c6))
* publish to crates.io in parallel with the wheel builds ([8439628](https://github.com/AzHicham/dataloader-rs/commit/84396280eb288a52e794c5856e43496d411cf0cc))
* publish to crates.io through the crates-io environment ([040be6e](https://github.com/AzHicham/dataloader-rs/commit/040be6e8da621ff7fb215e508bd9b2f55d485f3f))
* **renovate:** pin GitHub Actions to commit digests ([1c30bf6](https://github.com/AzHicham/dataloader-rs/commit/1c30bf65d6d3ef15bf8f4f92f121263051af6279))
* **renovate:** track the release tooling that npx runs ([4624c4f](https://github.com/AzHicham/dataloader-rs/commit/4624c4f569d156407ca601627d595d4564e4c5e3))
* run a real cargo publish dry run on dev ([df076cd](https://github.com/AzHicham/dataloader-rs/commit/df076cd6b0b4c4ae756d9aeaad55f582834b3b11))
* run semantic-release with npx instead of semantic-release-action ([a405254](https://github.com/AzHicham/dataloader-rs/commit/a405254b9df984a78b8eedb35f481088142c614f))
* run the crate dry run without waiting for the wheels ([a0775cd](https://github.com/AzHicham/dataloader-rs/commit/a0775cdf08d3666e38bf19fb66aa79eca0929177))
* update semantic-release plugins ([7e3a96e](https://github.com/AzHicham/dataloader-rs/commit/7e3a96e5c61bf4f18dca76dec22485aecb0f1b24))

### Miscellaneous Chores

* **deps:** pin dependencies ([ccc40de](https://github.com/AzHicham/dataloader-rs/commit/ccc40dec09cf248b42b5396b48184cbc0a355c48))
* **deps:** update actions/checkout action to v7 ([#27](https://github.com/AzHicham/dataloader-rs/issues/27)) ([f79c5bc](https://github.com/AzHicham/dataloader-rs/commit/f79c5bc26339d81fcba6897584f8d8d0d969dd7d))
* **deps:** update astral-sh/setup-uv action to v10 ([fd7d729](https://github.com/AzHicham/dataloader-rs/commit/fd7d72915561ea543c600f2809be4c447ee4ecca))
* **deps:** update cycjimmy/semantic-release-action action to v8 ([#32](https://github.com/AzHicham/dataloader-rs/issues/32)) ([b652929](https://github.com/AzHicham/dataloader-rs/commit/b6529295ff77b5e759514b2b7193be3232d1d011))
* **deps:** update dependency pytest to v9.0.3 [security] ([ce909dc](https://github.com/AzHicham/dataloader-rs/commit/ce909dc44bb74d0d1eb611eb1c423b423f42f0fb))
* **deps:** update dependency pytest to v9.1.1 ([a276b35](https://github.com/AzHicham/dataloader-rs/commit/a276b354d6bcdd37109bdc6bbd893eb45b8a4e8b))
* **deps:** update dependency pytest-benchmark to v5.3.0 ([#30](https://github.com/AzHicham/dataloader-rs/issues/30)) ([9076e74](https://github.com/AzHicham/dataloader-rs/commit/9076e74eef2e58f49f3b2c8a27ab809576f9f375))
* **deps:** update dependency python to 3.15 ([84a02f3](https://github.com/AzHicham/dataloader-rs/commit/84a02f3e24bcef188b1f0fd219c905d65ed2b714))
* **deps:** update dependency semantic-release to v25 ([740ea77](https://github.com/AzHicham/dataloader-rs/commit/740ea776ce2b88cdf45b490d85a05810e88390b5))
* **deps:** update dependency ubuntu to v26 ([4e8af13](https://github.com/AzHicham/dataloader-rs/commit/4e8af13fb92a43421a290ecc15baf7d132efaa8c))
* **deps:** update github-actions ([930cd8d](https://github.com/AzHicham/dataloader-rs/commit/930cd8d638f9e22a505d8d0b4b2b2649a694f9df))
* **deps:** update github-actions ([3b3d75b](https://github.com/AzHicham/dataloader-rs/commit/3b3d75b00df3d557e7e38ced40441381cb4df51e))
* **deps:** update j178/prek-action action to v3 ([#31](https://github.com/AzHicham/dataloader-rs/issues/31)) ([372d179](https://github.com/AzHicham/dataloader-rs/commit/372d179ef1bb5993e39b3a7bd22503e1bb23814a))
* **deps:** update pre-commit hook astral-sh/ruff-pre-commit to v0.15.20 ([433d561](https://github.com/AzHicham/dataloader-rs/commit/433d561ad8422ead37b2bdd8ecbf5c8a637a70d5))
* stop tracking a compiled pytest cache file ([4d03433](https://github.com/AzHicham/dataloader-rs/commit/4d03433183e3df486103ccc549392ec529933113))

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
