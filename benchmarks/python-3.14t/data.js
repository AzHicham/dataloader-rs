window.BENCHMARK_DATA = {
  "lastUpdate": 1792842552385,
  "repoUrl": "https://github.com/AzHicham/dataloader-rs",
  "entries": {
    "Python benchmarks (3.14t)": [
      {
        "commit": {
          "author": {
            "email": "hicham.azimani@owkin.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani@owkin.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "21a1d470a226ef26029810d8293092bd1188f572",
          "message": "ci: improve bench",
          "timestamp": "2026-04-05T09:02:06+01:00",
          "tree_id": "d9a119d68128ee709f9dbba24037ca6ad1b65201",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/21a1d470a226ef26029810d8293092bd1188f572"
        },
        "date": 1775377547792,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1141.305228632255,
            "unit": "iter/sec",
            "range": "stddev: 0.000023038467102675795",
            "extra": "mean: 876.1897999875146 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 3989.343664608477,
            "unit": "iter/sec",
            "range": "stddev: 0.000010946884076222735",
            "extra": "mean: 250.6678000372631 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4000.4736560782285,
            "unit": "iter/sec",
            "range": "stddev: 0.000012046992213199256",
            "extra": "mean: 249.97040000016565 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1.0657573418127078,
            "unit": "iter/sec",
            "range": "stddev: 0.26761370829667047",
            "extra": "mean: 938.2998932000191 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 127.10442991428806,
            "unit": "iter/sec",
            "range": "stddev: 0.0010078585083204865",
            "extra": "mean: 7.867546400029823 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1075.4538415171623,
            "unit": "iter/sec",
            "range": "stddev: 0.000045822164347346377",
            "extra": "mean: 929.8400000034235 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 10.403140243640285,
            "unit": "iter/sec",
            "range": "stddev: 0.006025119878647131",
            "extra": "mean: 96.12482160002855 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 10.822130806433263,
            "unit": "iter/sec",
            "range": "stddev: 0.0014959961014070916",
            "extra": "mean: 92.40324459999556 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 10.866699125501249,
            "unit": "iter/sec",
            "range": "stddev: 0.0005211371837837231",
            "extra": "mean: 92.02426499996363 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1618.2858533475326,
            "unit": "iter/sec",
            "range": "stddev: 0.000015786348733889488",
            "extra": "mean: 617.9378000069846 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1456.5492426250746,
            "unit": "iter/sec",
            "range": "stddev: 0.0000685631712782858",
            "extra": "mean: 686.5541999786728 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.902053990453788,
            "unit": "iter/sec",
            "range": "stddev: 0.00016104475420198128",
            "extra": "mean: 169.43254019997767 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.719164089355554,
            "unit": "iter/sec",
            "range": "stddev: 0.00020224589185691025",
            "extra": "mean: 174.85072720000971 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 10.82582132117643,
            "unit": "iter/sec",
            "range": "stddev: 0.00015286335818634806",
            "extra": "mean: 92.37174440002036 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hicham.azimani@owkin.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "6792398e971b3a53ac0da20ea6b56410f646c3c6",
          "message": "feat(python): add .pyi stub file for IDE type hints and inline documentation",
          "timestamp": "2026-04-06T21:09:46+01:00",
          "tree_id": "3949357e76bc6cd8b7263ae4f06ee04f1c00f3e0",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/6792398e971b3a53ac0da20ea6b56410f646c3c6"
        },
        "date": 1775506257094,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1200.3072786652917,
            "unit": "iter/sec",
            "range": "stddev: 0.000024149350074753422",
            "extra": "mean: 833.1199999986438 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4090.6788972350278,
            "unit": "iter/sec",
            "range": "stddev: 0.000009311685115831032",
            "extra": "mean: 244.4582000009632 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 3792.0471669878216,
            "unit": "iter/sec",
            "range": "stddev: 0.000033074127433700835",
            "extra": "mean: 263.7098000008109 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.0310902485952,
            "unit": "iter/sec",
            "range": "stddev: 0.1431506969913562",
            "extra": "mean: 492.3464138000014 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 176.34013387886452,
            "unit": "iter/sec",
            "range": "stddev: 0.00048451355796200646",
            "extra": "mean: 5.670858799999223 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1166.8385419841213,
            "unit": "iter/sec",
            "range": "stddev: 0.00005713542331921186",
            "extra": "mean: 857.016599999838 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 14.725733160974126,
            "unit": "iter/sec",
            "range": "stddev: 0.0006036458724470182",
            "extra": "mean: 67.90833359999908 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 14.718426738182089,
            "unit": "iter/sec",
            "range": "stddev: 0.000199183778710688",
            "extra": "mean: 67.94204419999801 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 16.48653848413704,
            "unit": "iter/sec",
            "range": "stddev: 0.010146312833727895",
            "extra": "mean: 60.655546400002436 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1503.6852317581588,
            "unit": "iter/sec",
            "range": "stddev: 0.00008209781674589693",
            "extra": "mean: 665.0328000034733 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1543.9151217171764,
            "unit": "iter/sec",
            "range": "stddev: 0.000014463811758423913",
            "extra": "mean: 647.7040000021361 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.902642260079976,
            "unit": "iter/sec",
            "range": "stddev: 0.0001424989768745636",
            "extra": "mean: 169.41565419999733 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.709006951147554,
            "unit": "iter/sec",
            "range": "stddev: 0.0003505275737697834",
            "extra": "mean: 175.1618116000003 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 21.709739412003636,
            "unit": "iter/sec",
            "range": "stddev: 0.00018896378596551983",
            "extra": "mean: 46.06227560000491 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "29139614+renovate[bot]@users.noreply.github.com",
            "name": "renovate[bot]",
            "username": "renovate[bot]"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "d0ac62344106368c4486330737ad2834c5f6d107",
          "message": "fix(deps): update cargo",
          "timestamp": "2026-07-06T14:19:21+01:00",
          "tree_id": "6ff1b9647383b0a3176a98f3e91dd77cd4020f2c",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/d0ac62344106368c4486330737ad2834c5f6d107"
        },
        "date": 1783344108756,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1119.9030970183312,
            "unit": "iter/sec",
            "range": "stddev: 0.000017412423333255666",
            "extra": "mean: 892.9344000051742 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4037.660062972064,
            "unit": "iter/sec",
            "range": "stddev: 0.000010940161272096916",
            "extra": "mean: 247.6681999979746 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 3890.996079376129,
            "unit": "iter/sec",
            "range": "stddev: 0.000013429561432387305",
            "extra": "mean: 257.0036000037135 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.1874696153634696,
            "unit": "iter/sec",
            "range": "stddev: 0.13016172575458831",
            "extra": "mean: 457.14920699999766 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 262.7443213459504,
            "unit": "iter/sec",
            "range": "stddev: 0.0004259971608871121",
            "extra": "mean: 3.8059814000064307 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1178.8738265391844,
            "unit": "iter/sec",
            "range": "stddev: 0.000049787601572480215",
            "extra": "mean: 848.2672000070579 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 20.56546625025374,
            "unit": "iter/sec",
            "range": "stddev: 0.0009826363668153828",
            "extra": "mean: 48.62520440000537 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 20.969448381844735,
            "unit": "iter/sec",
            "range": "stddev: 0.0006090836444219792",
            "extra": "mean: 47.68842659999564 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 21.13951015924477,
            "unit": "iter/sec",
            "range": "stddev: 0.0007271957953833081",
            "extra": "mean: 47.304785800000104 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1586.4534642492283,
            "unit": "iter/sec",
            "range": "stddev: 0.000020371065697912214",
            "extra": "mean: 630.3367999976217 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1548.5063417505173,
            "unit": "iter/sec",
            "range": "stddev: 0.00001436716660687871",
            "extra": "mean: 645.783600000982 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.897650767302313,
            "unit": "iter/sec",
            "range": "stddev: 0.00007167660863288734",
            "extra": "mean: 169.55903960000285 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.703180552810383,
            "unit": "iter/sec",
            "range": "stddev: 0.00014064576049516194",
            "extra": "mean: 175.3407578000008 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 20.945878505875807,
            "unit": "iter/sec",
            "range": "stddev: 0.00281500680097496",
            "extra": "mean: 47.742089200005466 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "91cb838cff2aaabe075f068f2832935dbdfd8dcd",
          "message": "fix: share dataset/collator via Arc and remove all unsafe code\n\nWorkers previously held raw pointers to the loader's dataset and collator,\nrelying on DataLoaderIter's Drop to join them before the borrow ended.\nLeaking the iterator (e.g. mem::forget) from safe code let workers read\nfreed memory. The loader now stores Arc<D>/Arc<C> and each worker holds a\nclone, so the crate is #![forbid(unsafe_code)]. DataLoaderIter no longer\ncarries a lifetime, which also removes the Python-only OwnedDataLoaderIter.\n\nAlso fix a deadlock when a threaded Python iterator is dropped: PyO3 never\ninvoked __del__, so Drop joined workers while holding the GIL that workers\ninside __getitem__ were waiting for. Drop now detaches before joining.\n\nBenchmarks (interleaved A/B, 2 rounds) show no measurable change.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017eJkQ4peEFnnwbBFmW44KN",
          "timestamp": "2026-10-24T10:25:17+01:00",
          "tree_id": "ac3514abc763acf792321f43c116158921dc02a7",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/91cb838cff2aaabe075f068f2832935dbdfd8dcd"
        },
        "date": 1792834111500,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1165.9852260529378,
            "unit": "iter/sec",
            "range": "stddev: 0.00001913566156053769",
            "extra": "mean: 857.6437999863629 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4459.284944608902,
            "unit": "iter/sec",
            "range": "stddev: 0.000005594394641158145",
            "extra": "mean: 224.2512000066199 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4436.88793140591,
            "unit": "iter/sec",
            "range": "stddev: 0.00002175754620135076",
            "extra": "mean: 225.38319999512169 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1.9862511378439192,
            "unit": "iter/sec",
            "range": "stddev: 0.041334399341681566",
            "extra": "mean: 503.46100799997663 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 308.5913434686613,
            "unit": "iter/sec",
            "range": "stddev: 0.000501612410562923",
            "extra": "mean: 3.240531600010854 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1073.5651962102952,
            "unit": "iter/sec",
            "range": "stddev: 0.000019051941448061945",
            "extra": "mean: 931.4758000073198 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 12.174238351829976,
            "unit": "iter/sec",
            "range": "stddev: 0.0028367946854528686",
            "extra": "mean: 82.14066220000404 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 12.29915292634592,
            "unit": "iter/sec",
            "range": "stddev: 0.003579997071496181",
            "extra": "mean: 81.30641239998795 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 12.366899849097821,
            "unit": "iter/sec",
            "range": "stddev: 0.0012686021219664183",
            "extra": "mean: 80.8610090000002 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1788.4314239889502,
            "unit": "iter/sec",
            "range": "stddev: 0.000011559762468580185",
            "extra": "mean: 559.1492000121434 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1678.2736069385646,
            "unit": "iter/sec",
            "range": "stddev: 0.000015105907885929225",
            "extra": "mean: 595.8503999977438 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 4.66937114464323,
            "unit": "iter/sec",
            "range": "stddev: 0.0018503519915890436",
            "extra": "mean: 214.1616010000007 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 4.521289506671425,
            "unit": "iter/sec",
            "range": "stddev: 0.0014239105189641217",
            "extra": "mean: 221.17583900001136 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 12.44888084934443,
            "unit": "iter/sec",
            "range": "stddev: 0.0007819933067175839",
            "extra": "mean: 80.32850600001211 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "ed1032ee6c80e74fbac261b30a7b8af94672b7a1",
          "message": "fix: raise a failing dataset __len__ instead of panicking\n\nDataset::len is infallible, so PyDataset::len panicked when __len__\nraised; Python saw a PanicException (a BaseException, which\n`except Exception` does not catch) instead of the real error.\n\nPyDataloader.__iter__ and __len__ now call __len__ fallibly first and\npropagate its exception.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017eJkQ4peEFnnwbBFmW44KN",
          "timestamp": "2026-10-24T10:25:30+01:00",
          "tree_id": "fcc315d1c8da18eccd4a701facfe51c56aeb595a",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/ed1032ee6c80e74fbac261b30a7b8af94672b7a1"
        },
        "date": 1792834150121,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 2145.645562512364,
            "unit": "iter/sec",
            "range": "stddev: 0.00001296602270740592",
            "extra": "mean: 466.0602000029712 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7247.668787342662,
            "unit": "iter/sec",
            "range": "stddev: 0.000006385461046284307",
            "extra": "mean: 137.97539999984565 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 7718.908237538203,
            "unit": "iter/sec",
            "range": "stddev: 0.000007653748883904221",
            "extra": "mean: 129.55200000135392 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 3.7140654566212956,
            "unit": "iter/sec",
            "range": "stddev: 0.07759213554844176",
            "extra": "mean: 269.24673559999803 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 453.97736378063416,
            "unit": "iter/sec",
            "range": "stddev: 0.00015944668832869408",
            "extra": "mean: 2.202753000000257 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1342.0253632090864,
            "unit": "iter/sec",
            "range": "stddev: 0.00002909877488145935",
            "extra": "mean: 745.1423999981444 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 20.710632071799733,
            "unit": "iter/sec",
            "range": "stddev: 0.0032833163116196646",
            "extra": "mean: 48.2843785999961 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 21.940160782851667,
            "unit": "iter/sec",
            "range": "stddev: 0.0013944731913231878",
            "extra": "mean: 45.578517399999896 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 22.41994008665611,
            "unit": "iter/sec",
            "range": "stddev: 0.0008002871001462965",
            "extra": "mean: 44.60315219999984 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2967.653173936903,
            "unit": "iter/sec",
            "range": "stddev: 0.000014291394692427273",
            "extra": "mean: 336.9665999997551 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2806.0172233788503,
            "unit": "iter/sec",
            "range": "stddev: 0.000010014670291703367",
            "extra": "mean: 356.37699999426786 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 7.701227906110641,
            "unit": "iter/sec",
            "range": "stddev: 0.0003617190972984932",
            "extra": "mean: 129.84942300000455 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 7.4149953204187256,
            "unit": "iter/sec",
            "range": "stddev: 0.00019556839931993435",
            "extra": "mean: 134.8618517999995 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 22.67582529947166,
            "unit": "iter/sec",
            "range": "stddev: 0.0001772319422753632",
            "extra": "mean: 44.0998282000038 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "9de80d366f960009bd131a5384d8a25fb8cc952b",
          "message": "fix(python): raise from len(loader) instead of guessing the sampler length\n\nlen(loader) with a Python sampler fell back to the dataset length whenever\nlen(sampler) failed, which lumped two cases together: a sampler without\n__len__ got a dataset-based count that can be wrong, and an exception\nraised by __len__ was silently dropped.\n\nPyDataloader keeps the user's sampler and __len__ now returns\nlen(sampler) batched, so a sampler without __len__ raises TypeError (as in\nPyTorch) and a raising __len__ propagates. SharedPySampler keeps the\nSampler::len default, which is right for the built-in samplers.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T10:31:57+01:00",
          "tree_id": "7f22cdc4aff076272bf7601b4bc4c271812d7194",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/9de80d366f960009bd131a5384d8a25fb8cc952b"
        },
        "date": 1792834399888,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1083.997209353034,
            "unit": "iter/sec",
            "range": "stddev: 0.000036843825070670096",
            "extra": "mean: 922.5116000038724 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4122.008143403099,
            "unit": "iter/sec",
            "range": "stddev: 0.000004307090342067196",
            "extra": "mean: 242.6002000021299 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 3960.3897657543625,
            "unit": "iter/sec",
            "range": "stddev: 0.000008706790700652151",
            "extra": "mean: 252.50039999775709 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.209948377161719,
            "unit": "iter/sec",
            "range": "stddev: 0.13708556920912796",
            "extra": "mean: 452.4992575999988 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 306.93654492092054,
            "unit": "iter/sec",
            "range": "stddev: 0.000227612448369285",
            "extra": "mean: 3.258002399999782 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1175.0257213096822,
            "unit": "iter/sec",
            "range": "stddev: 0.00006123561997010297",
            "extra": "mean: 851.0452000024316 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 21.010722044596495,
            "unit": "iter/sec",
            "range": "stddev: 0.0012071663959341645",
            "extra": "mean: 47.5947470000051 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 20.96072062823305,
            "unit": "iter/sec",
            "range": "stddev: 0.003122921760456334",
            "extra": "mean: 47.708283400001505 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 21.579333034262703,
            "unit": "iter/sec",
            "range": "stddev: 0.000774767726279038",
            "extra": "mean: 46.34063519999643 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1671.8941222858896,
            "unit": "iter/sec",
            "range": "stddev: 0.000013819330301965854",
            "extra": "mean: 598.1240000011212 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1590.175008299149,
            "unit": "iter/sec",
            "range": "stddev: 0.000011630094336432961",
            "extra": "mean: 628.8616000006186 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.899531690336757,
            "unit": "iter/sec",
            "range": "stddev: 0.00024861018136980824",
            "extra": "mean: 169.50497980000137 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.73389070573582,
            "unit": "iter/sec",
            "range": "stddev: 0.00009446322060908208",
            "extra": "mean: 174.4016500000015 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 21.615734009781725,
            "unit": "iter/sec",
            "range": "stddev: 0.0006779627883768222",
            "extra": "mean: 46.262597399999095 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "7e3a96e5c61bf4f18dca76dec22485aecb0f1b24",
          "message": "ci: update semantic-release plugins\n\nsemantic-release 25 uses commit-analyzer 13 / release-notes-generator 14,\nbuilt on conventional-changelog v8 APIs, while the conventionalcommits\npreset was still pinned at 7. Bump it to 10.4.1, and changelog 7.0.0,\ngit 11.0.1, exec 7.1.0 and backmerge 4.0.1. Their majors are the ESM\nswitch and a minimum semantic-release of 22 or 24.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T10:46:37+01:00",
          "tree_id": "651ffc27c187abb34ebaabf6442ff1d97fc6e7f9",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/7e3a96e5c61bf4f18dca76dec22485aecb0f1b24"
        },
        "date": 1792835276271,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1253.2217197372481,
            "unit": "iter/sec",
            "range": "stddev: 0.000016962742663515376",
            "extra": "mean: 797.9433999992125 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4422.234997649799,
            "unit": "iter/sec",
            "range": "stddev: 0.000011068991503263135",
            "extra": "mean: 226.12999999580552 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4278.177359533162,
            "unit": "iter/sec",
            "range": "stddev: 0.000005809169933819145",
            "extra": "mean: 233.7444000005462 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.1185093512483606,
            "unit": "iter/sec",
            "range": "stddev: 0.0847614796953957",
            "extra": "mean: 472.03001459999996 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 281.74891922519856,
            "unit": "iter/sec",
            "range": "stddev: 0.0004517449804715792",
            "extra": "mean: 3.5492594000004374 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1314.176840357957,
            "unit": "iter/sec",
            "range": "stddev: 0.000020291781402202027",
            "extra": "mean: 760.9326000050487 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 20.416152730853856,
            "unit": "iter/sec",
            "range": "stddev: 0.0014247738830347161",
            "extra": "mean: 48.98082480000028 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 20.796838434670615,
            "unit": "iter/sec",
            "range": "stddev: 0.003583374657366434",
            "extra": "mean: 48.08423180000716 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 17.789032400483553,
            "unit": "iter/sec",
            "range": "stddev: 0.01174742391581746",
            "extra": "mean: 56.214412200003494 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1693.5082072495443,
            "unit": "iter/sec",
            "range": "stddev: 0.00005390214796071581",
            "extra": "mean: 590.4901999997492 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1631.9885421430608,
            "unit": "iter/sec",
            "range": "stddev: 0.000029379933464742648",
            "extra": "mean: 612.7493999969147 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.841272269755684,
            "unit": "iter/sec",
            "range": "stddev: 0.0009848931630806368",
            "extra": "mean: 171.1955809999978 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.711126965021935,
            "unit": "iter/sec",
            "range": "stddev: 0.0005008460586866392",
            "extra": "mean: 175.0967901999985 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 21.558423510960722,
            "unit": "iter/sec",
            "range": "stddev: 0.00043032850215460924",
            "extra": "mean: 46.38558100000125 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "23977015d3a0bbc0afc7833bc3fde07149797a81",
          "message": "fix: re-raise worker panics instead of silently ending the epoch\n\nA panic in Dataset::get or Collator::collate killed its worker thread; its\nbatch never arrived and, once the other workers drained the queue, the\nconsumer saw a closed channel and ended the epoch early with no error.\n\nWorkers now catch the panic, forward the payload, and stop. The consumer\nre-raises it with resume_unwind, matching num_workers(0) where the panic\nalready happens on the caller's thread. The Python iterator joins workers\nwith the GIL released before re-raising, so a panic cannot deadlock.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017eJkQ4peEFnnwbBFmW44KN",
          "timestamp": "2026-10-24T10:51:02+01:00",
          "tree_id": "ca28965d39cc5c5f40bf7abfa1a3a4458f5b8780",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/23977015d3a0bbc0afc7833bc3fde07149797a81"
        },
        "date": 1792835561334,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 2337.96281144031,
            "unit": "iter/sec",
            "range": "stddev: 0.000006530135814265543",
            "extra": "mean: 427.7227999978095 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7902.71442448235,
            "unit": "iter/sec",
            "range": "stddev: 0.000005216615152893073",
            "extra": "mean: 126.53879999788842 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 8334.805815637947,
            "unit": "iter/sec",
            "range": "stddev: 0.000013054511169678472",
            "extra": "mean: 119.9788000008084 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 3.051203519218114,
            "unit": "iter/sec",
            "range": "stddev: 0.1470891687929574",
            "extra": "mean: 327.7395275999993 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 476.66578006956325,
            "unit": "iter/sec",
            "range": "stddev: 0.0002758991424154427",
            "extra": "mean: 2.0979059999945093 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1447.5466978502361,
            "unit": "iter/sec",
            "range": "stddev: 0.000022340358696025637",
            "extra": "mean: 690.8240000029764 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 23.06186793473253,
            "unit": "iter/sec",
            "range": "stddev: 0.0011968133190660508",
            "extra": "mean: 43.36162200000899 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 22.63836394861363,
            "unit": "iter/sec",
            "range": "stddev: 0.001027250343270253",
            "extra": "mean: 44.17280340001071 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 23.240708681905264,
            "unit": "iter/sec",
            "range": "stddev: 0.0013910908133558158",
            "extra": "mean: 43.027947799998856 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 3065.773095969402,
            "unit": "iter/sec",
            "range": "stddev: 0.00001406164145770278",
            "extra": "mean: 326.1820000034277 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2913.0789672659766,
            "unit": "iter/sec",
            "range": "stddev: 0.00000848001861454054",
            "extra": "mean: 343.27939998775037 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 8.020146235207811,
            "unit": "iter/sec",
            "range": "stddev: 0.0013175295363731614",
            "extra": "mean: 124.68600580000384 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 7.771347907857378,
            "unit": "iter/sec",
            "range": "stddev: 0.001749184265633827",
            "extra": "mean: 128.67780619999394 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 23.753662096153807,
            "unit": "iter/sec",
            "range": "stddev: 0.0010234547678651682",
            "extra": "mean: 42.098771799987844 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "7f7655a682d0a0c230761c3432d8ffe5bce07267",
          "message": "feat(python): add seed= for reproducible shuffling\n\nshuffle=True always seeded from OS entropy, so runs could not be\nreproduced. PyDataloader now takes seed=; with it, two loaders yield the\nsame sequence of epoch orders (each epoch still differs from the last).\nseed without shuffle=True is rejected rather than silently ignored.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017eJkQ4peEFnnwbBFmW44KN",
          "timestamp": "2026-10-24T10:53:05+01:00",
          "tree_id": "5c3e48e6e30be81ec34c503c58c502dd23b40adc",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/7f7655a682d0a0c230761c3432d8ffe5bce07267"
        },
        "date": 1792835656610,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1491.4416603369923,
            "unit": "iter/sec",
            "range": "stddev: 0.000027058674796056375",
            "extra": "mean: 670.4921999926228 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 5344.832562394076,
            "unit": "iter/sec",
            "range": "stddev: 0.00001290559200129262",
            "extra": "mean: 187.0966000012686 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 5357.83937638629,
            "unit": "iter/sec",
            "range": "stddev: 0.000015054652542268145",
            "extra": "mean: 186.64239999566234 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.398435686520316,
            "unit": "iter/sec",
            "range": "stddev: 0.05965403187232724",
            "extra": "mean: 416.93842599999584 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 248.01485192288854,
            "unit": "iter/sec",
            "range": "stddev: 0.0003783554203231752",
            "extra": "mean: 4.032016600001498 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 989.0963991116739,
            "unit": "iter/sec",
            "range": "stddev: 0.00004444587969006815",
            "extra": "mean: 1.0110238000038407 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 10.330838320174,
            "unit": "iter/sec",
            "range": "stddev: 0.0005589038836095716",
            "extra": "mean: 96.79756559999646 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 10.335034321512074,
            "unit": "iter/sec",
            "range": "stddev: 0.0008824441044982742",
            "extra": "mean: 96.7582660000005 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 13.530492855859391,
            "unit": "iter/sec",
            "range": "stddev: 0.0009973319453967543",
            "extra": "mean: 73.90713780000624 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2156.11757394519,
            "unit": "iter/sec",
            "range": "stddev: 0.000015482489940488938",
            "extra": "mean: 463.7966000018423 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2011.899986060479,
            "unit": "iter/sec",
            "range": "stddev: 0.000008210006872299558",
            "extra": "mean: 497.0425999943018 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.415021998586426,
            "unit": "iter/sec",
            "range": "stddev: 0.00015245300944241916",
            "extra": "mean: 184.67145660000028 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.230435015949939,
            "unit": "iter/sec",
            "range": "stddev: 0.000380334216934755",
            "extra": "mean: 191.1886863999939 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 13.456255328675564,
            "unit": "iter/sec",
            "range": "stddev: 0.003360169283685084",
            "extra": "mean: 74.31488000000854 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "noreply@anthropic.com",
            "name": "Claude",
            "username": "claude"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "2613e2ad60e14e6c528d0a8ebccb90807a7c1e51",
          "message": "fix: re-raise the original Python exception type\n\nEvery error from __getitem__ or collate_fn reached Python as\nRuntimeError(str(e)), losing the exception type and traceback, so\n`except KeyError` and similar handlers never matched. A non-iterable\nsampler likewise became ValueError instead of TypeError.\n\nThese errors cross the core as boxed PyErr; into_py_err downcasts them\nback to the original exception and only wraps non-Python errors in\nRuntimeError.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_017eJkQ4peEFnnwbBFmW44KN",
          "timestamp": "2026-10-24T10:59:26+01:00",
          "tree_id": "3b2b92213cf7c257caf249e9886865bb4bc5c491",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/2613e2ad60e14e6c528d0a8ebccb90807a7c1e51"
        },
        "date": 1792836038126,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1188.4928223358202,
            "unit": "iter/sec",
            "range": "stddev: 0.000005684561603172961",
            "extra": "mean: 841.4017999996304 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4443.899326102888,
            "unit": "iter/sec",
            "range": "stddev: 0.000006292261779754531",
            "extra": "mean: 225.02760000122635 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4385.9879964319825,
            "unit": "iter/sec",
            "range": "stddev: 0.000017449216998247172",
            "extra": "mean: 227.99879999979566 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1.8851243930416377,
            "unit": "iter/sec",
            "range": "stddev: 0.1562180970903964",
            "extra": "mean: 530.4689726000021 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 284.4882551298651,
            "unit": "iter/sec",
            "range": "stddev: 0.00022265604605555087",
            "extra": "mean: 3.5150836000013896 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1257.7227320002871,
            "unit": "iter/sec",
            "range": "stddev: 0.00005489005699579178",
            "extra": "mean: 795.0878000031025 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 21.3690451448175,
            "unit": "iter/sec",
            "range": "stddev: 0.0010507453826862103",
            "extra": "mean: 46.79666279999992 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 20.460939293465934,
            "unit": "iter/sec",
            "range": "stddev: 0.0026180639764398746",
            "extra": "mean: 48.87361159999841 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 19.96654389930941,
            "unit": "iter/sec",
            "range": "stddev: 0.004715175093601906",
            "extra": "mean: 50.08378040000139 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1766.4929497528271,
            "unit": "iter/sec",
            "range": "stddev: 0.000014729796620320628",
            "extra": "mean: 566.0933999990903 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1681.5398869607625,
            "unit": "iter/sec",
            "range": "stddev: 0.000011892683906444503",
            "extra": "mean: 594.6930000021666 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.9028128892663325,
            "unit": "iter/sec",
            "range": "stddev: 0.00006748621701361205",
            "extra": "mean: 169.41075700000567 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.737711116642107,
            "unit": "iter/sec",
            "range": "stddev: 0.00008482017313010154",
            "extra": "mean: 174.28552599999705 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 21.69595595408535,
            "unit": "iter/sec",
            "range": "stddev: 0.00024333388458529193",
            "extra": "mean: 46.09153899999967 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "31b67746b1330b6043c98f8b863c01ba153ad4d3",
          "message": "test(python): expect ValueError for a wrong-length __getitems__\n\nNow that worker errors keep their Python type, the threaded path raises\nthe same ValueError as num_workers=0.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T11:05:33+01:00",
          "tree_id": "5c1b1651992f27605787adb582d854782c0b323f",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/31b67746b1330b6043c98f8b863c01ba153ad4d3"
        },
        "date": 1792836402756,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1511.2840020001063,
            "unit": "iter/sec",
            "range": "stddev: 0.000006303142985420071",
            "extra": "mean: 661.689000000365 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 5532.485092701203,
            "unit": "iter/sec",
            "range": "stddev: 0.0000050079235519823",
            "extra": "mean: 180.75060000057874 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 5366.396057298874,
            "unit": "iter/sec",
            "range": "stddev: 0.000009733125325074586",
            "extra": "mean: 186.34479999661835 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.7792257327683236,
            "unit": "iter/sec",
            "range": "stddev: 0.10611344650015354",
            "extra": "mean: 359.8124428000034 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 422.3705700306463,
            "unit": "iter/sec",
            "range": "stddev: 0.00015631985406129066",
            "extra": "mean: 2.3675891999943133 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1520.67479031795,
            "unit": "iter/sec",
            "range": "stddev: 0.00004226050788520292",
            "extra": "mean: 657.602799998358 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 9642.067181674545,
            "unit": "iter/sec",
            "range": "stddev: 0.000010755135570926875",
            "extra": "mean: 103.71220000422454 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 9226.845691991775,
            "unit": "iter/sec",
            "range": "stddev: 0.000034624021066352255",
            "extra": "mean: 108.37940000101298 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 342.1971755176774,
            "unit": "iter/sec",
            "range": "stddev: 0.0008800085264764995",
            "extra": "mean: 2.922291800004473 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 1997.6579458395677,
            "unit": "iter/sec",
            "range": "stddev: 0.0000071781088190084964",
            "extra": "mean: 500.58619999617804 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 22.686800197574044,
            "unit": "iter/sec",
            "range": "stddev: 0.004085649649896452",
            "extra": "mean: 44.07849459999795 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 23.572708785037673,
            "unit": "iter/sec",
            "range": "stddev: 0.0004611293001283028",
            "extra": "mean: 42.4219383999997 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 23.422487173858936,
            "unit": "iter/sec",
            "range": "stddev: 0.0015334578091153364",
            "extra": "mean: 42.69401419999781 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2251.769440421137,
            "unit": "iter/sec",
            "range": "stddev: 0.000013479190374888711",
            "extra": "mean: 444.09520000101566 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2115.479811979133,
            "unit": "iter/sec",
            "range": "stddev: 0.000011641411923928562",
            "extra": "mean: 472.7059999993344 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 6.723624461745356,
            "unit": "iter/sec",
            "range": "stddev: 0.00007690253432272675",
            "extra": "mean: 148.72930599999847 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 6.562579104508258,
            "unit": "iter/sec",
            "range": "stddev: 0.00003788160480362426",
            "extra": "mean: 152.37911559999873 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 24.19187161725979,
            "unit": "iter/sec",
            "range": "stddev: 0.000228866157499871",
            "extra": "mean: 41.336198199999785 msec\nrounds: 5"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "committer": {
            "email": "hicham.azimani-ext@wearewaiv.com",
            "name": "Hicham Azimani",
            "username": "AzHicham"
          },
          "distinct": true,
          "id": "37a0c7de5173a2148ce040d31cfa858d13d3c6c0",
          "message": "refactor(python): leave no silent path from sampler errors to an empty epoch\n\nPySampler still implemented Sampler, and its Python branch turned any\nexception into an empty Vec (unwrap_or_default); only a comment kept\ncallers on SharedPySampler, which records the error. Drop that impl and\nlet SharedPySampler::indices handle every variant itself.\n\nTests: a negative index raises OverflowError, and a loader recovers after\na failed epoch with num_workers=2 as well as 0.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T12:48:17+01:00",
          "tree_id": "c1e52f0e6d0093f3324ca74a3eef99cf13cf666f",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/37a0c7de5173a2148ce040d31cfa858d13d3c6c0"
        },
        "date": 1792842551865,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 2265.4756909688163,
            "unit": "iter/sec",
            "range": "stddev: 0.000006633395382808782",
            "extra": "mean: 441.4084000046614 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7647.282843841225,
            "unit": "iter/sec",
            "range": "stddev: 0.00000814602322543722",
            "extra": "mean: 130.76540000156456 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 8592.246357279133,
            "unit": "iter/sec",
            "range": "stddev: 0.000002287900018083654",
            "extra": "mean: 116.38399999469584 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 3.646816473268095,
            "unit": "iter/sec",
            "range": "stddev: 0.059141125864532046",
            "extra": "mean: 274.21176999999943 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 510.6693117968481,
            "unit": "iter/sec",
            "range": "stddev: 0.00020447215242286958",
            "extra": "mean: 1.958214400002589 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1385.0826049400587,
            "unit": "iter/sec",
            "range": "stddev: 0.000036880907327091436",
            "extra": "mean: 721.9786000007389 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 13107.362405332578,
            "unit": "iter/sec",
            "range": "stddev: 0.000010101933900399847",
            "extra": "mean: 76.29300000076 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 12959.377534877753,
            "unit": "iter/sec",
            "range": "stddev: 0.00004863353611993628",
            "extra": "mean: 77.16420000178914 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 512.8093099099833,
            "unit": "iter/sec",
            "range": "stddev: 0.0001664708372351884",
            "extra": "mean: 1.9500425999979143 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 1564.0728707790176,
            "unit": "iter/sec",
            "range": "stddev: 0.000035384075060437235",
            "extra": "mean: 639.3564000006791 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 22.577588956885364,
            "unit": "iter/sec",
            "range": "stddev: 0.0004927842154251092",
            "extra": "mean: 44.29170900000088 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 22.366240532525207,
            "unit": "iter/sec",
            "range": "stddev: 0.003843738582329698",
            "extra": "mean: 44.710240800003476 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 22.99877991012659,
            "unit": "iter/sec",
            "range": "stddev: 0.0001710506497287095",
            "extra": "mean: 43.480567399998904 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 3089.053075495016,
            "unit": "iter/sec",
            "range": "stddev: 0.000007367974795637294",
            "extra": "mean: 323.7237999996978 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2879.8890436558668,
            "unit": "iter/sec",
            "range": "stddev: 0.000010660334461265957",
            "extra": "mean: 347.2355999974752 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 7.457591796455822,
            "unit": "iter/sec",
            "range": "stddev: 0.009987156903573371",
            "extra": "mean: 134.09154419999822 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 7.532214074442051,
            "unit": "iter/sec",
            "range": "stddev: 0.0012678434242006974",
            "extra": "mean: 132.76308800000152 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 17.202835526866597,
            "unit": "iter/sec",
            "range": "stddev: 0.002347159357616896",
            "extra": "mean: 58.12995179999518 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}