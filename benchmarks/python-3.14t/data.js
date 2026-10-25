window.BENCHMARK_DATA = {
  "lastUpdate": 1792939793624,
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
          "id": "bd3cd1d56260fc3ee511491348c07133560e7edd",
          "message": "docs(bench): re-measure the real-world workloads on dev\n\nThe recorded numbers came from a build that included the unmerged\nsingle-pool and async work. Re-run both benchmarks on dev (i9-13900H,\nCPython 3.13 GIL, torch 2.14.1, 4 workers) and update the conclusions:\narray samples are about 7x torch, 4 threads now beat w=0 by 1.5-1.8x,\nremote is capped by workers in both loaders, and torch keeps its 1.6x on\nGIL-bound Python. The glibc allocator effect on num_workers=0 (4.2x)\nreproduces. List the new benchmarks in the README.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T19:49:53+01:00",
          "tree_id": "44631d6e44fba9654175f55394aeede728beaf93",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/bd3cd1d56260fc3ee511491348c07133560e7edd"
        },
        "date": 1792867883778,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1197.4206601106325,
            "unit": "iter/sec",
            "range": "stddev: 0.000015991343752023862",
            "extra": "mean: 835.1283999957104 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4409.038882474806,
            "unit": "iter/sec",
            "range": "stddev: 0.000005516552475035636",
            "extra": "mean: 226.80679999780295 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4360.539904451345,
            "unit": "iter/sec",
            "range": "stddev: 0.000016961357608387007",
            "extra": "mean: 229.3294000082824 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.2805899975540167,
            "unit": "iter/sec",
            "range": "stddev: 0.11738078945865639",
            "extra": "mean: 438.48302460000355 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 306.8407255459503,
            "unit": "iter/sec",
            "range": "stddev: 0.00019535422284773865",
            "extra": "mean: 3.2590197999979864 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1243.041762986865,
            "unit": "iter/sec",
            "range": "stddev: 0.00002382528293914392",
            "extra": "mean: 804.4781999899442 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 7874.350563417344,
            "unit": "iter/sec",
            "range": "stddev: 0.000002898724508603318",
            "extra": "mean: 126.99459999225836 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 7277.19414652138,
            "unit": "iter/sec",
            "range": "stddev: 0.000036473079470855835",
            "extra": "mean: 137.4156000053972 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 288.3161494981679,
            "unit": "iter/sec",
            "range": "stddev: 0.00042573111019756987",
            "extra": "mean: 3.4684148000053483 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 1532.803841578061,
            "unit": "iter/sec",
            "range": "stddev: 0.000041410506556000954",
            "extra": "mean: 652.3991999983991 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 21.02493302234258,
            "unit": "iter/sec",
            "range": "stddev: 0.002244376755203165",
            "extra": "mean: 47.56257720000008 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 20.888188239639106,
            "unit": "iter/sec",
            "range": "stddev: 0.003110687722214218",
            "extra": "mean: 47.87394620000214 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 17.051220570702707,
            "unit": "iter/sec",
            "range": "stddev: 0.011542699191186662",
            "extra": "mean: 58.646827999996276 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 21.23449848697106,
            "unit": "iter/sec",
            "range": "stddev: 0.00923997742728291",
            "extra": "mean: 47.09317719999717 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 33.12432634228305,
            "unit": "iter/sec",
            "range": "stddev: 0.005755459552607491",
            "extra": "mean: 30.18929320001007 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 32.50534251308544,
            "unit": "iter/sec",
            "range": "stddev: 0.0038370596296508244",
            "extra": "mean: 30.764173599999367 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 62.63510156526057,
            "unit": "iter/sec",
            "range": "stddev: 0.0014573048150562685",
            "extra": "mean: 15.965488599999844 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.08719537281036,
            "unit": "iter/sec",
            "range": "stddev: 0.0010618484912737912",
            "extra": "mean: 323.9185989999953 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1748.4327049045608,
            "unit": "iter/sec",
            "range": "stddev: 0.000016504735537146462",
            "extra": "mean: 571.9408000061321 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1693.072388000728,
            "unit": "iter/sec",
            "range": "stddev: 0.000012034064384190173",
            "extra": "mean: 590.6421999952727 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.896919024923109,
            "unit": "iter/sec",
            "range": "stddev: 0.00004002658070058202",
            "extra": "mean: 169.5800800000029 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.675407563900103,
            "unit": "iter/sec",
            "range": "stddev: 0.001780189921332664",
            "extra": "mean: 176.1987996000073 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 14.60266184855151,
            "unit": "iter/sec",
            "range": "stddev: 0.0005779890050053166",
            "extra": "mean: 68.4806653999999 msec\nrounds: 5"
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
          "id": "fec83049242b4c80679dd7353f2e3d6e0ff11b83",
          "message": "docs: restructure the README\n\nBadges (docs.rs, MSRV), what the library offers, contents, Rust and\nPython quick starts, a concept map linking the module guides, every\nbuilder setting with rules of thumb, feature flags, a performance summary\nwith the benchmark dashboards, a documentation index, project layout,\nbuild and test commands, and the MSRV.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T20:31:25+01:00",
          "tree_id": "31dd5c55675ad5971245446ebf7a704eb7ec1175",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/fec83049242b4c80679dd7353f2e3d6e0ff11b83"
        },
        "date": 1792870355375,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1205.8406095745659,
            "unit": "iter/sec",
            "range": "stddev: 0.000013691283625834615",
            "extra": "mean: 829.2970000013611 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4399.894050595268,
            "unit": "iter/sec",
            "range": "stddev: 0.000005229011779159528",
            "extra": "mean: 227.27819999772692 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4395.55416079674,
            "unit": "iter/sec",
            "range": "stddev: 0.000020559393829939144",
            "extra": "mean: 227.50259999497757 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.216105227125948,
            "unit": "iter/sec",
            "range": "stddev: 0.13035208676611407",
            "extra": "mean: 451.24211059999766 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 267.14476318249257,
            "unit": "iter/sec",
            "range": "stddev: 0.00020396533202534788",
            "extra": "mean: 3.74328880000121 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1253.0445850805695,
            "unit": "iter/sec",
            "range": "stddev: 0.000055433687706754034",
            "extra": "mean: 798.0562000000191 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 7474.958887478292,
            "unit": "iter/sec",
            "range": "stddev: 0.000017260819295663943",
            "extra": "mean: 133.78000000443535 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 7645.938018474557,
            "unit": "iter/sec",
            "range": "stddev: 0.00001558705688998592",
            "extra": "mean: 130.78840000844139 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 256.9559121724487,
            "unit": "iter/sec",
            "range": "stddev: 0.00031188307263139184",
            "extra": "mean: 3.8917182000034245 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 1623.6270203688114,
            "unit": "iter/sec",
            "range": "stddev: 0.00005508639635148411",
            "extra": "mean: 615.9050000121624 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 20.858167415096442,
            "unit": "iter/sec",
            "range": "stddev: 0.0006065385644283851",
            "extra": "mean: 47.942850399994086 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 21.41172248297048,
            "unit": "iter/sec",
            "range": "stddev: 0.0006447954184516849",
            "extra": "mean: 46.703388800005996 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 20.810018828737114,
            "unit": "iter/sec",
            "range": "stddev: 0.0030944576023589935",
            "extra": "mean: 48.053776800003334 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 23.254907120274492,
            "unit": "iter/sec",
            "range": "stddev: 0.00913162012922195",
            "extra": "mean: 43.00167679999731 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 41.88400030098691,
            "unit": "iter/sec",
            "range": "stddev: 0.003991842833969209",
            "extra": "mean: 23.875465400004714 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 34.94008627741683,
            "unit": "iter/sec",
            "range": "stddev: 0.0029797120779127646",
            "extra": "mean: 28.620421599998735 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 60.75040799367668,
            "unit": "iter/sec",
            "range": "stddev: 0.004431014746931929",
            "extra": "mean: 16.460794799996847 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.0890868554227606,
            "unit": "iter/sec",
            "range": "stddev: 0.00013501086935697177",
            "extra": "mean: 323.72025999998755 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1763.2803220917956,
            "unit": "iter/sec",
            "range": "stddev: 0.000013423929845499988",
            "extra": "mean: 567.1247999941897 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1682.9576971437084,
            "unit": "iter/sec",
            "range": "stddev: 0.000013818509028288352",
            "extra": "mean: 594.1920000111622 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.903012505741235,
            "unit": "iter/sec",
            "range": "stddev: 0.00005031658185400116",
            "extra": "mean: 169.4050281999921 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.6763078672541285,
            "unit": "iter/sec",
            "range": "stddev: 0.0010101161084357846",
            "extra": "mean: 176.17085319999433 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 21.68015682627424,
            "unit": "iter/sec",
            "range": "stddev: 0.0005212593063119116",
            "extra": "mean: 46.125127600004134 msec\nrounds: 5"
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
          "id": "ebe5c6eba38e193535521ab11d91b01c9929ce80",
          "message": "docs: rewrite BENCH.md from a run calibrated for 4-vCPU runners\n\nOne reference run on 4 pinned performance cores, the size of the\nGitHub-hosted runners: the Rust suites, the Python guards and the\ncomparison with torch on CPython 3.13, 3.14t and 3.15t, throughput and\nfirst-batch latency. The document now says what each benchmark answers,\nhow a number is produced, how to choose the settings, and which\nbottlenecks remain (glibc malloc on array samples, the per-batch cost of\nthe Python parallel path).\n\nWith torch collating its own batches, array samples are at parity to\n1.6×, not ≈ 7×: the README and the Python README now quote the fair\nnumbers (≈ 5× on loader overhead, ≈ 3× on network-bound datasets,\nfirst batches in milliseconds). bench/ and benches/ READMEs describe the\ncurrent suites.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-24T22:02:05+01:00",
          "tree_id": "9ee9f9df3c002c10fc9bdda83148e9621380f7a2",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/ebe5c6eba38e193535521ab11d91b01c9929ce80"
        },
        "date": 1792875808636,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1219.4440847077594,
            "unit": "iter/sec",
            "range": "stddev: 0.000028156879744019552",
            "extra": "mean: 820.0458000004573 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 5205.7575678732865,
            "unit": "iter/sec",
            "range": "stddev: 0.000004155606037026389",
            "extra": "mean: 192.0949999998811 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 5465.872730362447,
            "unit": "iter/sec",
            "range": "stddev: 0.000022604858107054854",
            "extra": "mean: 182.95340000236138 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.03189085863079,
            "unit": "iter/sec",
            "range": "stddev: 0.17860087353944143",
            "extra": "mean: 492.15241840000203 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 340.535848141168,
            "unit": "iter/sec",
            "range": "stddev: 0.0005515669438352197",
            "extra": "mean: 2.9365483999953312 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 995.9383641628373,
            "unit": "iter/sec",
            "range": "stddev: 0.00004969122239416237",
            "extra": "mean: 1.0040782000004356 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 8246.63372428219,
            "unit": "iter/sec",
            "range": "stddev: 0.000008272461049293364",
            "extra": "mean: 121.26159999752417 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 10421.83415948815,
            "unit": "iter/sec",
            "range": "stddev: 0.0000019598033767997607",
            "extra": "mean: 95.95239999953264 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 347.1462357683962,
            "unit": "iter/sec",
            "range": "stddev: 0.00013052586540439852",
            "extra": "mean: 2.8806303999999727 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 1135.9575315435893,
            "unit": "iter/sec",
            "range": "stddev: 0.000057509850657279095",
            "extra": "mean: 880.3146000019524 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-1]",
            "value": 43.184549418094335,
            "unit": "iter/sec",
            "range": "stddev: 0.00014819186655786216",
            "extra": "mean: 23.15643009999775 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-1]",
            "value": 41.69911135524328,
            "unit": "iter/sec",
            "range": "stddev: 0.00013925496348180617",
            "extra": "mean: 23.981326399999148 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-4]",
            "value": 85.37476563882778,
            "unit": "iter/sec",
            "range": "stddev: 0.0004936973088005321",
            "extra": "mean: 11.713062899995919 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-4]",
            "value": 85.17241443333825,
            "unit": "iter/sec",
            "range": "stddev: 0.0003192736602611405",
            "extra": "mean: 11.740890600003695 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_sequential",
            "value": 6.330417757726263,
            "unit": "iter/sec",
            "range": "stddev: 0.00007661369116204389",
            "extra": "mean: 157.96745780000094 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[1]",
            "value": 13.952001932296547,
            "unit": "iter/sec",
            "range": "stddev: 0.0001748658209709446",
            "extra": "mean: 71.67430199999956 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[2]",
            "value": 13.93110543993095,
            "unit": "iter/sec",
            "range": "stddev: 0.00010122017646123305",
            "extra": "mean: 71.78181260000258 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[4]",
            "value": 13.924898405403685,
            "unit": "iter/sec",
            "range": "stddev: 0.00020816566741557762",
            "extra": "mean: 71.81380939999826 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 16.48526060597371,
            "unit": "iter/sec",
            "range": "stddev: 0.010953542607226632",
            "extra": "mean: 60.66024819999711 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 29.357735937508576,
            "unit": "iter/sec",
            "range": "stddev: 0.005536855280630447",
            "extra": "mean: 34.062572200002705 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 24.193014072672856,
            "unit": "iter/sec",
            "range": "stddev: 0.004211735820717396",
            "extra": "mean: 41.33424620000312 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 47.34925292869279,
            "unit": "iter/sec",
            "range": "stddev: 0.0028494148176815355",
            "extra": "mean: 21.1196574000013 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.0939657860332335,
            "unit": "iter/sec",
            "range": "stddev: 0.0002009777538529987",
            "extra": "mean: 323.2097796666646 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_items_in_parallel",
            "value": 23.267906952106134,
            "unit": "iter/sec",
            "range": "stddev: 0.00006474256956584639",
            "extra": "mean: 42.977651666665416 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2023.3944870443072,
            "unit": "iter/sec",
            "range": "stddev: 0.00002839757744625192",
            "extra": "mean: 494.2190000036817 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1910.4700252961886,
            "unit": "iter/sec",
            "range": "stddev: 0.000022643662192626904",
            "extra": "mean: 523.4314000006179 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.39097265843566,
            "unit": "iter/sec",
            "range": "stddev: 0.00015698106101646507",
            "extra": "mean: 185.49528320000377 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.2182230183105895,
            "unit": "iter/sec",
            "range": "stddev: 0.000387815843923289",
            "extra": "mean: 191.63611760000094 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[2]",
            "value": 10.334746212334496,
            "unit": "iter/sec",
            "range": "stddev: 0.0006238972319852321",
            "extra": "mean: 96.76096340000129 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 13.431055075483098,
            "unit": "iter/sec",
            "range": "stddev: 0.0005446212835817146",
            "extra": "mean: 74.45431460000407 msec\nrounds: 5"
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
          "id": "77877b7574bcb3f93d54fcdfe77f99897a9d0783",
          "message": "docs: say plainly when PyTorch is faster, and why\n\nThe workers are threads in one process. On regular CPython only one\nthread runs Python code at a time, so a dataset whose __getitem__ is\npure Python gets no faster with more workers, while PyTorch's worker\nprocesses each run their own interpreter (2.4-2.6x faster in the\nbenchmarks); on free-threaded builds, threads that share one dict or\nlist contend on its reference count (2.5-2.9x). The README, the Python\nREADME (the PyPI page) and BENCH.md's summary now say so up front, with\nwhat to do about it: work that releases the GIL, free-threaded CPython,\nper-thread tables.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-25T11:48:16+01:00",
          "tree_id": "114d4382390fadfb6a7cd6fe3360cf0ca6a9ebe7",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/77877b7574bcb3f93d54fcdfe77f99897a9d0783"
        },
        "date": 1792925363349,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1918.4784777163745,
            "unit": "iter/sec",
            "range": "stddev: 0.000006087600739007566",
            "extra": "mean: 521.246400006703 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7607.802562051995,
            "unit": "iter/sec",
            "range": "stddev: 0.000008419180930500754",
            "extra": "mean: 131.44400000442147 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 7902.452130962009,
            "unit": "iter/sec",
            "range": "stddev: 0.00001273073460966303",
            "extra": "mean: 126.54299999894647 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1048.0299552128695,
            "unit": "iter/sec",
            "range": "stddev: 0.00003791395091592563",
            "extra": "mean: 954.1711999986546 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 4346.563693614321,
            "unit": "iter/sec",
            "range": "stddev: 0.000005739499911227113",
            "extra": "mean: 230.06680000321467 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 4267.201944468536,
            "unit": "iter/sec",
            "range": "stddev: 0.00005817016540289504",
            "extra": "mean: 234.3456000005517 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 13605.294091126014,
            "unit": "iter/sec",
            "range": "stddev: 0.00000221408552209478",
            "extra": "mean: 73.50080000492198 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 13994.234375012946,
            "unit": "iter/sec",
            "range": "stddev: 0.00002189544854077277",
            "extra": "mean: 71.45800000216695 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 16328.0234582648,
            "unit": "iter/sec",
            "range": "stddev: 0.000006049728446275095",
            "extra": "mean: 61.244400006899006 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 8751.632179199305,
            "unit": "iter/sec",
            "range": "stddev: 0.000024456548354736677",
            "extra": "mean: 114.26440000263938 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-1]",
            "value": 63.765694841763796,
            "unit": "iter/sec",
            "range": "stddev: 0.00026816052571728533",
            "extra": "mean: 15.682413600001155 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-1]",
            "value": 64.44264736814245,
            "unit": "iter/sec",
            "range": "stddev: 0.00020458078810393432",
            "extra": "mean: 15.517674099999113 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-4]",
            "value": 186.50676521575903,
            "unit": "iter/sec",
            "range": "stddev: 0.00019570233631067044",
            "extra": "mean: 5.361735800002521 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-4]",
            "value": 179.67781396197418,
            "unit": "iter/sec",
            "range": "stddev: 0.00037837822158250957",
            "extra": "mean: 5.565517400003728 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_sequential",
            "value": 7.628487821870863,
            "unit": "iter/sec",
            "range": "stddev: 0.00960318024104815",
            "extra": "mean: 131.08757899999546 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[1]",
            "value": 14.3761936948972,
            "unit": "iter/sec",
            "range": "stddev: 0.0005764081615915019",
            "extra": "mean: 69.55944119999913 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[2]",
            "value": 14.464526623559232,
            "unit": "iter/sec",
            "range": "stddev: 0.00017228745712827192",
            "extra": "mean: 69.13465100000167 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[4]",
            "value": 14.514872395474786,
            "unit": "iter/sec",
            "range": "stddev: 0.0000302799248269839",
            "extra": "mean: 68.89485300000047 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 60.41010268109403,
            "unit": "iter/sec",
            "range": "stddev: 0.0001685406906075307",
            "extra": "mean: 16.553522600003134 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 128.34954777440012,
            "unit": "iter/sec",
            "range": "stddev: 0.000415571799341659",
            "extra": "mean: 7.791223400005265 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 56.100690956324975,
            "unit": "iter/sec",
            "range": "stddev: 0.00018065218307141277",
            "extra": "mean: 17.82509239999399 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 107.36408083138875,
            "unit": "iter/sec",
            "range": "stddev: 0.0002759392166210761",
            "extra": "mean: 9.314102000001867 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.0965966223117025,
            "unit": "iter/sec",
            "range": "stddev: 0.00009535983917510723",
            "extra": "mean: 322.9351839999974 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_items_in_parallel",
            "value": 24.644187224848032,
            "unit": "iter/sec",
            "range": "stddev: 0.000022209967925169814",
            "extra": "mean: 40.57751999999937 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2258.2743171106354,
            "unit": "iter/sec",
            "range": "stddev: 0.0002615375849761435",
            "extra": "mean: 442.8159999974923 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2903.114926245356,
            "unit": "iter/sec",
            "range": "stddev: 0.000013170325155924164",
            "extra": "mean: 344.45759999357506 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 7.844073363541147,
            "unit": "iter/sec",
            "range": "stddev: 0.002221099537280744",
            "extra": "mean: 127.48478419999856 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 7.908003500234048,
            "unit": "iter/sec",
            "range": "stddev: 0.002587306684079764",
            "extra": "mean: 126.45416760000217 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[2]",
            "value": 15.362165541305124,
            "unit": "iter/sec",
            "range": "stddev: 0.0006635015095875616",
            "extra": "mean: 65.09498920000851 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 22.736676720114346,
            "unit": "iter/sec",
            "range": "stddev: 0.0005244932762982666",
            "extra": "mean: 43.98180139999681 msec\nrounds: 5"
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
          "distinct": false,
          "id": "b574f4b90d233a58bb8882eaefba89b259a8cb26",
          "message": "build: target x86-64-v3 for Linux x86-64\n\n`+avx2,+fma` already made the Linux x86-64 wheels require AVX2, without\nsaying so. They now target the whole x86-64-v3 level (AVX2, FMA, BMI1/2,\nLZCNT, MOVBE, F16C: Intel Haswell 2013+, AMD Zen 2017+), and the Python\nREADME states the requirement next to the wheel table. pip still\ninstalls the wheel on older CPUs, where the import crashes (SIGILL).\n\nThe other entries did nothing: `+neon` is the aarch64 baseline,\naarch64-apple-darwin already targets the Apple M1 (lse, rdm, crc, rcpc,\nfp16, dotprod and more), and x86_64-apple-darwin is not built.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-25T15:37:43+01:00",
          "tree_id": "062e977bf2b5615b979124c69deb16db9f2fe266",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/b574f4b90d233a58bb8882eaefba89b259a8cb26"
        },
        "date": 1792939788580,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1851.0538419724903,
            "unit": "iter/sec",
            "range": "stddev: 0.000006528901144727976",
            "extra": "mean: 540.2328000002399 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7475.886528110779,
            "unit": "iter/sec",
            "range": "stddev: 0.000004323668154176732",
            "extra": "mean: 133.76339999808806 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 7585.86056263689,
            "unit": "iter/sec",
            "range": "stddev: 0.000017946355626087623",
            "extra": "mean: 131.82420000248385 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1052.683437453876,
            "unit": "iter/sec",
            "range": "stddev: 0.000025035911743078425",
            "extra": "mean: 949.953200003506 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 4406.9637077815205,
            "unit": "iter/sec",
            "range": "stddev: 0.000002288371978421689",
            "extra": "mean: 226.9135999995342 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 4945.8919420810735,
            "unit": "iter/sec",
            "range": "stddev: 0.00003531703327587216",
            "extra": "mean: 202.18800000293413 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 13679.516292504928,
            "unit": "iter/sec",
            "range": "stddev: 6.327175481889082e-7",
            "extra": "mean: 73.10199999892575 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 14093.876492792633,
            "unit": "iter/sec",
            "range": "stddev: 0.00001716591107147571",
            "extra": "mean: 70.95279999873583 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 16545.6079681852,
            "unit": "iter/sec",
            "range": "stddev: 0.000007153319436427528",
            "extra": "mean: 60.43900000065605 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 9172.124080152635,
            "unit": "iter/sec",
            "range": "stddev: 0.00001551579395535413",
            "extra": "mean: 109.02600000406437 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-1]",
            "value": 61.26574076937652,
            "unit": "iter/sec",
            "range": "stddev: 0.00004976553828393481",
            "extra": "mean: 16.322335899998563 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-1]",
            "value": 61.1326035005268,
            "unit": "iter/sec",
            "range": "stddev: 0.00004230904437629743",
            "extra": "mean: 16.357883399999196 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-4]",
            "value": 184.8228468393069,
            "unit": "iter/sec",
            "range": "stddev: 0.000017320530835911242",
            "extra": "mean: 5.41058649999826 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-4]",
            "value": 174.62195089771953,
            "unit": "iter/sec",
            "range": "stddev: 0.00040959953161875295",
            "extra": "mean: 5.726656900000648 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_sequential",
            "value": 7.662257015789079,
            "unit": "iter/sec",
            "range": "stddev: 0.00013559912855924035",
            "extra": "mean: 130.5098481999977 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[1]",
            "value": 14.441853540616165,
            "unit": "iter/sec",
            "range": "stddev: 0.00004970765257196162",
            "extra": "mean: 69.24318940000376 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[2]",
            "value": 14.397173459371585,
            "unit": "iter/sec",
            "range": "stddev: 0.0005477773542311877",
            "extra": "mean: 69.45807820000027 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[4]",
            "value": 14.438897247935177,
            "unit": "iter/sec",
            "range": "stddev: 0.00003171041170193845",
            "extra": "mean: 69.25736660000155 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 58.45568702929947,
            "unit": "iter/sec",
            "range": "stddev: 0.0000393094687625091",
            "extra": "mean: 17.10697539999444 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 131.86814114234718,
            "unit": "iter/sec",
            "range": "stddev: 0.0001904027321066586",
            "extra": "mean: 7.583332800001585 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 56.70390154823942,
            "unit": "iter/sec",
            "range": "stddev: 0.00028702164325285624",
            "extra": "mean: 17.635470799999098 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 108.63635797741753,
            "unit": "iter/sec",
            "range": "stddev: 0.0005085560501459701",
            "extra": "mean: 9.205021399998259 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.0969707404986444,
            "unit": "iter/sec",
            "range": "stddev: 0.0004502255548800043",
            "extra": "mean: 322.8961730000037 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_items_in_parallel",
            "value": 23.882007788096516,
            "unit": "iter/sec",
            "range": "stddev: 0.0012079974751742848",
            "extra": "mean: 41.87252633333571 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1908.1697141374252,
            "unit": "iter/sec",
            "range": "stddev: 0.0000851939779580382",
            "extra": "mean: 524.0624000009575 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1670.886769687473,
            "unit": "iter/sec",
            "range": "stddev: 0.000007869785059314481",
            "extra": "mean: 598.484599998983 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 7.577097619913149,
            "unit": "iter/sec",
            "range": "stddev: 0.0014410669483457359",
            "extra": "mean: 131.9766551999976 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 7.6221002455148215,
            "unit": "iter/sec",
            "range": "stddev: 0.0005309685319970983",
            "extra": "mean: 131.19743480000068 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[2]",
            "value": 15.2039969069841,
            "unit": "iter/sec",
            "range": "stddev: 0.0003365825513795801",
            "extra": "mean: 65.77217859999962 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 22.884800459278196,
            "unit": "iter/sec",
            "range": "stddev: 0.0006112755315219828",
            "extra": "mean: 43.697125599999254 msec\nrounds: 5"
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
          "distinct": false,
          "id": "6f113a472656eb928f05fa26e7df646f51f63b33",
          "message": "fix(python): keep thread pools alive while the interpreter shuts down\n\nA loader freed during shutdown (module teardown, or a reference cycle\nthe last collection frees) drops its pool, which wakes the pool's\nthreads. Each then re-attaches to a finalizing interpreter, which CPython\n3.13 answers with `pthread_exit`: that unwinds through Rust frames, and\nRust aborts the process (exit code 134). 3.14+ parks such threads\ninstead. The race was not reproduced locally, but the sdist makes 3.13\ninstallable, and loaders in cycles are now collected, possibly last.\n\nThe package registers `_keep_pools_alive` with `atexit`, which runs\nbefore shutdown starts: it leaks one reference to every live pool (the\nbinding tracks them as loaders are built or rebuilt after a fork), so\ntheir threads stay parked until the process exits.\n\nCo-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>",
          "timestamp": "2026-10-25T15:38:10+01:00",
          "tree_id": "d408318fad649da97679705bbdd72a6751993502",
          "url": "https://github.com/AzHicham/dataloader-rs/commit/6f113a472656eb928f05fa26e7df646f51f63b33"
        },
        "date": 1792939793347,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 1130.5638438458022,
            "unit": "iter/sec",
            "range": "stddev: 0.000021890346301681",
            "extra": "mean: 884.5143999991478 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 4422.649627115148,
            "unit": "iter/sec",
            "range": "stddev: 0.0000060708888758847794",
            "extra": "mean: 226.10879999831468 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 4202.969818513664,
            "unit": "iter/sec",
            "range": "stddev: 0.000023678637430864852",
            "extra": "mean: 237.92699999773959 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 570.4592470768589,
            "unit": "iter/sec",
            "range": "stddev: 0.00021775107247987534",
            "extra": "mean: 1.7529735999971763 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 3358.4884115568925,
            "unit": "iter/sec",
            "range": "stddev: 0.000008469541303747683",
            "extra": "mean: 297.7529999981243 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 2530.8497935771643,
            "unit": "iter/sec",
            "range": "stddev: 0.00012226921351378675",
            "extra": "mean: 395.1241999970989 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[64]",
            "value": 6648.007924393036,
            "unit": "iter/sec",
            "range": "stddev: 0.000036825775788264054",
            "extra": "mean: 150.42100000073333 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential_getitems[4096]",
            "value": 7428.460213645095,
            "unit": "iter/sec",
            "range": "stddev: 0.000017035684937494616",
            "extra": "mean: 134.6174000047995 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[64]",
            "value": 7191.037091541502,
            "unit": "iter/sec",
            "range": "stddev: 0.0000123485602025628",
            "extra": "mean: 139.06199999667024 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel_getitems[4096]",
            "value": 4182.266520752302,
            "unit": "iter/sec",
            "range": "stddev: 0.0000927295922107619",
            "extra": "mean: 239.1048000021101 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-1]",
            "value": 46.934172331102275,
            "unit": "iter/sec",
            "range": "stddev: 0.00008448821500592318",
            "extra": "mean: 21.30643729999946 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-1]",
            "value": 46.88192819669975,
            "unit": "iter/sec",
            "range": "stddev: 0.000020970694686719723",
            "extra": "mean: 21.330180699999346 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[0-4]",
            "value": 178.7541312537006,
            "unit": "iter/sec",
            "range": "stddev: 0.000024674262823399086",
            "extra": "mean: 5.594276300001866 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_first_batch.py::test_first_batch[1-4]",
            "value": 176.63810558034146,
            "unit": "iter/sec",
            "range": "stddev: 0.00019872697635565944",
            "extra": "mean: 5.6612925999999675 msec\nrounds: 10"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_sequential",
            "value": 6.55806336233717,
            "unit": "iter/sec",
            "range": "stddev: 0.0003006606112926386",
            "extra": "mean: 152.48404060000098 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[1]",
            "value": 14.109951497916407,
            "unit": "iter/sec",
            "range": "stddev: 0.00008483407353949595",
            "extra": "mean: 70.87196580000068 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[2]",
            "value": 14.128029691332324,
            "unit": "iter/sec",
            "range": "stddev: 0.00011787961539602236",
            "extra": "mean: 70.78127819999622 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_overlap_prefetch_depth[4]",
            "value": 14.1252469202626,
            "unit": "iter/sec",
            "range": "stddev: 0.00005927248998266287",
            "extra": "mean: 70.79522260000317 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[0]",
            "value": 32.15220411011424,
            "unit": "iter/sec",
            "range": "stddev: 0.0002372677259113072",
            "extra": "mean: 31.10206679999976 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_local_files[4]",
            "value": 69.3273889996538,
            "unit": "iter/sec",
            "range": "stddev: 0.0006911874054003912",
            "extra": "mean: 14.424313599997163 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[0]",
            "value": 24.163748716808772,
            "unit": "iter/sec",
            "range": "stddev: 0.00026163549031838455",
            "extra": "mean: 41.38430719999917 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_image_augment[4]",
            "value": 50.07755811890612,
            "unit": "iter/sec",
            "range": "stddev: 0.0004918829041853851",
            "extra": "mean: 19.969024800002444 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_threads",
            "value": 3.0839527336750905,
            "unit": "iter/sec",
            "range": "stddev: 0.0002987837270472947",
            "extra": "mean: 324.2591849999978 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_real_world.py::test_remote_items_in_parallel",
            "value": 24.51746883547581,
            "unit": "iter/sec",
            "range": "stddev: 0.00007148948188071425",
            "extra": "mean: 40.78724466666964 msec\nrounds: 3"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1738.185466474526,
            "unit": "iter/sec",
            "range": "stddev: 0.00002283243990838515",
            "extra": "mean: 575.3126000001885 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1678.9154743382958,
            "unit": "iter/sec",
            "range": "stddev: 0.000011890101153551665",
            "extra": "mean: 595.6225999966591 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.879336675457056,
            "unit": "iter/sec",
            "range": "stddev: 0.0000800816276868923",
            "extra": "mean: 170.08721480000304 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.873346903051443,
            "unit": "iter/sec",
            "range": "stddev: 0.00012421451799279364",
            "extra": "mean: 170.26067359999786 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[2]",
            "value": 11.723040156448528,
            "unit": "iter/sec",
            "range": "stddev: 0.00013116373223952894",
            "extra": "mean: 85.30210480000164 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 22.43897061114513,
            "unit": "iter/sec",
            "range": "stddev: 0.0003632359382522012",
            "extra": "mean: 44.56532420000201 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}