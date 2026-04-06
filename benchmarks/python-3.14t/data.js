window.BENCHMARK_DATA = {
  "lastUpdate": 1775506257410,
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
      }
    ]
  }
}