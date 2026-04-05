window.BENCHMARK_DATA = {
  "lastUpdate": 1775377548496,
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
      }
    ]
  }
}