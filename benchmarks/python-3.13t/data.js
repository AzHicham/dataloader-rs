window.BENCHMARK_DATA = {
  "lastUpdate": 1775377545141,
  "repoUrl": "https://github.com/AzHicham/dataloader-rs",
  "entries": {
    "Python benchmarks (3.13t)": [
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
        "date": 1775377544894,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 955.3554743955106,
            "unit": "iter/sec",
            "range": "stddev: 0.00001653915810421314",
            "extra": "mean: 1.0467308000016828 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 3493.7646781539684,
            "unit": "iter/sec",
            "range": "stddev: 0.000007643535841003046",
            "extra": "mean: 286.22420000203874 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 3659.8961029080074,
            "unit": "iter/sec",
            "range": "stddev: 0.000007891777285089045",
            "extra": "mean: 273.2317999971201 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1.3908084382667794,
            "unit": "iter/sec",
            "range": "stddev: 0.0364479506727762",
            "extra": "mean: 719.0062790000013 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 126.21197892533573,
            "unit": "iter/sec",
            "range": "stddev: 0.0011709027474306262",
            "extra": "mean: 7.923178199999371 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1400.7709282916455,
            "unit": "iter/sec",
            "range": "stddev: 0.000017576155327657636",
            "extra": "mean: 713.8925999981893 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 10.546849995725912,
            "unit": "iter/sec",
            "range": "stddev: 0.007533533476136708",
            "extra": "mean: 94.81503959999884 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 10.889664673923617,
            "unit": "iter/sec",
            "range": "stddev: 0.0014102594848273398",
            "extra": "mean: 91.83019220000403 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 11.02148029418044,
            "unit": "iter/sec",
            "range": "stddev: 0.00023165223730669307",
            "extra": "mean: 90.73191380000196 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1421.797982981902,
            "unit": "iter/sec",
            "range": "stddev: 0.000008358953658045205",
            "extra": "mean: 703.3347999993111 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1384.442521001958,
            "unit": "iter/sec",
            "range": "stddev: 0.000006228546081323522",
            "extra": "mean: 722.3123999949621 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.8730611366048375,
            "unit": "iter/sec",
            "range": "stddev: 0.0007285474706317426",
            "extra": "mean: 170.26895799999977 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.6213743730359775,
            "unit": "iter/sec",
            "range": "stddev: 0.008584997456237786",
            "extra": "mean: 177.89243939999722 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 11.021662263780586,
            "unit": "iter/sec",
            "range": "stddev: 0.00027463457873194334",
            "extra": "mean: 90.73041580000165 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}