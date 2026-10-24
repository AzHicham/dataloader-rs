window.BENCHMARK_DATA = {
  "lastUpdate": 1792834117402,
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
        "date": 1775506258626,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 990.3859276417078,
            "unit": "iter/sec",
            "range": "stddev: 0.000026849743933005815",
            "extra": "mean: 1.009707400004345 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 3447.679056949222,
            "unit": "iter/sec",
            "range": "stddev: 0.000012232747066039184",
            "extra": "mean: 290.05019999885917 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 2862.90793571519,
            "unit": "iter/sec",
            "range": "stddev: 0.00003043739418433536",
            "extra": "mean: 349.29520000446246 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 1.8404503817072504,
            "unit": "iter/sec",
            "range": "stddev: 0.1528648483999521",
            "extra": "mean: 543.345264800007 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 284.34866469304706,
            "unit": "iter/sec",
            "range": "stddev: 0.00024807406301074277",
            "extra": "mean: 3.5168091999992157 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1404.6419483404215,
            "unit": "iter/sec",
            "range": "stddev: 0.00014372297925741107",
            "extra": "mean: 711.9251999995413 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 22.089313571352804,
            "unit": "iter/sec",
            "range": "stddev: 0.00031387999827841965",
            "extra": "mean: 45.270759400006 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 21.141287456629,
            "unit": "iter/sec",
            "range": "stddev: 0.0019784959480660495",
            "extra": "mean: 47.30080899999507 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 22.119171068671502,
            "unit": "iter/sec",
            "range": "stddev: 0.00016631471870865404",
            "extra": "mean: 45.209650799995416 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1434.2243212885708,
            "unit": "iter/sec",
            "range": "stddev: 0.000013924507431238149",
            "extra": "mean: 697.2410000003038 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1378.1428202339787,
            "unit": "iter/sec",
            "range": "stddev: 0.00001776285910934796",
            "extra": "mean: 725.6142000073851 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.886275243371913,
            "unit": "iter/sec",
            "range": "stddev: 0.00007502713107780737",
            "extra": "mean: 169.88672100001168 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.769588119766234,
            "unit": "iter/sec",
            "range": "stddev: 0.00045406500283461256",
            "extra": "mean: 173.32259760000284 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 20.99682246686909,
            "unit": "iter/sec",
            "range": "stddev: 0.0052658849544902275",
            "extra": "mean: 47.626253999999335 msec\nrounds: 5"
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
        "date": 1783344104471,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 994.5906205316037,
            "unit": "iter/sec",
            "range": "stddev: 0.00003329567906886781",
            "extra": "mean: 1.005438800001457 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 3329.7860013350714,
            "unit": "iter/sec",
            "range": "stddev: 0.00001124324448018664",
            "extra": "mean: 300.3195999980335 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 3630.140914746912,
            "unit": "iter/sec",
            "range": "stddev: 0.000010233318784768617",
            "extra": "mean: 275.47140000478976 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 2.386367124093414,
            "unit": "iter/sec",
            "range": "stddev: 0.11836929102986973",
            "extra": "mean: 419.04700660000174 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 316.7611501824862,
            "unit": "iter/sec",
            "range": "stddev: 0.00028236789953844514",
            "extra": "mean: 3.1569528000005675 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1397.873331416322,
            "unit": "iter/sec",
            "range": "stddev: 0.0001039784505519476",
            "extra": "mean: 715.3724000062311 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 20.57370407482881,
            "unit": "iter/sec",
            "range": "stddev: 0.0015238172096689493",
            "extra": "mean: 48.60573459999671 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 21.464294791688946,
            "unit": "iter/sec",
            "range": "stddev: 0.0006485152162237503",
            "extra": "mean: 46.588998600000764 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 21.469583178434604,
            "unit": "iter/sec",
            "range": "stddev: 0.000704035164053658",
            "extra": "mean: 46.577522799998405 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 1344.66581827346,
            "unit": "iter/sec",
            "range": "stddev: 0.000018291615654309672",
            "extra": "mean: 743.679199999292 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 1302.4459152807171,
            "unit": "iter/sec",
            "range": "stddev: 0.000011490859786148219",
            "extra": "mean: 767.7862000008417 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 5.8994900505578345,
            "unit": "iter/sec",
            "range": "stddev: 0.00006350189297468026",
            "extra": "mean: 169.50617619999946 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 5.8156109645461616,
            "unit": "iter/sec",
            "range": "stddev: 0.0002117528432895796",
            "extra": "mean: 171.95097920000535 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 22.060414961257276,
            "unit": "iter/sec",
            "range": "stddev: 0.00041098291203622573",
            "extra": "mean: 45.33006300000295 msec\nrounds: 5"
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
        "date": 1792834117052,
        "tool": "pytest",
        "benches": [
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[1]",
            "value": 2059.8493343752557,
            "unit": "iter/sec",
            "range": "stddev: 0.000008922198011625216",
            "extra": "mean: 485.47240000118563 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[64]",
            "value": 7236.37174117941,
            "unit": "iter/sec",
            "range": "stddev: 0.0000033262890250925084",
            "extra": "mean: 138.19079999848327 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_sequential[4096]",
            "value": 8141.360063088889,
            "unit": "iter/sec",
            "range": "stddev: 0.0000046388246667529435",
            "extra": "mean: 122.82959999936337 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[1]",
            "value": 4.097614705132802,
            "unit": "iter/sec",
            "range": "stddev: 0.07090343215015225",
            "extra": "mean: 244.04441900000222 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[64]",
            "value": 554.2013952345549,
            "unit": "iter/sec",
            "range": "stddev: 0.0001563100562322626",
            "extra": "mean: 1.8043982000023107 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_batch_size.py::test_batch_size_parallel[4096]",
            "value": 1995.6072692830478,
            "unit": "iter/sec",
            "range": "stddev: 0.000018179645056550692",
            "extra": "mean: 501.1005999989493 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[1]",
            "value": 23.104506516437247,
            "unit": "iter/sec",
            "range": "stddev: 0.0012794089611222275",
            "extra": "mean: 43.2815995999988 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[4]",
            "value": 23.879682430216494,
            "unit": "iter/sec",
            "range": "stddev: 0.001509382895634063",
            "extra": "mean: 41.87660380000011 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_prefetch.py::test_prefetch_depth[16]",
            "value": 24.241909902555978,
            "unit": "iter/sec",
            "range": "stddev: 0.0006262487081641054",
            "extra": "mean: 41.25087520000079 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_sequential",
            "value": 2906.1876804166363,
            "unit": "iter/sec",
            "range": "stddev: 0.000010342750900985343",
            "extra": "mean: 344.09340000252087 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_sampler.py::test_sampler_shuffle",
            "value": 2785.004865375753,
            "unit": "iter/sec",
            "range": "stddev: 0.000007252842354003465",
            "extra": "mean: 359.06580000357735 usec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[0]",
            "value": 8.183556156373655,
            "unit": "iter/sec",
            "range": "stddev: 0.0013250827971775768",
            "extra": "mean: 122.19626539999524 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[1]",
            "value": 8.065010449132048,
            "unit": "iter/sec",
            "range": "stddev: 0.000052578579685153866",
            "extra": "mean: 123.99239980000516 msec\nrounds: 5"
          },
          {
            "name": "bench/test_bench_workers.py::test_num_workers[4]",
            "value": 24.584874559113096,
            "unit": "iter/sec",
            "range": "stddev: 0.0003987571083741361",
            "extra": "mean: 40.6754160000105 msec\nrounds: 5"
          }
        ]
      }
    ]
  }
}