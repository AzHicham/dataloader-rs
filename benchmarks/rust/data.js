window.BENCHMARK_DATA = {
  "lastUpdate": 1792836585738,
  "repoUrl": "https://github.com/AzHicham/dataloader-rs",
  "entries": {
    "Rust benchmarks": [
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
        "date": 1775377546019,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 178239,
            "range": "± 13335",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41699,
            "range": "± 873",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 22581,
            "range": "± 194",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 14112,
            "range": "± 179",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 9319,
            "range": "± 46",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 8640,
            "range": "± 72",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15655,
            "range": "± 1159",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 866024,
            "range": "± 95232",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 337005,
            "range": "± 7143",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 287535,
            "range": "± 8182",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 192773,
            "range": "± 21159",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 179519,
            "range": "± 3077",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 195603,
            "range": "± 4026",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 185064,
            "range": "± 9620",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 1988937,
            "range": "± 14937",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2094890,
            "range": "± 8501",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1829398,
            "range": "± 86961",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 1769057,
            "range": "± 81048",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 1258055,
            "range": "± 88446",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 757820,
            "range": "± 36695",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1184704,
            "range": "± 24223",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 892646,
            "range": "± 56037",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 902214,
            "range": "± 42427",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 1186008,
            "range": "± 50366",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 979625,
            "range": "± 15549",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 354794,
            "range": "± 43378",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 322583,
            "range": "± 9329",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 364339,
            "range": "± 8429",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 286559,
            "range": "± 6538",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 361809,
            "range": "± 14084",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2667,
            "range": "± 107",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 5169,
            "range": "± 71",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 35894,
            "range": "± 1180",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 57434,
            "range": "± 345",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 364918,
            "range": "± 9325",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 573778,
            "range": "± 4708",
            "unit": "ns/iter"
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
        "date": 1775506570417,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 177799,
            "range": "± 4510",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41299,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 22832,
            "range": "± 141",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 12779,
            "range": "± 80",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 9038,
            "range": "± 31",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 9646,
            "range": "± 32",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15072,
            "range": "± 89",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 915912,
            "range": "± 18405",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 259357,
            "range": "± 5033",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 199748,
            "range": "± 6766",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 170626,
            "range": "± 3070",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 162965,
            "range": "± 35240",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 159134,
            "range": "± 7472",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 161153,
            "range": "± 13636",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2165540,
            "range": "± 15040",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2289747,
            "range": "± 20817",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1246583,
            "range": "± 7116",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 904920,
            "range": "± 16309",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 829984,
            "range": "± 35642",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 648042,
            "range": "± 9940",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1160718,
            "range": "± 5341",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 692363,
            "range": "± 4370",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 506378,
            "range": "± 17847",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 716215,
            "range": "± 20506",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 620699,
            "range": "± 14506",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 216203,
            "range": "± 11782",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 221266,
            "range": "± 4827",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 196330,
            "range": "± 3447",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 187840,
            "range": "± 2418",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 178617,
            "range": "± 4986",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2789,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 5511,
            "range": "± 29",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 33774,
            "range": "± 156",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 57866,
            "range": "± 880",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 336438,
            "range": "± 862",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 602811,
            "range": "± 3794",
            "unit": "ns/iter"
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
        "date": 1783344422412,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 178170,
            "range": "± 707",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41291,
            "range": "± 214",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 21823,
            "range": "± 93",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 14117,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 9951,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 9104,
            "range": "± 34",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15000,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 880074,
            "range": "± 10044",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 248060,
            "range": "± 5583",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 196797,
            "range": "± 3856",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 166794,
            "range": "± 3326",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 160246,
            "range": "± 3001",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 158624,
            "range": "± 2108",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 164286,
            "range": "± 2077",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2166156,
            "range": "± 1993",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2269265,
            "range": "± 8657",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1290989,
            "range": "± 13141",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 791831,
            "range": "± 62082",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 865311,
            "range": "± 9080",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 460632,
            "range": "± 19385",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 842523,
            "range": "± 24099",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 502067,
            "range": "± 48899",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 357755,
            "range": "± 12850",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 349910,
            "range": "± 49230",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 414159,
            "range": "± 16053",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 213883,
            "range": "± 14531",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 225616,
            "range": "± 5378",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 206964,
            "range": "± 2687",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 202382,
            "range": "± 9090",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 202233,
            "range": "± 3799",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2759,
            "range": "± 27",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 4929,
            "range": "± 9",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 34468,
            "range": "± 245",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 53819,
            "range": "± 266",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 343935,
            "range": "± 5392",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 583250,
            "range": "± 5597",
            "unit": "ns/iter"
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
        "date": 1792834247384,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 132118,
            "range": "± 964",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 29333,
            "range": "± 216",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 15825,
            "range": "± 400",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 11556,
            "range": "± 417",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 6841,
            "range": "± 259",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 6072,
            "range": "± 63",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 5325,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 1358030,
            "range": "± 18007",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 266764,
            "range": "± 4062",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 162015,
            "range": "± 4324",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 110311,
            "range": "± 1642",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 86374,
            "range": "± 1304",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 86364,
            "range": "± 1361",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 85847,
            "range": "± 3509",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2166279,
            "range": "± 1990",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2261365,
            "range": "± 5943",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1262236,
            "range": "± 8206",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 746035,
            "range": "± 11310",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 843802,
            "range": "± 7763",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 651294,
            "range": "± 13835",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1165582,
            "range": "± 3517",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 693707,
            "range": "± 6872",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 514836,
            "range": "± 23930",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 514452,
            "range": "± 11850",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 604426,
            "range": "± 17335",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 203925,
            "range": "± 10470",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 211838,
            "range": "± 4887",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 200001,
            "range": "± 11878",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 197622,
            "range": "± 1894",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 196357,
            "range": "± 1899",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 1395,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 3091,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 18963,
            "range": "± 274",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 32230,
            "range": "± 357",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 193112,
            "range": "± 2681",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 309445,
            "range": "± 4111",
            "unit": "ns/iter"
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
        "date": 1792834315746,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 176277,
            "range": "± 3295",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41366,
            "range": "± 353",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 21820,
            "range": "± 178",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 11998,
            "range": "± 84",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 8589,
            "range": "± 24",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 8688,
            "range": "± 26",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15768,
            "range": "± 51",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 903993,
            "range": "± 14615",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 271986,
            "range": "± 4027",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 211815,
            "range": "± 3308",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 178659,
            "range": "± 2682",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 170952,
            "range": "± 2757",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 170759,
            "range": "± 2424",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 175759,
            "range": "± 2713",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2165753,
            "range": "± 34050",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2266076,
            "range": "± 6433",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1268040,
            "range": "± 19819",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 750453,
            "range": "± 11288",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 858331,
            "range": "± 39151",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 545860,
            "range": "± 2794",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 991529,
            "range": "± 2990",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 581302,
            "range": "± 7191",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 429335,
            "range": "± 13458",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 419161,
            "range": "± 7602",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 497397,
            "range": "± 16519",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 196486,
            "range": "± 15711",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 199077,
            "range": "± 5806",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 193401,
            "range": "± 10656",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 193230,
            "range": "± 3592",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 176957,
            "range": "± 7083",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2269,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 4599,
            "range": "± 41",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 28398,
            "range": "± 409",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 48274,
            "range": "± 523",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 301496,
            "range": "± 3163",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 515001,
            "range": "± 8345",
            "unit": "ns/iter"
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
        "date": 1792834674123,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 177242,
            "range": "± 3372",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41584,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 24606,
            "range": "± 190",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 12651,
            "range": "± 81",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 8784,
            "range": "± 40",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 9739,
            "range": "± 30",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15769,
            "range": "± 217",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 905683,
            "range": "± 7290",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 262906,
            "range": "± 2445",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 206815,
            "range": "± 2890",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 173484,
            "range": "± 2272",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 168551,
            "range": "± 3797",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 168227,
            "range": "± 2563",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 174737,
            "range": "± 2400",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2165898,
            "range": "± 43988",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2265384,
            "range": "± 28580",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1263856,
            "range": "± 20090",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 746377,
            "range": "± 20239",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 856319,
            "range": "± 8638",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 545796,
            "range": "± 2295",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 994181,
            "range": "± 2119",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 581416,
            "range": "± 20717",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 431083,
            "range": "± 39642",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 418799,
            "range": "± 7033",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 496890,
            "range": "± 15882",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 106363,
            "range": "± 3655",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 104802,
            "range": "± 1838",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 98644,
            "range": "± 1201",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 94002,
            "range": "± 1230",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 95970,
            "range": "± 995",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 1427,
            "range": "± 111",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 3219,
            "range": "± 56",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 20736,
            "range": "± 357",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 33849,
            "range": "± 480",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 197894,
            "range": "± 4846",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 329591,
            "range": "± 14385",
            "unit": "ns/iter"
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
        "date": 1792835805676,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 97184,
            "range": "± 11230",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 23397,
            "range": "± 541",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 11588,
            "range": "± 392",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 8547,
            "range": "± 1453",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 3933,
            "range": "± 42",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 3098,
            "range": "± 116",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 2086,
            "range": "± 28",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 611040,
            "range": "± 22286",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 171334,
            "range": "± 2310",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 128427,
            "range": "± 4524",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 109158,
            "range": "± 8356",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 93173,
            "range": "± 5070",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 90070,
            "range": "± 2030",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 86537,
            "range": "± 5811",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2166181,
            "range": "± 5284",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2311722,
            "range": "± 19462",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1301150,
            "range": "± 16683",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 757303,
            "range": "± 13954",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 866710,
            "range": "± 17725",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 661372,
            "range": "± 11333",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1187731,
            "range": "± 34770",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 706491,
            "range": "± 6880",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 552160,
            "range": "± 58525",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 561427,
            "range": "± 17976",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 643878,
            "range": "± 15685",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 117967,
            "range": "± 4479",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 113280,
            "range": "± 3197",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 110628,
            "range": "± 19681",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 100499,
            "range": "± 13805",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 104581,
            "range": "± 2358",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2615,
            "range": "± 21",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 4757,
            "range": "± 331",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 32845,
            "range": "± 599",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 54338,
            "range": "± 195",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 358068,
            "range": "± 11215",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 563696,
            "range": "± 2411",
            "unit": "ns/iter"
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
        "date": 1792835862599,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 177711,
            "range": "± 989",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 41285,
            "range": "± 766",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 21853,
            "range": "± 318",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 12818,
            "range": "± 729",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 8795,
            "range": "± 77",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 8971,
            "range": "± 51",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 15745,
            "range": "± 38",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 900714,
            "range": "± 8625",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 259074,
            "range": "± 3936",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 209060,
            "range": "± 2892",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 175704,
            "range": "± 9634",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 169160,
            "range": "± 2131",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 166876,
            "range": "± 2530",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 176091,
            "range": "± 2668",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 1562962,
            "range": "± 35615",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 1636564,
            "range": "± 46502",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 889213,
            "range": "± 19800",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 490622,
            "range": "± 19573",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 591081,
            "range": "± 16097",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 661494,
            "range": "± 3200",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1200690,
            "range": "± 12409",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 715831,
            "range": "± 17516",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 551360,
            "range": "± 17771",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 558307,
            "range": "± 15519",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 648440,
            "range": "± 17417",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 157021,
            "range": "± 5459",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 161557,
            "range": "± 4911",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 145343,
            "range": "± 3574",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 135740,
            "range": "± 5142",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 128551,
            "range": "± 2331",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2054,
            "range": "± 19",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 3855,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 22673,
            "range": "± 136",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 43232,
            "range": "± 147",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 242409,
            "range": "± 1210",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 444026,
            "range": "± 1298",
            "unit": "ns/iter"
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
        "date": 1792836237482,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 126256,
            "range": "± 1049",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 34467,
            "range": "± 431",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 17329,
            "range": "± 146",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 9564,
            "range": "± 254",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 5799,
            "range": "± 58",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 5136,
            "range": "± 74",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 3841,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 1047244,
            "range": "± 15553",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 261870,
            "range": "± 10790",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 155578,
            "range": "± 1807",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 127745,
            "range": "± 2165",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 102174,
            "range": "± 1587",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 101363,
            "range": "± 1637",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 100093,
            "range": "± 1938",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2159653,
            "range": "± 3282",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2267500,
            "range": "± 36716",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1263251,
            "range": "± 7160",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 739291,
            "range": "± 9898",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 810009,
            "range": "± 6851",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 652148,
            "range": "± 13537",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 1177406,
            "range": "± 13437",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 688483,
            "range": "± 8007",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 541980,
            "range": "± 9579",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 537091,
            "range": "± 9691",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 632310,
            "range": "± 15151",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 193084,
            "range": "± 7876",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 194768,
            "range": "± 3070",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 195454,
            "range": "± 17303",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 181252,
            "range": "± 4679",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 181706,
            "range": "± 6857",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 1456,
            "range": "± 284",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 2623,
            "range": "± 175",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 14995,
            "range": "± 330",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 29301,
            "range": "± 1211",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 141499,
            "range": "± 3707",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 285166,
            "range": "± 6589",
            "unit": "ns/iter"
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
        "date": 1792836585358,
        "tool": "cargo",
        "benches": [
          {
            "name": "batch_size/sequential/bs/1",
            "value": 153569,
            "range": "± 1214",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/8",
            "value": 44243,
            "range": "± 2515",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/32",
            "value": 21556,
            "range": "± 316",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/128",
            "value": 11477,
            "range": "± 135",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/512",
            "value": 6803,
            "range": "± 54",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/1024",
            "value": 5515,
            "range": "± 15",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/sequential/bs/4096",
            "value": 4632,
            "range": "± 16",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1",
            "value": 1117512,
            "range": "± 59615",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/8",
            "value": 312571,
            "range": "± 2673",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/32",
            "value": 198935,
            "range": "± 2926",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/128",
            "value": 171403,
            "range": "± 1772",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/512",
            "value": 140629,
            "range": "± 11075",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/1024",
            "value": 138431,
            "range": "± 9869",
            "unit": "ns/iter"
          },
          {
            "name": "batch_size/parallel/bs/4096",
            "value": 134986,
            "range": "± 1891",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/0",
            "value": 2157908,
            "range": "± 3394",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/1",
            "value": 2262319,
            "range": "± 15613",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/2",
            "value": 1262122,
            "range": "± 6700",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/4",
            "value": 744069,
            "range": "± 68619",
            "unit": "ns/iter"
          },
          {
            "name": "inter_workers/num_workers/8",
            "value": 808930,
            "range": "± 7859",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/0",
            "value": 431864,
            "range": "± 10335",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/1",
            "value": 821474,
            "range": "± 61794",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/2",
            "value": 480765,
            "range": "± 11608",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/4",
            "value": 336678,
            "range": "± 16744",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/intra_workers/8",
            "value": 321989,
            "range": "± 13579",
            "unit": "ns/iter"
          },
          {
            "name": "intra_workers/inter4_intra4",
            "value": 372328,
            "range": "± 15913",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/1",
            "value": 192637,
            "range": "± 14626",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/2",
            "value": 197658,
            "range": "± 3476",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/4",
            "value": 190614,
            "range": "± 6711",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/8",
            "value": 169814,
            "range": "± 5398",
            "unit": "ns/iter"
          },
          {
            "name": "prefetch_depth/depth/16",
            "value": 180380,
            "range": "± 2746",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/1000",
            "value": 2794,
            "range": "± 18",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/1000",
            "value": 4831,
            "range": "± 138",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/10000",
            "value": 27271,
            "range": "± 153",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/10000",
            "value": 50297,
            "range": "± 974",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/sequential/100000",
            "value": 285848,
            "range": "± 5204",
            "unit": "ns/iter"
          },
          {
            "name": "sampler/random/100000",
            "value": 537919,
            "range": "± 14324",
            "unit": "ns/iter"
          }
        ]
      }
    ]
  }
}