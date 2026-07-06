window.BENCHMARK_DATA = {
  "lastUpdate": 1783344422865,
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
      }
    ]
  }
}