window.BENCHMARK_DATA = {
  "lastUpdate": 1775377546413,
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
      }
    ]
  }
}