# Architecture

## Indexing path

```text
class-folder dataset
        ↓
recursive image discovery
        ↓
resize + grayscale
        ↓
┌──────────────────────┬─────────────────────────┐
│ SIFT local features  │ Harris corner patches   │
└──────────┬───────────┴────────────┬────────────┘
           ↓                        ↓
    descriptor sampling     descriptor sampling
           ↓                        ↓
 MiniBatch K-Means          MiniBatch K-Means
           ↓                        ↓
  SIFT visual words         Harris visual words
           ↓                        ↓
    BoVW histogram             BoVW histogram
           ↓                        ↓
        L2 normalization + persisted index artifact
```

## Query path

```text
query image
   ↓
SIFT + Harris
   ↓
dual BoVW histograms
   ↓
weighted cosine similarity against indexed images
   ↓
Top-10 retrieval
```

## Feedback path

```text
Top-5 results
   ↓
user selects relevant images
   ↓
centroid of selected indexed vectors
   ↓
Rocchio update of SIFT and Harris query vectors
   ↓
repeat similarity search
```

The default coursework weights are SIFT `0.60` and Harris `0.40`; the GUI normalizes any user-adjusted pair before scoring.
