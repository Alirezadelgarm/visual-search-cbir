# Interview guide

## What problem does the project solve?

It performs content-based image retrieval: given a query image, it ranks dataset images by visual similarity instead of relying on filenames or text labels.

## Why are SIFT and Harris both used?

SIFT provides a strong 128-dimensional local descriptor with robustness to scale and rotation. Harris identifies corner-rich structures; the project converts patches around those corners into a second descriptor family. Their similarity scores are combined with adjustable weights.

## Why Bag of Visual Words?

An image can have a different number of local descriptors. BoVW converts those variable-length sets into fixed-length histograms by clustering descriptors into visual words and counting assignments.

## Why L2 normalization and cosine similarity?

Normalization reduces the effect of raw histogram magnitude. With L2-normalized vectors, the dot product corresponds to cosine similarity, making ranking simple and efficient.

## What does Rocchio feedback do?

After the first search, the user marks relevant results. The query vector is moved toward the centroid of those selected vectors and normalized again, then retrieval is repeated.

## What are the main limitations?

BoVW discards spatial relationships, the Harris patch descriptor is comparatively simple, and classical feature extraction/indexing can become expensive on large datasets. Deep visual embeddings would be a natural modern comparison.
