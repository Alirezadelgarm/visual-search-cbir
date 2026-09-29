# Coursework findings

The original project report evaluated the retrieval system qualitatively with queries from multiple classes such as cars, doors, and windows.

The evaluation criteria were:

- for in-dataset queries, the first few results should contain the same image or highly similar examples;
- for external queries, most top-10 results should belong to the related class or share visual structure;
- after relevance feedback, results should contain fewer off-topic images and move toward the user's selected preference.

The report's main observations were:

- SIFT was generally the stronger descriptor because of its richer local representation and greater robustness to scale and rotation;
- Harris was useful for objects with strong corner structure, including windows, doors, and signs;
- combining the two feature families allowed the user to tune retrieval behavior through similarity weights;
- Rocchio feedback improved the ordering toward user-selected relevant results.

No numerical retrieval benchmark was reported in the coursework document, so this repository does not invent one. The refactored package provides `Precision@K` and `mAP@10` utilities for future quantitative experiments.
