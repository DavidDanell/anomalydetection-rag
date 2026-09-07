# Anomaly Explanation System

An alarm in sensor data helps nobody if the engineer can't find the right
maintenance instruction. This system turns a detected anomaly into a search
against maintenance documentation and generates a short explanation grounded
only in what was retrieved.

The hard part is the retrieval, not the language model. Four retrieval
approaches are compared against a hand-built ground truth set.

---

## Repository structure

```
anomaly-rag/
├── README.md
├── LIMITATIONS.md            # what this project does not show
├── config.yaml               # seeds, thresholds, model names, k values
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # notebook tooling
│
├── src/                      # pipeline source
├── notebooks/                # exploration only, nothing load-bearing
│
├── knowledge_base/           # AUTHORED — 15-20 maintenance documents
├── eval/                     # AUTHORED — retrieval ground truth + changelog
│
├── data/                     # GENERATED — not in version control
└── results/                  # metrics tables and plots
```

### The organising principle

The layout separates **authored** artifacts from **generated** ones, because
the two have opposite properties in version control.

`data/` is derived. The sensor time series, the fault manifest, the embedding
caches and the vector index are all reproducible from the committed seed and
`config.yaml`, so committing them would add hundreds of megabytes that carry
no information the repository doesn't already contain. Clone the repository,
run the pipeline, and the same files appear with the same contents.

`knowledge_base/` and `eval/` are not reproducible. The maintenance documents
were written by hand, deliberately including near-misses — documents that are
topically adjacent but operationally wrong for a given anomaly. The ground
truth set maps anomaly events to the documents that should be retrieved for
them. Neither can be regenerated from a seed, and together they are what makes
the evaluation mean anything. They are the substance of the project.

That distinction is why they are separate top-level directories rather than
subfolders of a single `data/`. A reader should be able to tell at a glance
which files were reasoned about and which fell out of a script.

### Evaluation integrity

`eval/` was committed before any retrieval code was written, so relevance
judgments could not be influenced by seeing retrieval output. The set is split
into a development portion used while tuning and a held-out portion untouched
until the final comparison.

Where a labelling error was found later, the correction and its reason are
recorded in the changelog alongside the ground truth rather than edited
silently. `LIMITATIONS.md` covers what this evaluation can and cannot support.