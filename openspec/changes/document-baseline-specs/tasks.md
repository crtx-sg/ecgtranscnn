## 1. Baseline Spec Documentation
- [x] 1.1 Populate `openspec/project.md` with project context, tech stack, conventions, and domain knowledge
- [x] 1.2 Update `openspec/config.yaml` with project-specific context and rules
- [x] 1.3 Write `model-architecture` delta spec (CNN backbone, transformer, classification head, loss function)
- [x] 1.4 Write `ecg-simulator` delta spec (conditions, morphology, noise pipeline, HDF5 schema)
- [x] 1.5 Write `data-pipeline` delta spec (generation, caching, normalization, augmentation)
- [x] 1.6 Write `inference-pipeline` delta spec (directory watcher, processing, reporting)
- [x] 1.7 Write `trained-models` delta spec (checkpoint inventory, performance baselines)

## 2. Validation
- [ ] 2.1 Run `openspec validate document-baseline-specs --strict` and resolve all issues
- [ ] 2.2 Request user approval of the baseline proposal
