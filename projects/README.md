# Projects

Methods that live next to GraspPanda instead of inside it: experimental ideas,
third-party contributions, and reproductions that have not passed the model-zoo gate.

Every project states a **support level** at the top of its README:

| Level | Meaning |
|---|---|
| `example` | Teaching material. Not a working method. |
| `experimental` | Runs, results not verified. |
| `community` | Maintained by its author; results are the author's claim. |
| `verified` | Trained and evaluated here; candidate for promotion into `configs/`. |

A project needs no fork: it registers its components with the public registries and is
loaded through `custom_imports`. Start from [`example_project/`](example_project/).
