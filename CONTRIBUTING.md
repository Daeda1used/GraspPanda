# Contributing to GraspPanda

Thank you for helping. This page is short on purpose; the reasons behind the rules are
in [`docs/design/ARCHITECTURE.md`](docs/design/ARCHITECTURE.md).

## Set up

```bash
git clone https://github.com/Daeda1used/GraspPanda.git && cd GraspPanda
pip install -e ".[dev]"
pre-commit install
pytest                      # the whole suite runs on CPU in seconds
```

## Ground rules

1. **Only advertise what is verified.** A method enters `configs/` and the model zoo
   after it has been trained and evaluated here. Until then it lives in `projects/`
   with a stated support level.
2. **Know where code comes from.** A component reaches the toolbox by one of three
   routes, and its `source` line says which:
   - *written here*: from the paper and from the *behaviour* of reference code. The ported
     grasp detection methods and both evaluators are written this way, because most grasp
     repositories are under non-commercial terms;
   - *bridge*: a thin adapter around a library the user installs as an optional extra
     (`timm`, `transformers`, `torchvision`, ...). The library's own code runs; the
     component's status is `bridge`;
   - *adapted*: source taken from a repository under the MIT, BSD or Apache-2.0 licence and
     fitted to a slot contract. The file keeps the upstream copyright notice and names the
     repository, the commit and the licence; `THIRD_PARTY_NOTICES.md` lists it.
     A file that follows an upstream in structure and parameter names (to load its
     weights) without taking its text keeps the copyright line too and says so. An e-mail
     address inside such a copyright line stays as the licence requires; no other e-mail
     address belongs in a public file.

   Never copy from a repository without a licence file or under non-commercial or
   research-only terms. Released weights are never committed; their licences are stated
   where the weights are documented.
3. **Every number states its provenance** (`reported`, `author-ckpt`, `reproduced`) and
   whether a collision filter was applied.
4. **Fail loudly.** A mismatch between components is an error with an explanation, never
   a silent interpolation, projection or squeeze.
5. **One owner per constant.** Do not repeat a value such as the number of angle bins in
   two components; inject it.
6. **Small core.** New heavy dependencies are optional extras, imported inside the
   component that needs them.

## Adding a component

See [Add a component](docs/advanced_guides/add_component.md). In short: annotate every
constructor parameter, take no `**kwargs`, subclass the slot's base class, and add a
test, including one for the configuration it must *reject*.

A component of the `grasppanda` package also states `source` (and `status`, `extras`)
on its class, gets one tiny configuration in `tests/unit/test_swap_matrix.py` when it
is a stage of the seed-based skeleton, and is followed by
`python -m tools.make_component_docs`, which regenerates the
[catalogue](docs/components.md) and the tables of
[Swapping components](docs/advanced_guides/swapping.md). The unit suite fails until all
three are done.

## Tests

- Unit tests live in `tests/unit/` and run on CPU.
- An operator is tested against a brute-force oracle written in the test.
- A bug fix comes with a test that fails without it.

## Pull requests

- One logical change per pull request.
- `pre-commit run --all-files` and `pytest` pass.
- User-visible changes get a line in `CHANGELOG.md`.
- English for code, comments, docs and commit messages. If you change `README.md`, change
  `README_zh-CN.md` in the same pull request; a test checks that they stay in step.

## Reporting bugs

Use the issue form and include the output of `grasppanda env`.
