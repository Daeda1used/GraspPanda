---
hide:
  - navigation
  - toc
---

# Combination explorer

Swap any stage of a grasp detector and see what happened when it was built and trained.

<div id="gp-explorer" class="gpx">
<noscript><p>The explorer needs JavaScript. The same records are on the page
<a href="../advanced_guides/measured_combinations/">Measured combinations</a>.</p></noscript>
</div>
<link rel="stylesheet" href="explorer.css">
<script type="application/json" id="gp-explorer-data">
--8<-- "docs/explorer/explorer.json"
</script>
<script type="application/json" id="gp-explorer-contracts">
--8<-- "docs/explorer/explorer_contracts.json"
</script>
<script src="explorer_logic.js"></script>
<script src="explorer.js"></script>

## What this page knows, and what it does not

Every stage of the pipeline is a slot of the config. Pick a method, replace the option of
a slot, and the page answers from the
[measured records](../advanced_guides/measured_combinations.md): whether the combination
was built, trained on real frames and asked to predict, the command that composes it, and
what the contract rules said when they refused it.

- **It reads records, it does not predict.** A verdict is the outcome of
  [`grasppanda compose` followed by `grasppanda probe`](../user_guides/combinations.md) for
  exactly the combination shown. Nothing is inferred from a neighbouring cell. The counts
  under the tool say how many records stand on a probe of their own, how many share one,
  and how many are refusals, for which nothing was built.
- **Where there is no record, it may hold the answer of `compose`.** For one option alone
  and for two options of two slots, `python -m tools.make_explorer_contracts` asked the
  real `grasppanda compose` and stored what it said: accepted, or refused with the
  conflict and the fix. The page shows that as a *contract check* with a dashed outline,
  apart from the measured verdicts: "accepted" says that the contracts fit, not that the
  combination trains. A stage together with an option of the training recipe (optimizer,
  schedule, hooks, augmentation) is not stored. Any other selection is *not measured*,
  and the page prints the two commands that find out.
- **No accuracy.** A probe trains 20 iterations on 8 frames. It shows that a combination
  works: it builds, its losses are finite, seeds reach its head, it predicts. How good it
  is takes a full training run and an evaluation on a benchmark split.
- **Live, on your machine.** `grasppanda explore` serves this page from the installed
  package on `127.0.0.1` and composes every selection with your installation: any
  selection, of any number of slots, gets the verdict of the contract rules at once, with
  the conflict and the fix when it is refused, and an accepted one can be saved as a
  config. It never trains anything; the probe stays a command you run.
- **The data is generated.** `python -m tools.make_explorer` writes it from the measured
  file, the presets, the catalogue and the slot pages. The explorer itself loads nothing
  from the network: it works without a connection when the site is served
  (`mkdocs serve`, or `grasppanda explore`). The theme of this documentation site loads
  its font and the repository widget on its own.
