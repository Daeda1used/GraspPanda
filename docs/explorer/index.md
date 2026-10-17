---
hide:
  - navigation
  - toc
---

# Combination explorer { .gx-sr }

Swap any stage of a grasp detector and see what happened when it was built and trained.
On your machine, `grasppanda explore --weights-dir <dir>` also runs a ported method on a
scene, in this page.
{ .gx-sr }

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

## What this page knows, and what it does not { #what-this-page-knows .gx-faq }

Every stage of the pipeline is a slot of the config. Pick a method, replace the option of a
slot, and the page answers from the
[measured records](../advanced_guides/measured_combinations.md).

- **It reads records, it does not predict.** A verdict is the outcome of
  [`grasppanda compose` followed by `grasppanda probe`](../user_guides/combinations.md) for
  exactly the combination shown; nothing is inferred from a neighbouring cell. The
  information button next to the record count says how many records stand on a probe of
  their own, how many share one, and how many are refusals, for which nothing was built.
- **Without a record, it may hold the answer of `compose`.** For one option alone and for
  two options of two slots, `python -m tools.make_explorer_contracts` asked the real
  `grasppanda compose` and stored what it said: accepted, or refused with the conflict and
  the fix. That answer is a *contract check*, drawn with a dashed outline and never like a
  measured verdict: "accepted" says that the contracts fit, not that the combination
  trains. A stage together with an option of the training recipe (optimizer, schedule,
  hooks, augmentation) is not stored. Any other selection is *not measured*, and the page
  prints the two commands that find out.
- **A method lists exactly its options.** A slot offers the options that
  `grasppanda compose <method config> --list <slot>` lists for that method, and the
  [page of the method](../methods/gsnet.md) shows the same list (a test holds the three
  together). An option without a cell of the grid says why.
- **Four levels, and classes per level.** A record is decided at four levels: *composes*,
  *runs*, *learns* and *predicts*; a record of the first judge says *not checked* for the
  last two. Compare shows the state of every option under every method at the level you
  pick, with the grasps per frame in the tooltip; the class of an option is given per level,
  and the filters use that level. A class is one run, seed 0, 20 iterations, 8 frames of one
  scene: it counts verdicts, not accuracy
  ([Measured classes](../advanced_guides/measured_combinations.md#measured-classes)).
- **No accuracy.** A probe trains 20 iterations on 8 frames. It shows that a combination
  works: it builds, its losses are finite, seeds reach its head, it predicts. How good it is
  takes a full training run and an evaluation on a benchmark split.
- **Live, on your machine.** `grasppanda explore` serves this page from the installed
  package on `127.0.0.1` and composes every selection with your installation: any
  selection, of any number of slots, gets the verdict of the contract rules at once, with
  the conflict and the fix when it is refused, and an accepted one can be saved as a config.
  It never trains anything; the probe stays a command you run.
- **Run a method, on your machine.** The Run tab is disabled here, because a run needs
  weights and a model on your computer. `grasppanda explore --weights-dir <dir>` enables it:
  pick a ported method (or your selection) and a weights file from `<dir>`, run it on the
  built-in example scene or on your own depth image, and read the picture, the best grasps
  and the counts `grasppanda demo` prints
  ([Run in the browser](../user_guides/inference.md#run-in-the-browser)).
- **The data is generated.** `python -m tools.make_explorer` writes it from the measured
  file, the presets, the catalogue and the slot pages. The explorer loads nothing from the
  network: it works offline when the site is served (`mkdocs serve`, or
  `grasppanda explore`). The theme of this documentation site loads its font and the
  repository widget on its own.
