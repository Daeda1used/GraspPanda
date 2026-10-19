---
hide:
  - navigation
  - toc
---

# Combination explorer { .gx-sr }

Pick a method, swap a stage, and see whether it still trains. Every answer is a recorded
probe, a stored answer of `grasppanda compose`, or says that nothing was measured. On your
machine, `grasppanda explore --weights-dir <dir>` also runs a ported method on a scene, in
this page.
{ .gx-sr }

<div id="gp-explorer" class="gpx">
<noscript><p>The explorer needs JavaScript. The same records are on the page
<a href="../advanced_guides/measured_combinations/">Does it run?</a>.</p></noscript>
</div>
<link rel="stylesheet" href="explorer.css">
<script type="application/json" id="gp-explorer-data">
--8<-- "docs/explorer/explorer.json"
</script>
<script type="application/json" id="gp-explorer-contracts">
--8<-- "docs/explorer/explorer_contracts.json"
</script>
<script type="application/json" id="gp-explorer-learns">
--8<-- "docs/explorer/explorer_learns.json"
</script>
<script src="explorer_logic.js"></script>
<script src="explorer.js"></script>

## Questions { #questions .gx-faq }

??? note "Where do the answers come from?"

    From records, never from a guess. A verdict is what
    [`grasppanda compose` and `grasppanda probe`](../user_guides/combinations.md) gave for
    exactly the combination shown. Without a record, the page may hold the answer of
    `compose` alone, drawn with a dashed outline: it says the stages fit, not that the
    combination trains. It is stored for one change and for two changes of two slots; a
    stage together with a recipe option has no stored compose answer. Anything else is
    *not measured*, and the page prints the two commands that find out. How each level is
    judged: [Evidence](../evidence/index.md).

??? note "Does a green tile mean a good combination?"

    --8<-- "docs/includes/caveat_probe.md"

    **No accuracy** appears on this page. *Trains like shipped* goes one step further: a
    swap that ran was trained longer, beside the component the method ships, and their
    losses were compared. [How we measure](../evidence/index.md).

??? note "Why does a method offer these options?"

    A slot lists what `grasppanda compose <method config> --list <slot>` lists for that
    method, and the [method's page](../methods/index.md) shows the same list. A method
    without a slot says so in Compare: HGGD reads an RGB-D image and has an image encoder
    instead of a scene encoder.

??? note "Can I run a method here?"

    On your machine. `grasppanda explore` serves this page from the installed package and
    checks a selection of any number of slots with your own `compose`, saves it as a
    config, and never trains: the probe stays a command you run. With weights,
    `grasppanda explore --weights-dir <dir>` also enables the Run tab: pick a method and a
    weights file, and run it on the example scene or on your depth image
    ([Run in the browser](../user_guides/inference.md#run-in-the-browser)).
