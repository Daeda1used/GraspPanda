# Guides

One task per page, from a dataset on disk to a trained, tested and extended detector; each
page gives the commands for every method it applies to.

<div class="grid cards" markdown>

-   :material-file-cog-outline:{ .lg .middle } **[Configuration](config.md)**

    ---

    One YAML file is the whole experiment: inherit a shipped config, override, validate.

-   :material-database-outline:{ .lg .middle } **[Datasets](datasets.md)**

    ---

    GraspNet-1Billion and GraspClutter6D on disk, and the label assets each method reads.

-   :material-school-outline:{ .lg .middle } **[Train and test](train_test.md)**

    ---

    Train one of the six methods, score it on the benchmark, resume a run.

-   :material-swap-horizontal:{ .lg .middle } **[Cross-dataset evaluation](cross_dataset.md)**

    ---

    Test GraspNet-1Billion weights on GraspClutter6D, with only the test data changed.

-   :material-camera-outline:{ .lg .middle } **[Your camera and Python](inference.md)**

    ---

    Grasps on your own depth images, from the command line, Python or a browser.

-   :material-chart-box-outline:{ .lg .middle } **[Read an analysis](analysis.md)**

    ---

    Where the AP of a result comes from, and whether two results really differ.

-   :material-puzzle-outline:{ .lg .middle } **[Compose and probe](combinations.md)**

    ---

    Another option in a slot of a method, checked on real frames in a minute.

-   :material-plus-box-outline:{ .lg .middle } **[Add a component](../advanced_guides/add_component.md)**

    ---

    Write, register and use your own encoder, stage, hook or evaluator without forking.

</div>
