# GraspPanda

**The toolbox for 6-DoF grasp detection in clutter.** Swappable components,
faithful ports, one pinned evaluation.

!!! note "Alpha"
    Working today: the config system, both benchmark readers and evaluators, six methods
    as swappable components, options for every slot of the pipeline, and the commands
    from `train` to `compose`, `probe` and `explore`. The model zoo is still empty: a
    method enters only after it has been trained and evaluated here, and no accuracy is
    claimed for an option that is not part of a ported method.

**[Try combinations in the browser](explorer/index.md)**: pick a method, swap a slot, read
the measured verdict, the command and the reason of a refusal. `grasppanda explore` serves
the same page locally and checks any selection with the `compose` of your installation.

![grasppanda print-config catching a misspelt key and an incompatible swap](assets/terminal-demo.png)

## Where to go

| I want to… | Read |
|---|---|
| install it and see it run | [Installation](get_started/installation.md), [Quick start](get_started/quickstart.md) |
| understand the config language | [Configuration](user_guides/config.md) |
| see every stage of the pipeline, its options and which fits which method | [Slots](slots/index.md) |
| try a combination in the browser | [Combination explorer](explorer/index.md) |
| compose, probe and train a combination | [Try a combination](user_guides/combinations.md) |
| see every component, its contract and where it comes from | [Components](components.md) |
| replace the encoder or another stage of a method | [Swapping components](advanced_guides/swapping.md) |
| add my own backbone, head or detector | [Add a component](advanced_guides/add_component.md) |
| know what the stages pass to each other | [Data contracts](advanced_guides/data_flow.md) |
| see why it is built this way | [Architecture](design/ARCHITECTURE.md) |

## The idea in one picture

![Pipeline slots and the contracts between them](assets/pipeline.png)

The panda is the animal, not the robot arm: GraspPanda is not tied to any robot.
