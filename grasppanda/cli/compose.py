"""``grasppanda compose``: turn "this method with that component" into a checked config.

    grasppanda compose --list                          # presets, slot by slot
    grasppanda compose --show backbone=ptv3            # one preset, with its notes
    grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e \\
        --set backbone=pointnet2 --set grouper=multi-scale -o my_config.yaml

The command builds the combination from the base config and the presets, fits a new
encoder to what the method feeds and expects, inserts the encoder adapter the contracts
ask for (and says so), runs the config check and prints the verdict of the contract rules
stage by stage. An accepted combination is written as a child config whose header says
how it was made; a refused one prints the conflict and the fix, and nothing is written.

Exit status: 0 accepted, 1 refused by the contracts or the config check, 2 the request
itself is wrong (unknown slot, preset or component; a detector without slots; a base that
cannot be inherited from; an output file that exists).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path
from typing import Any

from grasppanda.utils.console import style, table

__all__ = ["register"]


EPILOG = """\
slots of a seed-based detector (GSNet, EconomicGrasp, FineGrasp, GraspNet-baseline,
Scale-Balanced-Grasp), in the order data flows:
  backbone  proposal  view  assigner  grouper  head
A choice is a preset (`grasppanda compose --list`) or a registered component by its type
(`grasppanda components --slot groupers`), which is then built with its own defaults.

examples:
  grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=ptv3 -o gsnet_ptv3.yaml
  grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=pointnet2 \\
      --set grouper=multi-scale --opts train.epochs=20 -o mine.yaml
  grasppanda compose mine.yaml --set head=swad -o mine2.yaml    # composed again from its base

the base is a config that inherits fragments only, like the method configs under configs/
(a chain is two levels deep). A config that was composed is composed again from its own
base with both requests merged. Without -o the verdict is printed and nothing is written.

exit status:
  0  accepted (and written, with -o)
  1  refused by the contract rules or the config check; nothing is written
  2  the request is wrong: unknown slot, preset or component, a detector without slots,
     a base that cannot be inherited from, or an output file that exists (see --force)

next: grasppanda probe <the composed config>
"""


def register(commands: Any) -> None:
    from grasppanda.combine.compose import command_parser

    parser = commands.add_parser(
        "compose",
        help="compose a config from a base config and presets, and check it",
        description=(
            "Compose a child config: a base config with one component per --set, taken from "
            "a preset or from the registry. The contract rules and the config check decide "
            "whether it is written."
        ),
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    command_parser(parser)
    parser.add_argument("-o", "--output", metavar="FILE", help="write the composed config here")
    parser.add_argument("--force", action="store_true", help="with -o: replace FILE when it exists")
    parser.add_argument(
        "--list",
        dest="list_presets",
        nargs="?",
        const="all",
        metavar="SLOT",
        help="list the presets (of one slot, or all) and the components without one",
    )
    parser.add_argument("--show", metavar="SLOT=PRESET", help="print one preset file")
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    parser.set_defaults(handler=run)


# ---------------------------------------------------------------------------- listings


def _stage_slots() -> dict[str, Any]:
    from grasppanda.models.detectors.seed_based import STAGE_REGISTRIES

    return dict(STAGE_REGISTRIES)


def _slot(name: str) -> str | None:
    """A slot by its name or by the plural `grasppanda components --slot` uses."""
    registries = _stage_slots()
    plural = {registry.name.lower(): slot for slot, registry in registries.items()}
    name = name.strip().lower()
    return name if name in registries else plural.get(name)


def _list(which: str) -> int:
    from grasppanda.combine.presets import LOSS_SLOT, POST_SLOT, presets
    from grasppanda.combine.recipes import RECIPE_SLOTS, recipes
    from grasppanda.registry import LOSSES, POSTPROCESSORS

    registries = {**_stage_slots(), LOSS_SLOT: LOSSES, POST_SLOT: POSTPROCESSORS}
    found = presets()
    wanted = _slot(which) or which.strip()
    slots = [slot for slot in registries if which == "all" or slot == wanted]
    around = [slot for slot in RECIPE_SLOTS if which in ("all", slot)]
    if not slots and not around:
        known = ", ".join([*registries, *RECIPE_SLOTS])
        print(f"unknown slot '{which}'. Slots: {known}", file=sys.stderr)
        return 2
    color = sys.stdout.isatty()
    for slot in around:
        several = " (several: a+b)" if slot in ("hooks", "augment") else ""
        print(style(slot, "bold", "cyan", enabled=color) + f"  --set {slot}=<preset>{several}")
        rows = [
            [name, preset.type, _short(preset.summary)]
            for name, preset in recipes().get(slot, {}).items()
        ]
        if rows:
            print(table(["preset", "sets", "what it is"], rows, enabled=color))
        print()
    if not slots:
        print("One preset with its notes: grasppanda compose --show <slot>=<preset>.")
        return 0
    for slot in slots:
        members = found.get(slot, {})
        target = "<stage>.<loss option>" if slot == LOSS_SLOT else slot
        print(
            style(slot, "bold", "cyan", enabled=color) + f"  --set {target}=<preset or component>"
        )
        rows = [[name, preset.type, _short(preset.summary)] for name, preset in members.items()]
        if rows:
            print(table(["preset", "component", "what it is"], rows, enabled=color))
        covered = {preset.type for preset in members.values()}
        bare = [name for name in registries[slot].names() if name not in covered]
        if bare:
            print(f"  components without a preset: {', '.join(bare)}")
        print()
    if LOSS_SLOT in slots:
        print(
            "a loss goes into a loss option of a stage and keeps the weight of the loss it\n"
            "replaces, e.g.\n"
            "  --set head.score_loss=quality-focal --set proposal.objectness_loss=focal\n"
            "The config check refuses a loss in a term of another kind and names those that fit.\n"
        )
    if POST_SLOT in slots:
        print(
            "post is a chain: join steps with `+` and set a main option with `=`, e.g.\n"
            "  --set post=collision+nms+top-k=50      (also: --post ... of test, infer, demo)\n"
        )
    print(
        "A component can also be named by its type (`--set grouper=CylinderGroup`): it is then\n"
        "built with its constructor defaults, which `grasppanda components <type>` lists.\n"
        "One preset with its notes: grasppanda compose --show <slot>=<preset>."
    )
    return 0


def _short(summary: str, limit: int = 96) -> str:
    """A summary cut at a word boundary, for a table in a terminal."""
    if len(summary) <= limit:
        return summary
    return summary[:limit].rsplit(" ", 1)[0].rstrip(",;:") + " …"


def _show(text: str) -> int:
    from grasppanda.combine import recipes
    from grasppanda.combine.presets import PresetError, find, presets

    slot, separator, name = text.partition("=")
    known = {*presets(), *recipes.RECIPE_SLOTS}
    if not separator or (_slot(slot) is None and slot.strip() not in known):
        print(
            f"--show takes `slot=preset`, for example `backbone=ptv3`; got '{text}'. "
            f"Slots: {', '.join(dict.fromkeys([*_stage_slots(), *sorted(known)]))}. "
            "`grasppanda compose --list` shows the presets.",
            file=sys.stderr,
        )
        return 2
    try:
        if slot.strip() in recipes.RECIPE_SLOTS:
            preset = recipes.find(slot.strip(), name.strip())
        else:
            preset = find(_slot(slot) or slot.strip(), name.strip())
    except PresetError as error:
        print(str(error), file=sys.stderr)
        return 2
    print(f"# {preset.location}")
    sys.stdout.write(preset.path.read_text(encoding="utf-8"))
    return 0


# ------------------------------------------------------------------------------ report


def _paragraph(mark: str, text: str) -> None:
    """Print ``text`` after ``mark``, wrapped, with the following lines indented under it."""
    lines = textwrap.wrap(text, 96, break_long_words=False, break_on_hyphens=False) or [""]
    print(f"  {mark} {lines[0]}")
    for line in lines[1:]:
        print(f"    {line}")


def _report(composition: Any, written: Path | None, color: bool) -> None:
    ok = style("✓", "bold", "green", enabled=color)
    bad = style("✗", "bold", "red", enabled=color)
    print(style(composition.command, "bold", enabled=color))
    print()
    if composition.stages:
        rows = [
            [slot, name, composition.origins[slot]] for slot, name in composition.stages.items()
        ]
        print(table(["slot", "component", "from"], rows, enabled=color))
        print()
    if composition.actions:
        print(style("what compose did", "bold", enabled=color))
        for action in composition.actions:
            _paragraph("•", action)
        print()
    if composition.seed_based:
        print(style("contracts", "bold", enabled=color))
        shown: set[str] = set()
        for verdict in composition.verdicts:
            label = " · ".join(verdict.slots)
            new = [problem for problem in verdict.problems if problem not in shown]
            if not verdict.problems:
                print(f"  {ok} {label}")
            elif new:
                for problem in new:
                    _paragraph(bad, f"{label}: {problem}")
            else:
                print(f"  {bad} {label}: as above")
            shown.update(verdict.problems)
        for problem in composition.problems:
            if problem not in shown:
                _paragraph(bad, f"whole stage set: {problem}")
    else:
        _paragraph(
            "-",
            f"contracts: {composition.detector} is not built on the seed-based skeleton, so "
            "there are no stage contracts to compare; the config check below is the verdict.",
        )
    print()
    print(style("config check", "bold", enabled=color))
    if composition.accepted:
        print(f"  {ok} valid  (hash {composition.config_hash[:12]})")
    else:
        for issue in composition.issues:
            _paragraph(bad, f"{issue.path or '<file>'}: {issue.message}")
        if not composition.issues:
            print(f"  {bad} refused by the contracts above")
    print()
    if not composition.accepted:
        print(
            f"{bad} refused: nothing was written. The messages above name the conflict and a fix."
        )
        for hint in composition.hints:
            _paragraph("→", hint)
    elif written is not None:
        print(f"{ok} written {written}")
        print(f"  next: grasppanda probe {written}")
    else:
        print(f"{ok} accepted. Pass -o FILE to write the config, or copy it from --json.")


def _json(composition: Any, written: Path | None, text: str) -> dict[str, Any]:
    return {
        "command": composition.command,
        "base": composition.base,
        "detector": composition.detector,
        "accepted": composition.accepted,
        "adapter": composition.adapter,
        "stages": composition.stages,
        "origins": composition.origins,
        "actions": composition.actions,
        "contracts": [
            {"slots": list(verdict.slots), "problems": list(verdict.problems)}
            for verdict in composition.verdicts
        ],
        "problems": composition.problems,
        "issues": [{"path": issue.path, "message": issue.message} for issue in composition.issues],
        "hints": composition.hints,
        "config_hash": composition.config_hash,
        "output": None if written is None else str(written),
        "config": text,
    }


def _output_problem(composition: Any, target: Path, force: bool) -> str:
    """Why the composed config may not be written to ``target``; empty when it may."""
    from grasppanda.config import Config

    inherited = {path.resolve() for path in Config.load(composition.base_path).sources}
    if target.resolve() in inherited:
        return (
            f"{target} is the base config or a file it inherits from: writing the composed "
            "config there would make it inherit from itself. Choose another name for -o."
        )
    if target.exists() and not force:
        return f"{target} exists. Choose another name for -o, or pass --force to replace it."
    return ""


def run(args: argparse.Namespace) -> int:
    from grasppanda.combine.compose import ComposeError, compose
    from grasppanda.config import Config

    if args.list_presets:
        return _list(args.list_presets)
    if args.show:
        return _show(args.show)
    if not args.base:
        print(
            "compose needs a base config (or --list / --show), for example\n"
            "  grasppanda compose gsnet/gsnet_sunet14_g1b-realsense_10e --set backbone=ptv3",
            file=sys.stderr,
        )
        return 2

    def wrong(message: str) -> int:
        mark = style("✗", "bold", "red", enabled=sys.stderr.isatty())
        lines = textwrap.wrap(message, 96, break_long_words=False, break_on_hyphens=False)
        print(f"{mark} " + "\n  ".join(lines), file=sys.stderr)
        return 2

    try:
        composition = compose(args.base, args.sets, args.opts, adapters=args.adapters)
    except ComposeError as error:
        return wrong(str(error))
    written = None
    if composition.accepted and args.output:
        written = Path(args.output)
        problem = _output_problem(composition, written, args.force)
        if problem:
            return wrong(problem)
        written.parent.mkdir(parents=True, exist_ok=True)
        written.write_text(composition.text(written), encoding="utf-8")
        # The checks ran on a scratch copy; the file the user gets must load to the same config.
        if Config.load(written).digest() != composition.config_hash:
            written.unlink()
            return wrong(
                f"the file written to {written} does not load to the config that was checked; "
                "it was removed. Please report this with the command."
            )
    text = composition.text(written)
    if args.json:
        print(json.dumps(_json(composition, written, text), indent=2, ensure_ascii=False))
    else:
        _report(composition, written, sys.stdout.isatty())
    return 0 if composition.accepted else 1
