"""Generate the contract answers of the combination explorer.

The explorer answers a selection from the measured records. Most selections of two slots
have no record, and for those the page can still say one thing that is a fact: what
``grasppanda compose`` answers. This tool asks it, for every method, for every option of a
slot alone and for every pair of options of two different slots, and writes the answers
into ``docs/explorer/explorer_contracts.json``: accepted (with the adapter compose
inserted, if any) or refused with the messages, whole, and the kind of the refusal. The
slots of the training recipe (optimizer, schedule, hooks, augmentation) are asked alone and
in pairs among themselves; a recipe option together with a stage is not stored (it would
double the file and the time for an answer the contract rules give on the stage alone), so
the page says "not measured" there and prints the commands.

Nothing is trained and no dataset is read: this is the check of the contract rules and of
the config, the same call ``grasppanda compose`` makes. The page shows it as a second
tier, "contract check", never as a measured verdict.

    python -m tools.make_explorer_contracts            # rewrite (a few minutes, --jobs N)
    python -m tools.make_explorer_contracts --sample 40 # exit 1 when 40 answers drawn from
                                                        # the file differ from compose now

``tests/unit/test_explorer.py`` runs the sample check, so answers that went stale with a
change of the contract rules fail CI; an option added since the last run simply has no
contract answer on the page until this tool runs again. Run ``python -m
tools.make_explorer`` first: the options are read from its file.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
EXPLORER = REPO / "docs" / "explorer" / "explorer.json"
OUT = REPO / "docs" / "explorer" / "explorer_contracts.json"
#: The lane whose choices are recorded sets of loss terms, not options: every one of them
#: has a record, and a pair with one is left to the commands.
LOSS = "loss"


def answer(config: str, sets: list[str]) -> dict[str, Any]:
    """What compose answers for ``sets`` on ``config``, as the page stores it: accepted
    (with the adapter it inserted), or refused with its messages and the kind of the
    refusal as a whole (:func:`grasppanda.combine.levels.refusal_kind`, the rule a record
    of the grid is told by)."""
    from grasppanda.combine.compose import compose
    from grasppanda.combine.levels import refusal_kind

    try:
        composition = compose(config, sets)
    except Exception as error:  # compose says why it cannot even start
        return {"ok": False, "messages": [f"{type(error).__name__}: {error}"]}
    if composition.accepted:
        found: dict[str, Any] = {"ok": True}
        if composition.adapter:
            found["adapter"] = composition.adapter
        return found
    messages = composition.refusal()
    refused: dict[str, Any] = {"ok": False, "messages": messages}
    kind = refusal_kind({"refusal": messages, "refusal_kinds": composition.refusal_kinds()})
    if kind:
        refused["kind"] = kind
    return refused


def _answer(task: tuple[str, list[str]]) -> dict[str, Any]:
    return answer(*task)


def offered(data: dict[str, Any]) -> dict[str, dict[str, list[str]]]:
    """Method → slot → the options the page offers there, without the shipped one."""
    found: dict[str, dict[str, list[str]]] = {}
    for method in data["methods"]:
        slots: dict[str, list[str]] = {}
        for slot in method["slots"]:
            if slot == LOSS:
                continue
            ships = method["ships"].get(slot, {}).get("option", "")
            names = [option["name"] for option in data["options"].get(slot, [])]
            slots[slot] = [name for name in names if name != ships]
        found[method["name"]] = slots
    return found


def tasks(data: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    """Every (method, key, sets) the file answers: singles, then pairs of two slots."""
    configs = {method["name"]: method["config"] for method in data["methods"]}
    around = {lane["key"] for lane in data["lanes"] if lane.get("recipe")}
    found = []
    for name, slots in offered(data).items():
        for slot, names in slots.items():
            found += [(name, slot, [f"{slot}={option}"]) for option in names]
        for first, second in itertools.combinations(slots, 2):
            if (first in around) != (second in around):
                continue  # a stage with a recipe option: left to the commands
            for a, b in itertools.product(slots[first], slots[second]):
                found.append((name, f"{first}|{second}", [f"{first}={a}", f"{second}={b}"]))
    assert all(name in configs for name, _, _ in found)
    return found


def build(jobs: int) -> dict[str, Any]:
    data = json.loads(EXPLORER.read_text(encoding="utf-8"))
    configs = {method["name"]: method["config"] for method in data["methods"]}
    todo = tasks(data)
    work = [(configs[name], sets) for name, _, sets in todo]
    if jobs > 1:
        with ProcessPoolExecutor(jobs) as pool:
            answers = list(pool.map(_answer, work, chunksize=24))
    else:
        answers = [_answer(item) for item in work]
    # Answers and messages repeat a great deal: each is stored once and referred to by
    # its position. Position 0 is "accepted as it is".
    messages: list[str] = []
    message_at: dict[str, int] = {}
    table: list[dict[str, Any]] = [{"ok": True}]
    table_at: dict[str, int] = {json.dumps(table[0], sort_keys=True): 0}
    methods: dict[str, Any] = {
        name: {"options": slots, "answers": {}} for name, slots in offered(data).items()
    }
    for (name, key, _), found in zip(todo, answers, strict=True):
        if not found["ok"]:
            places = []
            for text in found["messages"]:
                if text not in message_at:
                    message_at[text] = len(messages)
                    messages.append(text)
                places.append(message_at[text])
            kind = found.get("kind")
            found = {"ok": False, "messages": places}
            if kind:
                found["kind"] = kind
        spelled = json.dumps(found, sort_keys=True)
        if spelled not in table_at:
            table_at[spelled] = len(table)
            table.append(found)
        methods[name]["answers"].setdefault(key, []).append(table_at[spelled])
    return {
        "generated_by": "tools/make_explorer_contracts.py; do not edit by hand",
        "what": (
            "What `grasppanda compose` answers for one option alone and for two options of "
            "two slots: accepted, or refused with the messages. No probe, no training."
        ),
        "counts": {
            "answers": len(todo),
            "accepted": sum(found["ok"] for found in answers),
            "refused": sum(not found["ok"] for found in answers),
        },
        "messages": messages,
        "table": table,
        "methods": methods,
    }


def lookup(contracts: dict[str, Any], method: str, sets: list[str]) -> dict[str, Any] | None:
    """The stored answer for one or two ``slot=option`` sets, or None (mirrors the page)."""
    entry = contracts["methods"].get(method)
    picks = dict(item.split("=", 1) for item in sets)
    if entry is None or not 1 <= len(picks) <= 2 or len(picks) != len(sets):
        return None
    slots = list(picks)
    # A pair is stored once, under the order of the pipeline: ``first|second``.
    if len(slots) == 2 and "|".join(slots) not in entry["answers"]:
        slots.reverse()
    row = entry["answers"].get("|".join(slots))
    if row is None:
        return None
    try:
        at = [entry["options"][slot].index(picks[slot]) for slot in slots]
    except (KeyError, ValueError):
        return None
    position = at[0] if len(at) == 1 else at[0] * len(entry["options"][slots[1]]) + at[1]
    found = dict(contracts["table"][row[position]])
    if not found["ok"]:
        found["messages"] = [contracts["messages"][index] for index in found["messages"]]
    return found


def sample(count: int) -> list[str]:
    """Differences between ``count`` stored answers and what compose answers now.

    The draw is fixed by the content of the file, so a given file is always checked at
    the same places and a regenerated one at others.
    """
    contracts = json.loads(OUT.read_text(encoding="utf-8"))
    data = json.loads(EXPLORER.read_text(encoding="utf-8"))
    configs = {method["name"]: method["config"] for method in data["methods"]}
    known = {
        slot: {option["name"] for option in members} for slot, members in data["options"].items()
    }
    stored = []
    for name, entry in contracts["methods"].items():
        options = entry["options"]
        for key in entry["answers"]:
            slots = key.split("|")
            for names in itertools.product(*(options[slot] for slot in slots)):
                if name in configs and all(n in known.get(s, ()) for s, n in zip(slots, names)):
                    stored.append((name, [f"{s}={n}" for s, n in zip(slots, names)]))
    seed = hashlib.sha256(OUT.read_bytes()).hexdigest()
    stored.sort(key=lambda item: hashlib.sha256(f"{seed}{item}".encode()).hexdigest())
    # Refusals are the rarer and the more telling answers: half of the draw.
    refused = [item for item in stored if not (lookup(contracts, *item) or {}).get("ok")]
    drawn = refused[: count // 2] + [item for item in stored if item not in refused[: count // 2]]
    problems = []
    for name, sets in drawn[:count]:
        now, then = answer(configs[name], sets), lookup(contracts, name, sets)
        if now != then:
            problems.append(f"{name} {' '.join(sets)}: stored {then}, compose now says {now}")
    return problems


def text(jobs: int) -> str:
    encoded = json.dumps(build(jobs), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return re.sub(r"</", r"<\\/", encoded).replace("<!--", "<\\u0021--") + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--jobs", type=int, default=4, help="processes that compose (default 4)")
    parser.add_argument(
        "--sample", type=int, metavar="N", help="do not write; compare N stored answers"
    )
    args = parser.parse_args(argv)
    if args.sample:
        problems = sample(args.sample)
        for problem in problems:
            print(f"stale: {problem}")
        if problems:
            print("run python -m tools.make_explorer_contracts")
        return 1 if problems else 0
    content = text(args.jobs)
    OUT.write_text(content, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO)} ({len(content)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
