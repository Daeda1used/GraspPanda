/* The decisions of the combination explorer against every measured record.
 *
 * Run by tests/unit/test_explorer.py:  node tests/js/explorer_logic.test.js <repo root>
 * Exit status 0 and a last line "ok <n> checks" when everything holds.
 */
"use strict";

const assert = require("assert");
const fs = require("fs");
const path = require("path");

const repo = path.resolve(process.argv[2] || path.join(__dirname, "..", ".."));
const L = require(path.join(repo, "docs", "explorer", "explorer_logic.js"));
const data = L.prepare(
  JSON.parse(fs.readFileSync(path.join(repo, "docs", "explorer", "explorer.json"), "utf8"))
);
// Level 2 is a file of its own; the page attaches it to the data as `learns`.
data.learns = JSON.parse(
  fs.readFileSync(path.join(repo, "docs", "explorer", "explorer_learns.json"), "utf8")
);
const measured = JSON.parse(
  fs.readFileSync(path.join(repo, "docs", "advanced_guides", "measured_combinations.json"), "utf8")
);

// Refused records whose combination compose now accepts or refuses otherwise: they await
// their probe (the same list as tests/unit/test_combine_grid.py).
const awaitingFile = path.join(repo, "tests", "unit", "fixtures", "awaiting_probe.json");
const awaiting = new Set(
  fs.existsSync(awaitingFile) ? JSON.parse(fs.readFileSync(awaitingFile, "utf8")) : []
);

const contractsFile = path.join(repo, "docs", "explorer", "explorer_contracts.json");
const contracts = fs.existsSync(contractsFile) ? JSON.parse(fs.readFileSync(contractsFile, "utf8")) : null;

let checks = 0;
function check(condition, message) {
  assert.ok(condition, message);
  checks += 1;
}

/* The selection a visitor makes to arrive at a record. */
function picksOf(record) {
  const picks = {};
  if (record.group === "loss") {
    picks.loss = L.lossName(record);
    return picks;
  }
  for (const item of record.sets || []) {
    const at = item.indexOf("=");
    picks[item.slice(0, at)] = item.slice(at + 1);
  }
  return picks;
}

// 1. Every record is reachable by a selection, and the page shows its verdict.
for (const record of measured.cells) {
  const result = L.answer(data, record.method, picksOf(record));
  check(result.cell, `${record.id}: no answer`);
  check(result.cell.verdict === record.verdict, `${record.id}: ${result.cell.verdict}`);
  check(result.state === (result.sets.length ? "measured" : "shipped"), `${record.id}: state`);
  const kind = result.badge.kind;
  const expected =
    record.verdict === "refused"
      ? "refused"
      : record.verdict === "fails"
        ? "fails"
        : record.verdict === "runs-unstable"
          ? "unstable"
          : record.verdict === "runs-empty"
            ? "empty"
            : result.cell.adapter
              ? "adapter"
              : "runs";
  check(kind === expected, `${record.id}: badge ${kind}, expected ${expected}`);
  // The command of a selection is the recorded one, up to the name of the output file
  // and to sets that name what the method ships with.
  if (record.compose && result.sets.length === record.sets.length) {
    const recorded = record.compose.replace(/ -o \S+$/, "");
    const shown = result.compose.replace(/ -o \S+$/, "");
    check(shown === recorded, `${record.id}: command\n  ${shown}\n  ${recorded}`);
  }
  if (record.verdict === "refused") {
    check(result.cell.refusal && result.cell.refusal.length, `${record.id}: no reason`);
  }
  if (record.verdict === "fails") check(result.cell.failure.phase, `${record.id}: no phase`);
}

// 2. A selection without a record is "not measured": a command, and no verdict.
const first = data.methods.find((m) => m.seed_based);
const slots = first.slots.filter((slot) => slot !== "loss" && (data.options[slot] || []).length);
let unmeasured = null;
outer: for (const a of L.choices(data, first.name, slots[0])) {
  for (const b of L.choices(data, first.name, slots[slots.length - 1])) {
    const picks = { [slots[0]]: a.name, [slots[slots.length - 1]]: b.name };
    const result = L.answer(data, first.name, picks);
    if (result.sets.length === 2 && !result.cell) {
      unmeasured = result;
      break outer;
    }
  }
}
check(unmeasured, "no unmeasured pair found: extend the test");
check(unmeasured.state === "unmeasured" && unmeasured.badge.kind === "none", "unmeasured state");
check(unmeasured.badge.label === "not measured", "unmeasured label");
check(unmeasured.compose.startsWith(`grasppanda compose ${first.config} --set `), "command");
check(unmeasured.probe.includes(unmeasured.output), "probe names the composed file");
check(unmeasured.parts.length === 2, "one part per changed slot");

// 3. Nothing changed is the method as shipped; picks that name what it ships with vanish.
for (const m of data.methods) {
  const shipped = {};
  for (const slot of Object.keys(m.ships)) shipped[slot] = m.ships[slot].option;
  const result = L.answer(data, m.name, shipped);
  check(result.state === "shipped" && result.sets.length === 0, `${m.name}: shipped`);
  check(result.compose === "" && result.probe.includes(m.config), `${m.name}: probe`);
  check(Object.keys(L.normalise(data, m.name, { nonsense: "x", backbone: "no-such" })).length === 0, "unknown picks");
}

// 4. A slot a method does not have offers nothing.
for (const m of data.methods) {
  for (const entry of data.lanes) {
    const offered = L.choices(data, m.name, entry.key);
    check(m.slots.includes(entry.key) || offered.length === 0, `${m.name}/${entry.key}`);
  }
}

// 5. The grid: one row per option, and a cell is the record of that single swap.
for (const entry of data.lanes) {
  const grid = L.gridRows(data, entry.key);
  check(grid.rows.length === data.options[entry.key].length, `${entry.key}: rows`);
  for (const row of grid.rows) {
    for (const m of grid.columns) {
      for (const cell of row.cells[m.name]) {
        check(cell.method === m.name, "cell under its method");
        if (entry.key === "loss") {
          check(cell.sets.some((item) => item.endsWith("=" + row.option.name)), "cell of its loss");
        } else {
          // the option's request under the method: `slot=option`, or the sets of its list
          const listed = L.choices(data, m.name, entry.key).find((choice) => choice.name === row.option.name);
          const key = L.keyOf(data, m.name, listed ? listed.sets : [entry.key + "=" + row.option.name]);
          check(cell.key === key, "cell of its option");
        }
      }
    }
  }
}

// 6. Filters.
const all = data.options.backbone || [];
check(all.filter((o) => L.matches(o, {})).length === all.length, "empty filter");
check(all.filter((o) => L.matches(o, { extra: true })).every((o) => o.extras.length), "extra");
check(all.filter((o) => L.matches(o, { weights: true })).every((o) => o.weights), "weights");
for (const level of [...data.levels.map((item) => item.key), "level2"]) {
  for (const tone of L.tonesOf(data, "backbone", level)) {
    const hits = all.filter((o) => L.matches(o, { level, tones: [tone] }, data));
    check(hits.length && hits.every((o) => L.toneOf(data, o, level) === tone), `tone ${tone} at ${level}`);
  }
}
// The levels of a record are four letters; the page reads them through level_states.
for (const cell of data.cells) {
  check(typeof cell.levels === "string" && cell.levels.length === data.levels.length, `${cell.id}: levels`);
  for (const level of data.levels) check(L.levelState(data, cell, level.key) in { yes: 1, no: 1, ...data.undecided }, `${cell.id}: ${level.key}`);
  check(!L.noGrasp(cell) || (cell.grasps.length && cell.grasps.every((count) => count === 0)), `${cell.id}: grey`);
}
if (all.length) {
  const hits = all.filter((o) => L.matches(o, { q: all[0].name.toUpperCase() }));
  check(hits.some((o) => o.name === all[0].name), "search is case-insensitive");
  check(all.filter((o) => L.matches(o, { q: "no-such-thing-zzz" })).length === 0, "no match");
}

// 7. A link carries the selection; without a method it carries none (the gallery opens), and
// Compare and Run need no method.
const back = L.fromHash("#" + L.toHash("GSNet", { backbone: "ptv3", post: "a b" }, "grid", "view"));
check(back.method === "GSNet" && back.picks.backbone === "ptv3" && back.picks.post === "a b", "hash");
check(back.view === "grid" && back.slot === "view", "hash view");
check(L.fromHash("#%E0%A4%A&m=x").method === "x", "a broken fragment is ignored");
check(L.toHash("", {}, "build", "") === "", "no method, no fragment");
check(L.toHash("", { backbone: "ptv3" }, "build", "") === "", "no picks without a method");
check(L.toHash("", {}, "grid", "post") === "grid=post", "Compare without a method");
check(L.toHash("", {}, "run", "") === "run=1", "Run without a method");
const gridOnly = L.fromHash("#grid=backbone");
check(gridOnly.method === "" && gridOnly.view === "grid" && gridOnly.slot === "backbone", "a Compare link");
check(L.fromHash("").method === "" && L.fromHash("").view === "build", "the empty fragment opens the gallery");
check(L.method(data, "") === null, "no method is a method");

// 8. No accuracy: the facts of a record are parameters, losses and counts of grasps.
for (const cell of data.cells) {
  for (const fact of L.facts(cell)) {
    check(!/\bAP\b|accuracy|precision/i.test(fact[0]), `${cell.id}: ${fact[0]}`);
  }
}
check(L.number(15353775) === "15 353 775", "digit groups");
check(L.outputName("GSNet", ["backbone=ptv3", "grouper=multi-scale"]) === "gsnet_ptv3_multi-scale.yaml", "name");

// 9. A refused combination has no probe command: compose writes no file for it.
for (const cell of data.cells) {
  if (cell.verdict !== "refused") continue;
  const result = L.answer(data, cell.method, picksOf(cell));
  check(result.probe === "", `${cell.id}: a probe command for a refused combination`);
}

// 10. A link that names what the page does not know is reported, not repaired silently.
check(L.dropped(data, L.fromHash("#m=GSNet&backbone=doesnotexist")).join() === "`backbone=doesnotexist`", "unknown option");
check(L.dropped(data, L.fromHash("#m=Nope&backbone=ptv3"))[0] === "the method `Nope`", "unknown method");
check(L.dropped(data, L.fromHash("#m=HGGD&backbone=ptv3"))[0].includes("not a slot of HGGD"), "slot of another method");
check(L.dropped(data, L.fromHash("#m=GSNet&backbone=ptv3")).length === 0, "a known link");
check(L.dropped(data, L.fromHash("#m=GSNet&backbone=sparse-unet14")).length === 0, "the shipped option");
check(L.dropped(data, L.fromHash("")).length === 0, "no fragment");
check(L.dropped(data, L.fromHash("#backbone=ptv3"))[0] === "`backbone=ptv3` (no method chosen)", "a pick without a method");
check(L.dropped(data, L.fromHash("#grid=post")).length === 0, "Compare without a method");

// 11. Losses: a choice is named by its terms, and a grid cell never shows one verdict
// for records that disagree.
for (const m of data.methods) {
  for (const choice of L.choices(data, m.name, "loss")) {
    check(choice.label === L.lossLabel(choice.sets), "loss label");
    check(choice.sets.every((item) => choice.label.includes(item)) || / on \d+ terms$/.test(choice.label), `${choice.label}`);
  }
}
for (const row of L.gridRows(data, "loss").rows) {
  for (const found of Object.values(row.cells)) {
    const kinds = new Set(found.map((cell) => L.badge(cell).kind));
    const shown = L.badgeOf(found);
    check(kinds.size <= 1 ? shown.kind === (found.length ? [...kinds][0] : "none") : shown.kind === "mixed", "loss cell");
    if (kinds.size > 1) for (const cell of found) check(shown.label.includes(cell.sets.join(" ")), "mixed label");
  }
}

// 12. The contract check is a tier of its own: it never replaces or imitates a record.
if (contracts) {
  const withContracts = Object.assign({}, data, { contracts });
  let accepted = 0;
  let refused = 0;
  for (const m of data.methods) {
    const entry = contracts.methods[m.name];
    check(entry, `${m.name}: no contract answers`);
    for (const key of Object.keys(entry.answers)) {
      const slots = key.split("|");
      const sizes = slots.map((slot) => entry.options[slot].length);
      check(entry.answers[key].length === sizes.reduce((a, b) => a * b, 1), `${m.name}/${key}: size`);
    }
    // every single swap and a spread of pairs
    const slots = Object.keys(entry.options);
    for (const slot of slots) {
      for (const name of entry.options[slot]) {
        const sets = [slot + "=" + name];
        const found = L.contract(withContracts, m.name, sets);
        check(found && typeof found.ok === "boolean", `${m.name} ${sets}: no contract answer`);
        const result = L.answer(withContracts, m.name, { [slot]: name });
        if (result.cell) {
          check(result.contract === null && result.badge.kind !== "accepted", "a record decides");
          // A record that was refused is refused by compose now as well, and the other way round.
          if (!awaiting.has(result.cell.id)) {
            check((result.cell.verdict === "refused") === !found.ok, `${m.name} ${sets}: record ${result.cell.verdict}, compose now ${found.ok}`);
          }
        } else if (result.sets.length > 1) {
          // An option that needs another slot's option (a recipe for one encoder): two
          // sets, a stage with a recipe option, which the contract answers do not store.
          check(result.state === "unmeasured" && result.badge.kind === "none", "an option with its requirement");
        } else if (result.sets.length) {
          check(result.state === "unmeasured", "no record stays unmeasured");
          check(result.badge.kind === (found.ok ? "accepted" : "refused"), "contract badge");
          check(/contract check/.test(result.badge.label), "the badge says it is the contract check");
          check(found.ok ? result.probe !== "" : result.probe === "" && found.messages.length > 0, "probe and messages");
          found.ok ? accepted++ : refused++;
        }
      }
    }
    for (let i = 0; i < slots.length; i++) {
      for (let j = i + 1; j < slots.length; j++) {
        const a = entry.options[slots[i]];
        const b = entry.options[slots[j]];
        for (let k = 0; k < Math.min(a.length * b.length, 40); k++) {
          const x = a[(k * 7) % a.length];
          const y = b[(k * 3) % b.length];
          const result = L.answer(withContracts, m.name, { [slots[i]]: x, [slots[j]]: y });
          if (result.sets.length !== 2) continue;
          const found = L.contract(withContracts, m.name, result.sets);
          // A stage together with an option of the training recipe is not stored: the
          // page has no answer of the whole there, and must not make one up.
          if (Boolean(L.lane(data, slots[i]).recipe) !== Boolean(L.lane(data, slots[j]).recipe)) {
            check(found === null, `${m.name} ${result.sets}: an answer that the tool does not store`);
            if (!result.cell) check(result.state === "unmeasured" && result.badge.kind === "none", "stage with recipe");
            continue;
          }
          check(found, `${m.name} ${result.sets}: no contract answer`);
          if (result.cell) check(result.contract === null, "a record decides");
          else check(result.badge.kind === (found.ok ? "accepted" : "refused") && result.state === "unmeasured", "pair");
        }
      }
    }
  }
  check(accepted > 0 && refused >= 0, "contract answers for unmeasured single swaps");
  // Three changes: no answer of the whole, only what is known of the parts.
  const three = L.answer(withContracts, "GSNet", { backbone: "ptv3", grouper: "multi-scale", post: "approach-cone" });
  if (!three.cell && three.sets.length === 3) {
    check(three.contract === null && three.badge.kind === "none", "three changes are not answered");
    check(three.pairs.length === 3, "its pairs are listed");
  }
  check(L.contract(withContracts, "GSNet", ["head.width_loss=focal"]) === null, "no contract answer for a loss term");
  check(L.contract(withContracts, "GSNet", ["backbone=no-such"]) === null, "no contract answer for an unknown option");
}

// 13. The tiles of the grid at a level: a record decides, the contract check fills in only
// where no record exists and says so, and a refusal reaches no level. What a method ships
// is "own" at every level and at level 2; an option without a record of its own is muted.
{
  const withContracts = contracts ? Object.assign({}, data, { contracts }) : data;
  const recordStates = new Set(["yes", "no", "refused", "unstable", "undecided", "mixed"]);
  let owned = 0;
  for (const entry of data.lanes) {
    const grid = L.gridRows(data, entry.key);
    for (const row of grid.rows) {
      for (const m of grid.columns) {
        const found = row.cells[m.name];
        const sets = [entry.key + "=" + row.option.name];
        const checked = entry.key === "loss" ? null : L.contract(withContracts, m.name, sets);
        const own = found.length > 0 && found.every((cell) => cell.own);
        for (const level of [...data.levels.map((item) => item.key), "level2"]) {
          const tile = L.levelTile(data, found, level, found.length ? null : checked, row.noRecord);
          if (own) {
            check(tile.state === "own" && !tile.contract, `${m.name}/${row.option.name}: what it ships shown as ${tile.state} at ${level}`);
            owned += 1;
          } else if (row.noRecord && !found.length) {
            check(tile.state === "muted" && !tile.contract, `${m.name}/${row.option.name}: no record of its own, shown as ${tile.state}`);
          } else if (level === "level2") {
            check(tile.state !== "own" && tile.state !== "accepted" && tile.state !== "check-refused", `${m.name}/${row.option.name}: level 2 ${tile.state}`);
          } else if (found.length) {
            check(recordStates.has(tile.state) && !tile.contract, `${m.name}/${row.option.name}: a record shown as ${tile.state}`);
            if (found.length === 1) {
              const cell = found[0];
              const said = L.levelState(data, cell, level);
              if (cell.verdict === "refused") check(tile.state === "refused", `${cell.id}: refused at ${level}`);
              else check((tile.state === "yes") === (said === "yes"), `${cell.id}: ${level} ${said} as ${tile.state}`);
              if (tile.state === "undecided") check(tile.label in data.undecided, `${cell.id}: ${tile.label}`);
            }
          } else if (checked) {
            check(tile.contract && tile.state === (checked.ok ? "accepted" : "check-refused"), "a contract answer is marked as one");
          } else {
            check(tile.state === "none" && !tile.contract, "no answer");
          }
        }
      }
    }
  }
  check(owned > 0, "a method's own option is drawn as its own somewhere");
  check(L.levelTile(data, [], "runs", null).state === "none", "nothing");
  // Every record of what a method ships is marked: it composes to the method as shipped.
  for (const m of data.methods) {
    for (const slot of Object.keys(m.ships)) {
      const name = m.ships[slot].option;
      if (!name || slot === "loss") continue;
      const cell = data.cells.find((item) => item.method === m.name && item.sets.length === 1 && item.sets[0] === slot + "=" + name);
      if (cell) check(cell.own, `${cell.id}: what ${m.name} ships is not marked as its own`);
    }
  }
}
check(L.thousands(1262) === "1,262" && L.thousands(15353775) === "15,353,775" && L.thousands(12) === "12", "comma groups");

// 14. Where an option works is a fraction of grasppanda.combine.reach, never a class word,
// and its tone follows from it.
const FRACTION = /^(\d+)\/(\d+)$/;
for (const [slot, members] of Object.entries(data.options)) {
  for (const option of members) {
    check(option.reach.length === data.levels.length && Number.isInteger(option.learns), `${slot}/${option.name}: reach`);
    for (const level of [...data.levels.map((item) => item.key), "level2"]) {
      const found = L.reachOf(data, option, level);
      check(found && found.fraction && found.tone, `${slot}/${option.name} ${level}: no fraction`);
      check(!/\b(universal|most|some|specific)\b/i.test(found.fraction + " " + found.tail), `${slot}/${option.name}: a class word`);
      const parts = FRACTION.exec(found.fraction.replace(/ on a par$/, ""));
      if (parts) {
        const [reached, of] = [Number(parts[1]), Number(parts[2])];
        const tone = level === "level2" ? found.tone : reached === of ? "ok" : reached ? "warn" : "no";
        check(reached <= of && found.tone === tone, `${slot}/${option.name} ${level}: ${found.fraction} ${found.tone}`);
        if (level !== "level2") check(reached === of || found.tail, `${slot}/${option.name} ${level}: who is left out`);
      }
    }
  }
}
// The fraction agrees with the tiles of Compare: under every method an option is a swap
// under (it is not what the method ships: its record shares the probe of the method as
// shipped), a tile that reached the level counts once.
for (const entry of data.lanes) {
  if (entry.key === "loss") continue;
  const grid = L.gridRows(data, entry.key);
  for (const row of grid.rows) {
    for (const level of data.levels.map((item) => item.key)) {
      const found = L.reachOf(data, row.option, level);
      const parts = FRACTION.exec(found.fraction);
      if (!parts) continue;
      const own = (m) => (m.ships[entry.key] || {}).option === row.option.name || L.isOwn(row.cells[m.name]);
      const swaps = grid.columns.filter((m) => row.cells[m.name].length && !own(m));
      const yes = swaps.filter((m) => L.levelTile(data, row.cells[m.name], level, null).state === "yes").length;
      check(yes === Number(parts[1]), `${entry.key}/${row.option.name} ${level}: ${yes} tiles, fraction ${found.fraction}`);
    }
  }
}

// 15. Slots are grouped by the methods that have them; a method without a slot has no column
// there, and the page names it instead.
const groups = L.slotGroups(data);
check(groups.map((group) => group.key).join() === "all,seed-based,hggd", `groups ${groups.map((group) => group.key)}`);
check(groups.reduce((count, group) => count + group.lanes.length, 0) === data.lanes.length, "every slot in one group");
for (const group of groups) {
  check(group.title === data.scopes[group.key], `${group.key}: title`);
  for (const entry of group.lanes) {
    const columns = L.gridRows(data, entry.key).columns.map((m) => m.name);
    const absent = L.without(data, entry.key).map((m) => m.name);
    check(columns.length + absent.length === data.methods.length, `${entry.key}: columns and absent`);
    check(absent.every((name) => !columns.includes(name)), `${entry.key}: a column of a method without the slot`);
    if (group.key === "all") check(absent.length === 0, `${entry.key}: a slot of every method`);
    if (group.key === "seed-based") check(columns.every((name) => L.method(data, name).seed_based) && columns.length === data.methods.filter((m) => m.seed_based).length, `${entry.key}: seed-based`);
    if (group.key === "hggd") check(columns.join() === "HGGD", `${entry.key}: HGGD only`);
    if (entry.counterpart) check(L.lane(data, entry.counterpart).scope !== entry.scope, `${entry.key}: its counterpart`);
  }
}

// 16. Level 2: every record says how it trained beside the method's own component, or why it
// was not compared; the tiles of Compare count as the fraction does.
for (const cell of data.cells) {
  const said = L.learnsOf(data, cell);
  check(said && (said.outcome || said.none), `${cell.id}: level 2`);
  if (data.learns.cells[cell.id]) check(data.learns.outcomes.includes(said.outcome), `${cell.id}: outcome`);
}
// The tiles say what grasppanda.combine.reach.level2 says, method by method: "4/4 on a par",
// "3/5 on a par · 2 worse", "HGGD: worse", "not compared", "—" (a slot without level 2).
// A swap that did not run at level 1, or that trained on another schedule than its
// reference, is no outcome: its tile says why, and the tail names the method.
const COMPARED = ["yes", "unstable", "no"];
let level2Rows = 0;
for (const entry of data.lanes) {
  const grid = L.gridRows(data, entry.key);
  for (const row of grid.rows) {
    const found = L.reachOf(data, row.option, "level2");
    const tiles = grid.columns.map((m) => [m, L.levelTile(data, row.cells[m.name], "level2", null, row.noRecord)]);
    const compared = tiles.filter(([, tile]) => COMPARED.includes(tile.state));
    const onPar = compared.filter(([, tile]) => tile.state === "yes").length;
    const label = `${entry.key}/${row.option.name}: tiles ${onPar}/${compared.length}, fraction ${found.fraction}`;
    const many = /^(\d+)\/(\d+) on a par(?: · |$)/.exec(found.fraction);
    const one = /^([A-Za-z-]+): (.+)$/.exec(found.fraction);
    if (many) {
      check(onPar === Number(many[1]) && compared.length === Number(many[2]), label);
    } else if (one) {
      const tile = tiles.find(([m]) => m.name === one[1]);
      check(compared.length === 1 && tile && tile[1].label === one[2], label);
    } else {
      check(found.fraction === "not compared" || found.fraction === "—", label);
      check(compared.length === 0, label);
    }
    for (const [m, tile] of tiles) {
      if (tile.state === "undecided" && /^not compared: (the swap did not run at level 1|it trained on a schedule of its own)/.test(tile.label)) {
        check(found.tail.includes("not compared under") && found.tail.includes(m.name), `${label}: ${m.name} left out without the tail saying so`);
      }
    }
    level2Rows += 1;
  }
}
check(level2Rows > 0, "level 2 rows");
// A record compared at level 2 ran at level 1; no record of a slot without level 2 is.
for (const cell of data.cells) {
  if (!data.learns.cells[cell.id]) continue;
  check(L.levelState(data, cell, "runs") === "yes", `${cell.id}: compared at level 2 although it did not run at level 1`);
  check(!["post", "schedule"].includes(cell.slot), `${cell.id}: a slot without level 2`);
}
for (const cell of data.cells.filter((item) => item.verdict === "runs-unstable")) {
  const said = L.learnsOf(data, cell);
  check(said.none && !said.outcome, `${cell.id}: an unstable swap shown with a level-2 outcome`);
}
check(L.learnsTile(data, []).state === "none", "no record, no level 2");
const words2 = L.level2Words(data);
check(words2.header === "Trains like shipped" && words2.name === "trains like shipped" && words2.meaning, "the name of level 2");
const shipped = data.methods.map((m) => L.answer(data, m.name, {}).cell).filter(Boolean);
check(shipped.length === data.methods.length && shipped.every((cell) => L.learnsOf(data, cell).own), "the methods as shipped are the references");

// 17. No method is chosen for the visitor: the gallery lists them in page order, and the
// methods carry what the gallery shows.
for (const m of data.methods) {
  check(m.venue && /\d{4}$/.test(m.venue) && m.line && m.page.startsWith("methods/") && ["depth", "RGB-D"].includes(m.inputs), `${m.name}: gallery`);
  check(m.family === (m.seed_based ? "seed-based" : "own"), `${m.name}: family`);
}
const years = data.methods.filter((m) => m.seed_based).map((m) => m.year);
check(years.every((year, index) => !index || years[index - 1] <= year), "the seed-based methods by year");
check(data.methods[data.methods.length - 1].family === "own", "the families: seed-based first");
const index = {};
for (const cell of data.cells) index[cell.key] = true;
check(Object.keys(data.index).length === Object.keys(index).length, "an index entry per key");

// 18. "Try one": each example on a method of its own, the seed-based methods in rotation (every
// one has its turn), the refusal tagged and still refused by compose, the image encoder under
// the family of its own, and Compare on a slot of every method.
{
  const withContracts = contracts ? Object.assign({}, data, { contracts }) : data;
  const seed = data.methods.filter((m) => m.seed_based).map((m) => m.name);
  const turns = new Set();
  for (let turn = 0; turn < seed.length; turn++) {
    const shown = L.examples(withContracts, turn);
    const kinds = shown.map((item) => item.kind);
    check(kinds.join() === "swap,refused,family,compare", `turn ${turn}: ${kinds}`);
    const methods = shown.filter((item) => item.cell).map((item) => item.method);
    check(new Set(methods).size === methods.length, `turn ${turn}: two examples on one method (${methods})`);
    const [swap, refused, family, compare] = shown;
    check(swap.cell.verdict === "runs" && swap.slot === "backbone" && seed.includes(swap.method) && !swap.refused, `turn ${turn}: swap`);
    check(refused.cell.verdict === "refused" && refused.refused && seed.includes(refused.method), `turn ${turn}: refusal`);
    if (contracts) {
      const now = L.contract(withContracts, refused.method, refused.cell.sets);
      check(!(now && now.ok), `turn ${turn}: compose accepts the refusal example now`);
    }
    check(family.cell.verdict === "runs" && !L.method(data, family.method).seed_based && family.slot === "image_encoder", `turn ${turn}: family`);
    check(compare.cell === null && compare.slot === L.slotGroups(data)[0].lanes[0].key, `turn ${turn}: compare`);
    for (const item of shown.filter((example) => example.cell)) {
      check(L.answer(data, item.method, { [item.slot]: item.option }).cell === item.cell, `turn ${turn}: ${item.method} ${item.slot}=${item.option} opens its record`);
    }
    turns.add(swap.method);
  }
  check(turns.size === seed.length, `every seed-based method has its turn (${[...turns]})`);
  check(L.examples(withContracts, -3).length === 4, "a negative turn");
}

// 19. Compare: options without a record of their own (adapters, wrappers, a post-processor no
// dataset feeds) come last; adapters and wrappers are no options of their own and are not
// counted (the count of every page, docs_chips.option_count), and they have no record.
for (const entry of data.lanes) {
  const rows = L.gridRows(data, entry.key).rows;
  check(rows.filter((row) => !row.option.not_option).length === entry.count, `${entry.key}: ${entry.count} options counted`);
  for (const row of rows) check(!row.option.not_option || row.noRecord, `${entry.key}/${row.option.name}: a listed non-option with a record`);
  const first = rows.findIndex((row) => row.noRecord);
  if (first < 0) continue;
  check(rows.slice(first).every((row) => row.noRecord), `${entry.key}: a row with a record after the ones without`);
  for (const row of rows.slice(first)) check(L.reachOf(data, row.option, "runs").fraction === "no record", `${entry.key}/${row.option.name}: no record`);
}
check(L.gridRows(data, "backbone").rows.some((row) => row.noRecord), "the scene encoder lists its adapters and wrappers");

// 20. What a record says: what it was measured on only where a probe ran, and that the
// combination works only where it ran.
for (const cell of data.cells) {
  const said = L.recordSummary(data, cell);
  const measuredOn = said.full.includes(data.measured_on.slice(1));
  const works = /shows that the combination (works|trains)/.test(said.full);
  check(said.text, `${cell.id}: a sentence`);
  check(measuredOn === (cell.verdict !== "refused"), `${cell.id}: measured on (${cell.verdict})`);
  check(works === (cell.verdict === "runs" || cell.verdict === "runs-empty"), `${cell.id}: works (${cell.verdict})`);
}

console.log(`ok ${checks} checks`);
