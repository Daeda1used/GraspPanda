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
          const key = L.keyOf(data, m.name, [entry.key + "=" + row.option.name]);
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
for (const level of data.levels.map((item) => item.key)) {
  for (const name of L.classesOf(data, "backbone", level)) {
    const hits = all.filter((o) => L.matches(o, { level, classes: [name] }, data));
    check(hits.length && hits.every((o) => L.classColumn(data, o, level) === name), `class ${name} at ${level}`);
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

// 7. A link carries the selection.
const back = L.fromHash("#" + L.toHash("GSNet", { backbone: "ptv3", post: "a b" }, "grid", "view"));
check(back.method === "GSNet" && back.picks.backbone === "ptv3" && back.picks.post === "a b", "hash");
check(back.view === "grid" && back.slot === "view", "hash view");
check(L.fromHash("#%E0%A4%A&m=x").method === "x", "a broken fragment is ignored");

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
// where no record exists and says so, and a refusal reaches no level.
{
  const withContracts = contracts ? Object.assign({}, data, { contracts }) : data;
  const recordStates = new Set(["yes", "no", "refused", "unstable", "undecided", "mixed"]);
  for (const entry of data.lanes) {
    const grid = L.gridRows(data, entry.key);
    for (const row of grid.rows) {
      for (const m of grid.columns) {
        const found = row.cells[m.name];
        const sets = [entry.key + "=" + row.option.name];
        const checked = entry.key === "loss" ? null : L.contract(withContracts, m.name, sets);
        for (const level of data.levels.map((item) => item.key)) {
          const tile = L.levelTile(data, found, level, found.length ? null : checked);
          if (found.length) {
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
  check(L.levelTile(data, [], "runs", null).state === "none", "nothing");
}
check(L.thousands(1262) === "1,262" && L.thousands(15353775) === "15,353,775" && L.thousands(12) === "12", "comma groups");

console.log(`ok ${checks} checks`);
