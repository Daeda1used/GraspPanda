/* The decisions of the combination explorer, as pure functions.
 *
 * Nothing here touches the page: every function takes the data of explorer.json (written
 * by tools/make_explorer.py) and a selection, and returns what the page shows. The file
 * runs in a browser (it defines `GraspPandaExplorer`) and under node (`module.exports`),
 * where tests/unit/test_explorer.py exercises it against every measured record.
 */
(function (root, factory) {
  "use strict";
  var api = factory();
  if (typeof module === "object" && module.exports) {
    module.exports = api;
  } else {
    root.GraspPandaExplorer = api;
  }
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var LOSS = "loss";

  function method(data, name) {
    for (var i = 0; i < data.methods.length; i++) {
      if (data.methods[i].name === name) return data.methods[i];
    }
    return null;
  }

  function lane(data, slot) {
    for (var i = 0; i < data.lanes.length; i++) {
      if (data.lanes[i].key === slot) return data.lanes[i];
    }
    return null;
  }

  /* The name of a loss choice: its cell of the grid without group and method,
   * `<presets>@<terms>`, which is unique under a method. */
  function lossName(cell) {
    return cell.id.split("/").slice(2).join("/");
  }

  /* The loss choices of a method: a loss sits inside a stage, so a choice is a set of
   * terms (one term, or several replaced at once), the cells of the grid that the method's
   * list names (tools/make_explorer.py, from grasppanda.combine.options); without such a
   * list, the sets recorded under the method. */
  function lossSets(data, methodName) {
    var m = method(data, methodName);
    var listed = m && m.options ? m.options[LOSS] : null;
    if (listed) {
      return listed.map(function (item) {
        var sets = typeof item === "string" ? [] : item.sets;
        var name = typeof item === "string" ? item : item.name;
        return { name: name, label: lossLabel(sets), sets: sets, reason: notProbed(m, LOSS, name) };
      });
    }
    return data.cells
      .filter(function (cell) {
        return cell.group === LOSS && cell.method === methodName;
      })
      .map(function (cell) {
        return {
          name: lossName(cell),
          label: lossLabel(cell.sets),
          sets: cell.sets,
          component: cell.component,
          cell: cell,
        };
      });
  }

  /* Why an option of a method has no cell of the grid, or "". */
  function notProbed(m, slot, name) {
    return (m && m.not_probed && m.not_probed[slot] && m.not_probed[slot][name]) || "";
  }

  /* How a set of loss terms is named on the page: by the terms it replaces, because one
   * loss fits one term and not another. The same loss on several terms is said once. */
  function lossLabel(sets) {
    var names = {};
    sets.forEach(function (item) {
      names[item.slice(item.indexOf("=") + 1)] = true;
    });
    var distinct = Object.keys(names);
    if (sets.length > 1 && distinct.length === 1) return distinct[0] + " on " + sets.length + " terms";
    return sets.join(" + ");
  }

  /* What a lane offers under a method: [{name, sets, option, reason}] with the shipped one
   * first. The options are the method's list (the lists of `grasppanda compose <config>
   * --list`); `reason` says why an option has no cell of the grid. */
  function choices(data, methodName, slot) {
    var m = method(data, methodName);
    if (!m || m.slots.indexOf(slot) < 0) return [];
    if (slot === LOSS) return lossSets(data, methodName);
    var ships = (m.ships[slot] || {}).option || "";
    var listed = m.options && m.options[slot];
    var found = listed
      ? listed.map(function (item) {
          var name = typeof item === "string" ? item : item.name;
          var sets = typeof item === "string" ? [slot + "=" + name] : item.sets;
          return { name: name, sets: sets, option: option(data, slot, name), reason: notProbed(m, slot, name) };
        })
      : (data.options[slot] || []).map(function (entry) {
          return { name: entry.name, sets: [slot + "=" + entry.name], option: entry, reason: "" };
        });
    return found.sort(function (a, b) {
      return (b.name === ships) - (a.name === ships);
    });
  }

  function option(data, slot, name) {
    var members = data.options[slot] || [];
    for (var i = 0; i < members.length; i++) {
      if (members[i].name === name) return members[i];
    }
    return null;
  }

  /* The option a slot has in the method as shipped: {option, component}. */
  function shipped(data, methodName, slot) {
    var m = method(data, methodName);
    return (m && m.ships[slot]) || { option: "", component: "" };
  }

  /* Drop picks that do not exist under the method or that name what it ships with. */
  function normalise(data, methodName, picks) {
    var m = method(data, methodName);
    var kept = {};
    if (!m) return kept;
    m.slots.forEach(function (slot) {
      var name = picks && picks[slot];
      if (!name || name === shipped(data, methodName, slot).option) return;
      var known = choices(data, methodName, slot).some(function (choice) {
        return choice.name === name;
      });
      if (known) kept[slot] = name;
    });
    return kept;
  }

  /* What a link named that this page does not know: [text]. A pick that names what the
   * method ships with is no change and is not reported. */
  function dropped(data, from) {
    var out = [];
    var named = from.method;
    var m = method(data, named);
    if (named && !m) out.push("the method `" + named + "`");
    Object.keys(from.picks || {})
      .sort()
      .forEach(function (slot) {
        var name = from.picks[slot];
        var text = "`" + slot + "=" + name + "`";
        if (!m) {
          out.push(text + " (no method chosen)");
          return;
        }
        if (m.slots.indexOf(slot) < 0) {
          out.push(text + (lane(data, slot) ? " (not a slot of " + m.name + ")" : ""));
          return;
        }
        var known = choices(data, m.name, slot).some(function (choice) {
          return choice.name === name;
        });
        if (!known && name !== shipped(data, m.name, slot).option) out.push(text);
      });
    if (from.view === "grid" && from.slot && !lane(data, from.slot)) out.push("the slot `" + from.slot + "`");
    return out;
  }

  /* The `--set` pairs of a selection, in the order of the pipeline. */
  function selectionSets(data, methodName, picks) {
    var kept = normalise(data, methodName, picks);
    var sets = [];
    data.lanes.forEach(function (entry) {
      var name = kept[entry.key];
      if (!name) return;
      choices(data, methodName, entry.key).forEach(function (choice) {
        if (choice.name === name) sets = sets.concat(choice.sets);
      });
    });
    /* A choice may name what another lane names too (a recipe that needs its encoder):
     * one `--set` each. */
    return sets.filter(function (item, index) {
      return sets.indexOf(item) === index;
    });
  }

  /* The data as the page uses it: every cell with its key, which the data file leaves out
     to stay small (it follows from the method and the sets), the name of its run and the
     options of its probe, which the file gives as indexes into `runs` and `probe_opts`, and
     the index from a key to the cell that answers it. Two records with one key are one
     experiment: the first wins, unless it shares the probe of another record and a later
     one with the same verdict has a probe of its own (tools/make_explorer.py, index_of).
     Idempotent. */
  function prepare(data) {
    if (data && data.cells && data.methods && !data.keyed) {
      var index = {};
      data.cells.forEach(function (cell, position) {
        if (cell.key === undefined) cell.key = keyOf(data, cell.method, cell.sets || []);
        if (typeof cell.run === "number" && data.runs) cell.run = data.runs[cell.run];
        if (typeof cell.probe_opts === "number" && data.probe_opts) cell.probe_opts = data.probe_opts[cell.probe_opts];
        var kept = index[cell.key];
        if (kept === undefined) index[cell.key] = position;
        else if (data.cells[kept].shares && !cell.shares && data.cells[kept].verdict === cell.verdict) index[cell.key] = position;
      });
      if (!data.index) data.index = index;
      data.keyed = true;
    }
    return data;
  }

  /* The key of a combination; tools/make_explorer.py computes the same for a record. */
  function keyOf(data, methodName, sets) {
    var m = method(data, methodName);
    var own = {};
    Object.keys(m.ships).forEach(function (slot) {
      own[slot + "=" + m.ships[slot].option] = true;
    });
    var changed = sets.filter(function (item) {
      return !own[item];
    });
    return methodName + "|" + changed.slice().sort().join("&");
  }

  function lookup(data, methodName, sets) {
    var position = data.index[keyOf(data, methodName, sets)];
    return position === undefined ? null : data.cells[position];
  }

  /* What `grasppanda compose` answered for one or two `slot=option` sets when
   * tools/make_explorer_contracts.py asked it: {ok, adapter, messages} or null. This is
   * the check of the contract rules and of the config; it is no probe. */
  function contract(data, methodName, sets) {
    var entry = data.contracts && data.contracts.methods[methodName];
    if (!entry || sets.length < 1 || sets.length > 2) return null;
    var picks = {};
    for (var i = 0; i < sets.length; i++) {
      var at = sets[i].indexOf("=");
      picks[sets[i].slice(0, at)] = sets[i].slice(at + 1);
    }
    var slots = Object.keys(picks);
    if (slots.length !== sets.length) return null;
    // A pair is stored once, under the order of the pipeline: `first|second`.
    if (slots.length === 2 && !entry.answers[slots.join("|")]) slots.reverse();
    var row = entry.answers[slots.join("|")];
    if (!row) return null;
    var places = slots.map(function (slot) {
      return (entry.options[slot] || []).indexOf(picks[slot]);
    });
    if (places.indexOf(-1) >= 0) return null;
    var position = places.length === 1 ? places[0] : places[0] * entry.options[slots[1]].length + places[1];
    var found = data.contracts.table[row[position]];
    if (!found) return null;
    return {
      ok: found.ok,
      adapter: found.adapter || "",
      kind: found.kind || "",
      messages: (found.messages || []).map(function (index) {
        return data.contracts.messages[index];
      }),
    };
  }

  /* The words of a refusal kind ("structural" -> "incompatible by design"), or "". */
  function refusalWords(data, kind) {
    return (kind && data.refusal_words && data.refusal_words[kind]) || "";
  }

  /* How an answer of the contract check is shown where no record exists. */
  function contractBadge(found) {
    if (!found) return badge(null);
    if (found.ok) {
      return {
        kind: "accepted",
        symbol: "◇",
        label: "passes the contract check" + (found.adapter ? " in the adapter " + found.adapter : "") + " · not probed",
        short: "accepted by compose, not probed",
      };
    }
    return { kind: "refused", symbol: "✗", label: "refused by the contract check", short: "refused by compose", check: true };
  }

  /* The mark of a selection: its record when one exists, else the contract check. */
  function mark(data, methodName, sets) {
    var cell = lookup(data, methodName, sets);
    if (cell) return badge(cell);
    return contractBadge(contract(data, methodName, sets));
  }

  /* The mark of several records that share a grid cell (a loss was probed in sets). */
  function badgeOf(found) {
    if (!found.length) return badge(null);
    var first = badge(found[0]);
    var same = found.every(function (cell) {
      return badge(cell).kind === first.kind;
    });
    if (same) return first;
    return {
      kind: "mixed",
      symbol: "±",
      label: found
        .map(function (cell) {
          return badge(cell).label + " as " + cell.sets.join(" ");
        })
        .join("; "),
    };
  }

  /* Whether a probe predicted no grasp on any of its frames: such a cell is greyed. */
  function noGrasp(cell) {
    return Boolean(cell && cell.grasps && cell.grasps.length && !cell.grasps.some(Boolean));
  }

  /* How a record is shown: {kind, symbol, label, empty}. `kind` is one of runs, adapter,
   * empty, unstable, refused, fails, none; it names the colour. `empty`: no grasp was
   * predicted on any probed frame (the cell is greyed). */
  function badge(cell) {
    if (!cell) return { kind: "none", symbol: "·", label: "not measured" };
    if (cell.verdict === "runs-empty") {
      var where = cell.adapter ? " in the adapter " + cell.adapter : "";
      return { kind: "empty", symbol: "∅", label: "runs" + where + ", no grasp predicted", empty: true };
    }
    if (cell.verdict === "runs" || cell.verdict === "runs-unstable") {
      var unstable = cell.verdict === "runs-unstable";
      var label = unstable ? "runs, unstable" : "runs";
      if (cell.adapter) label += " in the adapter " + cell.adapter;
      if (unstable && cell.levels && cell.levels.charAt(1) === "R") label += ", not repeated";
      if (noGrasp(cell)) label += ", no grasp predicted";
      return {
        kind: unstable ? "unstable" : cell.adapter ? "adapter" : "runs",
        symbol: unstable ? "≈" : "✓",
        label: label,
        empty: noGrasp(cell),
      };
    }
    if (cell.verdict === "refused") {
      return { kind: "refused", symbol: "✗", label: "refused" + (cell.kind ? " (" + cell.kind + ")" : "") };
    }
    if (cell.verdict === "fails") {
      var phase = cell.failure && cell.failure.phase ? " at " + cell.failure.phase : "";
      return { kind: "fails", symbol: "!", label: "fails" + phase };
    }
    return { kind: "none", symbol: "?", label: cell.verdict };
  }

  function slug(text) {
    return String(text)
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "");
  }

  /* A file name for the composed config: the method and what was chosen. */
  function outputName(methodName, sets) {
    var parts = sets.map(function (item) {
      return slug(item.split("=")[1]);
    });
    var name = [slug(methodName)].concat(parts).join("_");
    if (name.length > 72) name = slug(methodName) + "_" + sets.length + "-swaps";
    return name + ".yaml";
  }

  function composeCommand(config, sets, output) {
    var parts = ["grasppanda compose " + config];
    sets.forEach(function (item) {
      parts.push("--set " + item);
    });
    if (output) parts.push("-o " + output);
    return parts.join(" ");
  }

  function probeCommand(data, target, cell) {
    var text =
      "grasppanda probe " +
      target +
      " --iters " +
      data.probe.iterations +
      " --frames " +
      data.probe.frames;
    if (cell && cell.probe_opts) text += " --opts " + cell.probe_opts;
    return text;
  }

  /* The answer to a selection.
   *
   * state: "shipped"   the method as it is shipped (nothing changed);
   *        "measured"  a record exists for exactly this combination;
   *        "unmeasured" no record: the page shows the command that checks it, and what
   *                    was measured for each changed slot alone.
   */
  function answer(data, methodName, picks) {
    var m = method(data, methodName);
    var kept = normalise(data, methodName, picks);
    var sets = selectionSets(data, methodName, kept);
    var cell = lookup(data, methodName, sets);
    var output = sets.length ? outputName(methodName, sets) : "";
    var parts = [];
    data.lanes.forEach(function (entry) {
      if (!kept[entry.key]) return;
      var own = choices(data, methodName, entry.key).filter(function (choice) {
        return choice.name === kept[entry.key];
      })[0];
      var alone = lookup(data, methodName, own.sets);
      parts.push({
        slot: entry.key,
        name: own.name,
        sets: own.sets,
        option: own.option || null,
        reason: own.reason || "",
        label: own.label || own.name,
        cell: alone,
        badge: alone ? badge(alone) : contractBadge(contract(data, methodName, own.sets)),
      });
    });
    // Without a record the page may still hold what compose answered (one or two
    // changes). That is shown as the contract check, never as a verdict of a probe.
    var checked = !cell && sets.length ? contract(data, methodName, sets) : null;
    var pairs = [];
    if (!cell && !checked && parts.length > 2) {
      for (var i = 0; i < parts.length; i++) {
        for (var j = i + 1; j < parts.length; j++) {
          var both = parts[i].sets.concat(parts[j].sets);
          var record = lookup(data, methodName, both);
          var known = record ? null : contract(data, methodName, both);
          if (record || known) pairs.push({ sets: both, badge: record ? badge(record) : contractBadge(known) });
        }
      }
    }
    // compose writes no file when it refuses: there is nothing to probe then.
    var refused = (cell && cell.verdict === "refused") || (checked && !checked.ok);
    return {
      method: m,
      picks: kept,
      sets: sets,
      state: !sets.length ? "shipped" : cell ? "measured" : "unmeasured",
      cell: cell,
      contract: checked,
      badge: cell || !checked ? badge(cell) : contractBadge(checked),
      parts: parts,
      pairs: pairs,
      output: output,
      compose: sets.length ? composeCommand(m.config, sets, output) : "",
      probe: refused ? "" : probeCommand(data, sets.length ? output : m.config, cell),
    };
  }

  /* Why an option has no record of its own (an adapter compose inserts, a point-painting
   * wrapper a config names by hand: grasppanda.combine.reach says "no record"), or "". */
  function noRecord(data, entry) {
    var found = reachOf(data, entry, "runs");
    return found && found.fraction === "no record" ? found.tail || "no record of its own" : "";
  }

  /* The cells of the grid of one slot: rows are options, columns are methods. A cell is
   * the record of the option's request under the method: its one `slot=option`, or the sets
   * the method's list gives it (a recipe that needs its scene encoder). An option without a
   * record of its own (noRecord) comes last, with `noRecord` saying why. */
  function gridRows(data, slot) {
    var columns = data.methods.filter(function (m) {
      return m.slots.indexOf(slot) >= 0;
    });
    var requests = {};
    columns.forEach(function (m) {
      requests[m.name] = {};
      if (slot === LOSS) return;
      choices(data, m.name, slot).forEach(function (choice) {
        requests[m.name][choice.name] = choice.sets;
      });
    });
    var rows = (data.options[slot] || []).map(function (entry) {
      var cellsOf = {};
      columns.forEach(function (m) {
        if (slot === LOSS) {
          cellsOf[m.name] = (entry.cells || [])
            .map(function (position) {
              return data.cells[position];
            })
            .filter(function (cell) {
              return cell.method === m.name;
            });
        } else {
          var found = lookup(data, m.name, requests[m.name][entry.name] || [slot + "=" + entry.name]);
          cellsOf[m.name] = found ? [found] : [];
        }
      });
      return { option: entry, cells: cellsOf, noRecord: noRecord(data, entry) };
    });
    var measured = rows.filter(function (row) {
      return !row.noRecord;
    });
    return {
      columns: columns,
      rows: measured.concat(
        rows.filter(function (row) {
          return row.noRecord;
        })
      ),
    };
  }

  /* Whether the records of a cell are the method as shipped: the option is what it ships
   * (the same config as its reference), so it is no swap there and no fraction counts it. */
  function isOwn(found) {
    return Boolean(
      found &&
        found.length &&
        found.every(function (cell) {
          return cell.own;
        })
    );
  }

  /* The tile of what a method ships, at every level and at level 2. */
  var OWN_TILE = { state: "own", label: "what the method ships: this is the method as shipped", contract: false, own: true };

  /* The position of a level in the data's order of levels (composes, runs, ...). */
  function levelIndex(data, level) {
    var keys = (data.levels || []).map(function (item) {
      return item.key;
    });
    var at = keys.indexOf(level || "runs");
    return at < 0 ? 1 : at;
  }

  /* The key of level 2 in Compare: a fifth column after the four levels of a record. */
  var LEVEL2 = "level2";

  /* Where an option works at a level (grasppanda.combine.reach, exported by
   * tools/make_explorer.py): {fraction, tone, tail}. The level `level2` is level 2: how it
   * trains beside each method's own component. Null without. */
  function reachOf(data, entry, level) {
    if (!entry || !data.reach) return null;
    var at = level === LEVEL2 ? entry.learns : entry.reach ? entry.reach[levelIndex(data, level)] : undefined;
    var row = at === undefined || at === null ? null : data.reach[at];
    return row ? { fraction: row[0], tone: row[1], tail: row[2] } : null;
  }

  function toneOf(data, entry, level) {
    var found = reachOf(data, entry, level);
    return found ? found.tone : "";
  }

  /* The tones of a fraction, from every method it swaps into to none, then undecided. */
  var TONE_ORDER = ["ok", "warn", "no", "muted"];

  /* Filters of the grid: {q, level, tones: [..], extra: bool, weights: bool}; the tones
   * are those of the level (runs by default). */
  function matches(entry, filter, data) {
    filter = filter || {};
    if (filter.tones && filter.tones.length && filter.tones.indexOf(toneOf(data || {}, entry, filter.level)) < 0) {
      return false;
    }
    if (filter.extra && !(entry.extras && entry.extras.length)) return false;
    if (filter.weights && !entry.weights) return false;
    var q = (filter.q || "").trim().toLowerCase();
    if (!q) return true;
    var hay = [entry.name, entry.component, entry.summary, entry.source].join(" ").toLowerCase();
    return q.split(/\s+/).every(function (word) {
      return hay.indexOf(word) >= 0;
    });
  }

  /* The tones that occur among the options of a slot at a level, in TONE_ORDER. */
  function tonesOf(data, slot, level) {
    var used = {};
    (data.options[slot] || []).forEach(function (entry) {
      var tone = toneOf(data, entry, level);
      if (tone) used[tone] = true;
    });
    return TONE_ORDER.filter(function (tone) {
      return used[tone];
    });
  }

  /* What the page calls level 2, as docs_chips names it: {name, header, meaning}. */
  function level2Words(data) {
    var side = data.learns || {};
    return {
      name: side.name || "trains like shipped",
      header: side.header || "Trains like shipped",
      meaning: side.meaning || "",
    };
  }

  /* Level 2 of a record (explorer_learns.json): {outcome, tone, rank, reasons} when it was
   * compared beside the method's own component (rank 0 is the best outcome); {none, own}
   * when it was not, with why (the rules of grasppanda.combine.reach.level2: the method as
   * shipped, a slot without level 2, a swap that did not run at level 1, a run on another
   * schedule than the reference's); null when the page does not hold level 2. */
  function learnsOf(data, cell) {
    var side = data.learns;
    if (!side || !cell) return null;
    var found = side.cells[cell.id];
    if (found) {
      return { outcome: side.outcomes[found[0]], tone: side.tones[found[0]], rank: found[0], reasons: found.slice(1) };
    }
    var why = side.none[cell.id];
    var words = side.none_words;
    var own = why === 0 || Boolean(cell.own);
    return { none: own ? words[0] : why === undefined ? "not measured at level 2" : words[why], own: own, rank: -1 };
  }

  /* A tone of level 2 as a state of a tile of the grid. */
  var LEVEL2_TILE = { ok: "yes", warn: "unstable", no: "no", muted: "undecided" };

  /* The tile of a grid cell at level 2: {state, label, own}. Of several records (a loss in
   * several terms), those that replace one term decide, as grasppanda.combine.reach reads
   * them, and the worst outcome is shown. */
  function learnsTile(data, found) {
    if (!found || !found.length) return { state: "none", label: "no record", contract: false };
    if (isOwn(found)) return OWN_TILE;
    if (!data.learns) return { state: "none", label: "level 2 is not loaded", contract: false };
    var single = found.filter(function (cell) {
      return (cell.sets || []).length <= 1;
    });
    var pool = single.length ? single : found;
    var worst = null;
    var others = [];
    pool.forEach(function (cell) {
      var said = learnsOf(data, cell);
      if (said && !said.none) {
        if (worst === null || said.rank > worst.rank) worst = said;
      } else if (said) {
        others.push(said);
      }
    });
    if (worst) return { state: LEVEL2_TILE[worst.tone] || "undecided", label: worst.outcome, contract: false };
    var first = others[0] || { none: "", own: false };
    if (first.own) return OWN_TILE;
    return { state: "undecided", label: "not compared: " + first.none, contract: false };
  }

  /* The slots in groups by the methods that have them: every method, the seed-based
   * methods, HGGD only (docs_chips.scope): [{key, title, lanes}]. */
  function slotGroups(data) {
    var order = ["all", "seed-based", "hggd"];
    var groups = {};
    data.lanes.forEach(function (entry) {
      var key = entry.scope || "other";
      (groups[key] = groups[key] || []).push(entry);
    });
    var keys = order
      .filter(function (key) {
        return groups[key];
      })
      .concat(
        Object.keys(groups).filter(function (key) {
          return order.indexOf(key) < 0;
        })
      );
    return keys.map(function (key) {
      return { key: key, title: (data.scopes || {})[key] || key, lanes: groups[key] };
    });
  }

  /* The methods that do not have a slot, in page order. */
  function without(data, slot) {
    return data.methods.filter(function (m) {
      return m.slots.indexOf(slot) < 0;
    });
  }

  /* The state of a record at a level: "yes", "no" or why it is undecided. A record
   * holds its four levels as a word of four letters (tools/make_explorer.py). */
  function levelState(data, cell, level) {
    if (!cell || !cell.levels) return "";
    var code = cell.levels.charAt(levelIndex(data, level));
    return (data.level_states || {})[code] || code;
  }

  /* What a record shows at a level: {state, label}. `state` is "yes", "no", "refused"
   * (compose refused it, so it reached no level), "unstable" (an unstable probe that did
   * not reach the level) or "undecided" (the label says why). */
  function recordTile(data, cell, level) {
    if (cell.verdict === "refused") {
      return { state: "refused", label: "refused by compose" + (cell.kind ? " (" + cell.kind + ")" : ""), contract: false };
    }
    var said = levelState(data, cell, level);
    if (said === "yes") return { state: "yes", label: "yes", contract: false };
    if (cell.verdict === "runs-unstable" && (said === "no" || said === "unstable, not repeated")) {
      return { state: "unstable", label: said === "no" ? "no: unstable, and so was its repeat" : said, contract: false };
    }
    if (said === "no") return { state: "no", label: "no", contract: false };
    return { state: "undecided", label: said || "not checked", contract: false };
  }

  /* The tile of a cell of the grid at a level: {state, label, contract}. `found` are the
   * records of the cell; `checked` is the answer of the contract check for the single swap
   * (contract()), used only where no record exists, and then marked `contract: true` with
   * the states "accepted" or "check-refused", which no record ever gets. What the method
   * ships is "own" at every level (it is no swap there). Records that disagree at the level
   * give "mixed"; nothing at all gives "none", and "muted" for an option without a record of
   * its own (`noRecordWhy`, from gridRows: an adapter or a wrapper). At level 2 the tile is
   * that of learnsTile(). */
  function levelTile(data, found, level, checked, noRecordWhy) {
    if (isOwn(found)) return OWN_TILE;
    if (noRecordWhy && !(found && found.length)) {
      return { state: "muted", label: "no record of its own", contract: false };
    }
    if (level === LEVEL2) return learnsTile(data, found);
    if (found && found.length) {
      var tiles = found.map(function (cell) {
        return recordTile(data, cell, level);
      });
      var same = tiles.every(function (tile) {
        return tile.state === tiles[0].state;
      });
      return same ? tiles[0] : { state: "mixed", label: "its records disagree", contract: false };
    }
    if (checked) {
      return checked.ok
        ? { state: "accepted", label: "accepted by compose, not probed", contract: true }
        : { state: "check-refused", label: "refused by compose, not probed", contract: true };
    }
    return { state: "none", label: "no answer on this page", contract: false };
  }

  /* A count with comma digit groups, as the page prints its numbers: 1,262. */
  function thousands(value) {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }

  /* The levels a record reached, as one line: "composes ✓ · runs ✓ · learns ? · ...". */
  function levelsText(data, cell) {
    if (!cell || !cell.levels) return "";
    var marks = { yes: "✓", no: "✗" };
    return (data.levels || [])
      .map(function (level) {
        var state = levelState(data, cell, level.key);
        return level.key + " " + (marks[state] || state);
      })
      .join(" · ");
  }

  /* A selection as the fragment of a link, and back. No method chosen: no `m=` and no
   * picks (the page opens on the gallery of methods, or on the view the link names). */
  function toHash(methodName, picks, view, slot) {
    var parts = [];
    if (methodName) {
      parts.push("m=" + encodeURIComponent(methodName));
      Object.keys(picks || {})
        .sort()
        .forEach(function (key) {
          parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(picks[key]));
        });
    }
    if (view === "grid") parts.push("grid=" + encodeURIComponent(slot || ""));
    else if (view === "run") parts.push("run=1");
    return parts.join("&");
  }

  function fromHash(text) {
    var out = { method: "", picks: {}, view: "build", slot: "" };
    String(text || "")
      .replace(/^#/, "")
      .split("&")
      .forEach(function (pair) {
        var at = pair.indexOf("=");
        if (at < 1) return;
        var key, value;
        try {
          key = decodeURIComponent(pair.slice(0, at));
          value = decodeURIComponent(pair.slice(at + 1));
        } catch (error) {
          return;
        }
        if (key === "m") out.method = value;
        else if (key === "run") out.view = "run"; /* the Run panel; "run" is no slot */
        else if (key === "grid") {
          out.view = "grid";
          out.slot = value;
        } else out.picks[key] = value;
      });
    return out;
  }

  function number(value) {
    return String(value).replace(/\B(?=(\d{3})+(?!\d))/g, " ");
  }

  /* The numbers of a record that are honest to show, as [label, text] pairs. */
  function facts(cell) {
    var out = [];
    if (!cell || cell.parameters === undefined) return out;
    out.push(["parameters", number(cell.parameters)]);
    if (cell.loss) out.push(["total loss, first → last iteration", cell.loss[0] + " → " + cell.loss[1]]);
    if (cell.grasps && cell.grasps.length) {
      out.push(["grasps predicted per frame", cell.grasps.join(", ")]);
    }
    if (cell.fall) {
      out.push([
        "fall of every loss term",
        Object.keys(cell.fall)
          .map(function (term) {
            var value = cell.fall[term];
            return term + " " + (value === null ? "never active" : Math.round(value * 100) + " %");
          })
          .join(", "),
      ]);
    } else if (cell.levels) {
      out.push(["fall of every loss term", "not checked (a record of the first judge)"]);
    }
    if (cell.post && cell.post.grasps) {
      out.push([
        "grasps before → after " + cell.post.steps.join(" → "),
        cell.post.grasps
          .map(function (frame) {
            return frame.join(" → ");
          })
          .join(";  "),
      ]);
    }
    return out;
  }

  /* What a record says in one sentence, and what stands behind it (its tooltip):
   * {text, full}. What it was measured on is said only where a probe ran, and that the
   * combination works only where it ran (runs, runs-empty); a refusal built nothing. */
  function recordSummary(data, cell) {
    var it = data.probe.iterations;
    var fr = data.probe.frames;
    var measuredOn = capitalise(data.measured_on) + ".";
    var said = {
      runs: ["It builds, trains " + it + " iterations on " + fr + " real frames and predicts grasps.", measuredOn + " It shows that the combination works, not how well."],
      "runs-empty": ["It builds and trains, but predicted no grasp on the test frames.", measuredOn + " It shows that the combination trains, not how well; it predicted no grasp."],
      "runs-unstable": ["It builds and trains, but the short run was unstable.", measuredOn + " An unstable run does not show that the combination works."],
      fails: ["It passed the config check, then stopped with an error.", measuredOn + " The error is below."],
      refused: ["Refused by compose: nothing was built.", "The answer of `grasppanda compose`: it checks that the stages fit before anything is built. Nothing was trained."],
    }[cell.verdict];
    if (said) return { text: said[0], full: said[1] };
    return { text: capitalise(badge(cell).label) + ".", full: "" };
  }

  function capitalise(text) {
    text = String(text || "");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  /* Preferred options of the examples of the gallery, best first: a scene encoder from
   * another paper, another method's head bundle (incompatible by design), a pretrained
   * image encoder. */
  var EXAMPLE_PREFER = {
    swap: ["backbone=ptv3"],
    refused: ["head=economic", "grouper=depth-embed", "assigner=compact-label"],
    family: ["image_encoder=dinov2-small-reg"],
  };

  /* The selections the gallery offers to start from, each on a method of its own: a scene
   * encoder from another paper that runs, a swap compose refuses (`refused`), an image
   * encoder of a family of its own, and Compare on the first slot of every method. The two
   * seed-based examples take the methods of the skeleton in rotation: `turn` (the page
   * passes the day) names the method of the swap, and the refusal goes to the method before
   * it, so no method is shown twice and every method has its turn.
   * [{kind, slot, option, method, cell, refused}], kind "swap" | "refused" | "family" |
   * "compare" (no cell; `method` is the title of the group of every method). */
  function examples(data, turn) {
    var seed = data.methods.filter(function (m) {
      return m.seed_based;
    }).map(function (m) {
      return m.name;
    });
    var others = data.methods.filter(function (m) {
      return !m.seed_based;
    }).map(function (m) {
      return m.name;
    });
    var n = seed.length;
    var start = n ? (((turn || 0) % n) + n) % n : 0;
    function rotation(from, step) {
      var out = [];
      for (var i = 0; i < n; i++) out.push(seed[(((from + step * i) % n) + n) % n]);
      return out;
    }
    var used = {};
    function first(names, test, prefer) {
      for (var i = 0; i < names.length; i++) {
        var name = names[i];
        if (used[name]) continue;
        var found = data.cells.filter(function (cell) {
          return cell.method === name && cell.sets.length === 1 && test(cell);
        });
        if (!found.length) continue;
        var rank = function (cell) {
          var at = prefer.indexOf(cell.sets[0]);
          return at < 0 ? prefer.length : at;
        };
        found.sort(function (a, b) {
          return rank(a) - rank(b) || data.cells.indexOf(a) - data.cells.indexOf(b);
        });
        used[name] = true;
        return found[0];
      }
      return null;
    }
    var picked = [
      ["swap", first(rotation(start, 1), function (cell) {
        return cell.group === "encoder" && cell.verdict === "runs";
      }, EXAMPLE_PREFER.swap)],
      ["refused", first(rotation(start - 1, -1), function (cell) {
        if (cell.verdict !== "refused" || (cell.group !== "stage" && cell.group !== "encoder")) return false;
        var now = contract(data, cell.method, cell.sets);
        return !(now && now.ok);
      }, EXAMPLE_PREFER.refused)],
      ["family", first(others, function (cell) {
        return cell.group === "family-slot" && cell.verdict === "runs";
      }, EXAMPLE_PREFER.family)],
    ];
    var out = [];
    picked.forEach(function (item) {
      var cell = item[1];
      if (!cell) return;
      var at = cell.sets[0].indexOf("=");
      out.push({ kind: item[0], slot: cell.sets[0].slice(0, at), option: cell.sets[0].slice(at + 1), method: cell.method, cell: cell, refused: cell.verdict === "refused" });
    });
    var everyone = slotGroups(data)[0];
    if (everyone && everyone.key === "all" && everyone.lanes.length) {
      out.push({ kind: "compare", slot: everyone.lanes[0].key, option: "", method: everyone.title, cell: null, refused: false });
    }
    return out;
  }

  var api = {
    prepare: prepare,
    answer: answer,
    badge: badge,
    badgeOf: badgeOf,
    choices: choices,
    contract: contract,
    contractBadge: contractBadge,
    dropped: dropped,
    lossLabel: lossLabel,
    lossName: lossName,
    mark: mark,
    reachOf: reachOf,
    toneOf: toneOf,
    tonesOf: tonesOf,
    learnsOf: learnsOf,
    learnsTile: learnsTile,
    level2Words: level2Words,
    examples: examples,
    isOwn: isOwn,
    noRecord: noRecord,
    recordSummary: recordSummary,
    refusalWords: refusalWords,
    slotGroups: slotGroups,
    without: without,
    levelState: levelState,
    levelTile: levelTile,
    recordTile: recordTile,
    thousands: thousands,
    levelsText: levelsText,
    noGrasp: noGrasp,
    notProbed: notProbed,
    composeCommand: composeCommand,
    facts: facts,
    fromHash: fromHash,
    gridRows: gridRows,
    keyOf: keyOf,
    lane: lane,
    lookup: lookup,
    matches: matches,
    method: method,
    normalise: normalise,
    number: number,
    option: option,
    outputName: outputName,
    probeCommand: probeCommand,
    selectionSets: selectionSets,
    shipped: shipped,
    toHash: toHash,
  };
  // Every entry point that is given the data prepares it first (a no-op after the first
  // time, and for arguments that are not the data).
  Object.keys(api).forEach(function (name) {
    var plain = api[name];
    if (name === "prepare") return;
    api[name] = function () {
      prepare(arguments[0]);
      return plain.apply(null, arguments);
    };
  });
  return api;
});
