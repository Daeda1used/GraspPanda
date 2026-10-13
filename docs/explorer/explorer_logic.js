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

  /* The loss sets recorded under a method: a loss sits inside a stage, and a probe
   * replaced several terms at once, so one recorded set is one choice of the lane. */
  function lossSets(data, methodName) {
    return data.cells
      .filter(function (cell) {
        return cell.group === LOSS && cell.method === methodName;
      })
      .map(function (cell) {
        return {
          name: cell.variant,
          label: lossLabel(cell.sets),
          sets: cell.sets,
          component: cell.component,
          cell: cell,
        };
      });
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

  /* What a lane offers under a method: [{name, sets, option}] with the shipped one first. */
  function choices(data, methodName, slot) {
    var m = method(data, methodName);
    if (!m || m.slots.indexOf(slot) < 0) return [];
    if (slot === LOSS) return lossSets(data, methodName);
    var ships = (m.ships[slot] || {}).option || "";
    return (data.options[slot] || [])
      .map(function (option) {
        return { name: option.name, sets: [slot + "=" + option.name], option: option };
      })
      .sort(function (a, b) {
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
    m = m || data.methods[0];
    Object.keys(from.picks || {})
      .sort()
      .forEach(function (slot) {
        var name = from.picks[slot];
        var text = "`" + slot + "=" + name + "`";
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
    return sets;
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
      messages: (found.messages || []).map(function (index) {
        return data.contracts.messages[index];
      }),
    };
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

  /* How a record is shown: {kind, symbol, label}. `kind` is one of runs, adapter,
   * unstable, refused, fails, none; it names the colour. */
  function badge(cell) {
    if (!cell) return { kind: "none", symbol: "·", label: "not measured" };
    if (cell.verdict === "runs" || cell.verdict === "runs-unstable") {
      var unstable = cell.verdict === "runs-unstable";
      var label = unstable ? "runs, unstable" : "runs";
      if (cell.adapter) label += " in the adapter " + cell.adapter;
      return {
        kind: unstable ? "unstable" : cell.adapter ? "adapter" : "runs",
        symbol: unstable ? "≈" : "✓",
        label: label,
      };
    }
    if (cell.verdict === "refused") return { kind: "refused", symbol: "✗", label: "refused" };
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

  /* The cells of the grid of one slot: rows are options, columns are methods. */
  function gridRows(data, slot) {
    var columns = data.methods.filter(function (m) {
      return m.slots.indexOf(slot) >= 0;
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
          var found = lookup(data, m.name, [slot + "=" + entry.name]);
          cellsOf[m.name] = found ? [found] : [];
        }
      });
      return { option: entry, cells: cellsOf };
    });
    return { columns: columns, rows: rows };
  }

  /* Filters of the grid: {q, classes: [..], extra: bool, weights: bool}. */
  function matches(entry, filter) {
    filter = filter || {};
    if (filter.classes && filter.classes.length && filter.classes.indexOf(entry["class"]) < 0) {
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

  /* The classes that occur among the options of a slot, in the order of the definition. */
  function classesOf(data, slot) {
    var order = ["universal", "most", "some", "specific", "none"];
    var used = {};
    (data.options[slot] || []).forEach(function (entry) {
      if (entry["class"]) used[entry["class"]] = true;
    });
    return order.filter(function (name) {
      return used[name];
    });
  }

  /* A selection as the fragment of a link, and back. */
  function toHash(methodName, picks, view, slot) {
    var parts = ["m=" + encodeURIComponent(methodName)];
    Object.keys(picks || {})
      .sort()
      .forEach(function (key) {
        parts.push(encodeURIComponent(key) + "=" + encodeURIComponent(picks[key]));
      });
    if (view === "grid") parts.push("grid=" + encodeURIComponent(slot || ""));
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

  return {
    answer: answer,
    badge: badge,
    badgeOf: badgeOf,
    choices: choices,
    contract: contract,
    contractBadge: contractBadge,
    dropped: dropped,
    lossLabel: lossLabel,
    mark: mark,
    classesOf: classesOf,
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
});
