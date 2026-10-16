/* The combination explorer: the page around explorer_logic.js.
 *
 * No framework, no network: the data is the JSON element of the page (or `fetch`ed from
 * `data-src` when the page is served by `grasppanda explore`). What a selection shows is
 * decided in explorer_logic.js; this file only draws it.
 */
(function () {
  "use strict";

  var L = window.GraspPandaExplorer;
  var root = document.getElementById("gp-explorer");
  if (!L || !root) return;

  var data = null;
  var state = {
    method: "",
    picks: {},
    open: "",
    view: "build",
    slot: "",
    laneQuery: "",
    filter: { q: "", level: "runs", classes: [], extra: false, weights: false },
    yamlTab: "",
    expanded: {},
    folds: {},
    flash: "",
    notice: [],
    cellMenu: "",
    gridFocus: "",
    focusKey: "",
  };
  var pushNext = false;
  var barWatch = null;
  /* How many options a lane shows before "show all". */
  var FEW = 14;
  var docsBase = root.getAttribute("data-docs") || "../";
  /* Set by `grasppanda explore`: the address of the compose check of this installation. */
  var liveBase = root.getAttribute("data-live") || "";
  var live = { key: "", status: "", report: null, error: "" };
  var saving = { key: "", name: "", status: "", message: "" };
  var tip = null;

  // ------------------------------------------------------------------------ helpers

  function h(tag, attrs, children) {
    var node = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (name) {
      var value = attrs[name];
      if (value === false || value === null || value === undefined) return;
      if (name === "class") node.className = value;
      else if (name === "text") node.textContent = value;
      else if (name.slice(0, 2) === "on") node.addEventListener(name.slice(2), value);
      else node.setAttribute(name, value === true ? "" : value);
    });
    (children || []).forEach(function (child) {
      if (child === null || child === undefined || child === false) return;
      node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    });
    return node;
  }

  /* Text with `code` spans, as the records write them. */
  function rich(text, tag) {
    var node = h(tag || "span", {});
    String(text)
      .split(/(`[^`]+`)/)
      .forEach(function (piece) {
        if (piece.length > 2 && piece[0] === "`" && piece[piece.length - 1] === "`") {
          node.appendChild(h("code", { text: piece.slice(1, -1) }));
        } else if (piece) {
          node.appendChild(document.createTextNode(piece));
        }
      });
    return node;
  }

  function mark(badge, extra) {
    return h("span", {
      class: "gx-mark k-" + badge.kind + (extra ? " " + extra : ""),
      "aria-hidden": "true",
      text: badge.symbol,
    });
  }

  function copyText(text, button) {
    function done(ok) {
      var old = button.getAttribute("data-label");
      button.textContent = ok ? "Copied" : "Press Ctrl+C";
      window.setTimeout(function () {
        button.textContent = old;
      }, 1400);
    }
    function fallback() {
      var area = h("textarea", { class: "gx-offscreen" });
      area.value = text;
      document.body.appendChild(area);
      area.select();
      var ok = false;
      try {
        ok = document.execCommand("copy");
      } catch (error) {
        ok = false;
      }
      document.body.removeChild(area);
      done(ok);
    }
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () {
        done(true);
      }, fallback);
    } else {
      fallback();
    }
  }

  function codeBlock(label, text, key, pre) {
    var button = h("button", {
      type: "button",
      class: "gx-copy",
      "data-key": "copy-" + key,
      "data-label": "Copy",
      "aria-label": "Copy " + label,
      text: "Copy",
      onclick: function () {
        copyText(text, button);
      },
    });
    return h("div", { class: "gx-code" }, [
      h("div", { class: "gx-code-bar" }, [h("span", { text: label }), button]),
      h("div", { class: "gx-pre" + (pre ? " is-pre" : ""), tabindex: "0", role: "group", "aria-label": label, text: text }),
    ]);
  }

  /* A page of the documentation. The site uses directory addresses, which a browser does
   * not resolve when the site is opened from a file. */
  function docHref(path) {
    return docsBase + path + "/" + (window.location.protocol === "file:" ? "index.html" : "");
  }

  function pageLink(stem, text) {
    return h("a", { href: docHref("slots/" + stem), text: text });
  }

  function docLinks(pages) {
    if (!pages.length) return null;
    return h("p", { class: "gx-fine gx-docs" }, [h("span", { text: "Docs: " })].concat(
      pages.slice(0, 4).map(function (page, index) {
        return h("span", {}, [index ? h("span", { class: "gx-sep", text: " · " }) : null, pageLink(page.stem, page.title.replace(/`/g, ""))]);
      })
    ));
  }

  /* An option name that may break after a hyphen, and nowhere else. */
  function breakable(name) {
    var node = h("code", {});
    name.split("-").forEach(function (piece, index, all) {
      node.appendChild(document.createTextNode(piece + (index < all.length - 1 ? "-" : "")));
      if (index < all.length - 1) node.appendChild(document.createElement("wbr"));
    });
    return node;
  }

  /* A part of the answer that is closed until asked for; it stays as the visitor left it. */
  function fold(key, summary, children) {
    var open = Boolean(state.folds[key]);
    return h("div", { class: "gx-fold" + (open ? " is-open" : "") }, [
      h("button", {
        type: "button",
        class: "gx-fold-head",
        "data-key": "fold-" + key,
        "aria-expanded": open ? "true" : "false",
        onclick: function () {
          var folds = {};
          Object.keys(state.folds).forEach(function (name) {
            folds[name] = state.folds[name];
          });
          folds[key] = !open;
          set({ folds: folds });
        },
      }, [h("span", { text: summary }), h("span", { class: "gx-caret", "aria-hidden": "true", text: open ? "\u2013" : "+" })]),
      open ? h("div", { class: "gx-fold-body" }, children) : null,
    ]);
  }

  /* Change the state and draw. `push` makes the change a step of the browser history. */
  function set(patch, push) {
    Object.keys(patch).forEach(function (name) {
      state[name] = patch[name];
    });
    if (push) pushNext = true;
    render();
  }

  function pick(slot, name) {
    var picks = {};
    Object.keys(state.picks).forEach(function (key) {
      picks[key] = state.picks[key];
    });
    if (name) picks[slot] = name;
    else delete picks[slot];
    set({ picks: L.normalise(data, state.method, picks), flash: slot, notice: [] }, true);
  }

  // -------------------------------------------------------------------------- header

  function header() {
    var counts = data.counts;
    var items = data.verdict_order.map(function (name) {
      var badge = L.badge({ verdict: name });
      return h("li", {}, [mark(badge), h("b", { text: L.number(counts.by_verdict[name] || 0) }), h("span", { class: "gx-word", text: " " + badge.label })]);
    });
    return h("div", { class: "gx-head" }, [
      h("ul", { class: "gx-stats", "aria-label": "Recorded combinations by verdict" }, [
        h("li", { class: "gx-total" }, [h("b", { text: L.number(counts.records) }), " combinations on record"]),
      ].concat(items)),
      h("p", { class: "gx-fine", text: data.no_accuracy }),
      data.caveat ? h("p", { class: "gx-fine gx-caveat", text: "Class: " + data.caveat + "." }) : null,
    ]);
  }

  /* What stands behind the numbers of the page, computed from the data. */
  function footer() {
    var counts = data.counts;
    var lines = [
      h("li", {}, [
        h("b", { text: L.number(counts.probed) }),
        " of the " + L.number(counts.records) + " records were trained on real frames by a probe of their own; ",
        h("b", { text: L.number(counts.shared_probe) }),
        " compose to the same config as another record and share its probe; ",
        h("b", { text: L.number(counts.by_verdict.refused || 0) }),
        " were refused by compose, so nothing was built.",
      ]),
      h("li", {}, [
        "The probe grid lists " + L.number(counts.grid) + " combinations; ",
        h("b", { text: L.number(counts.grid - counts.grid_without_record) }),
        " have a record. Of its " + L.number(counts.stage_sets.listed) + " sets of several stages, ",
        h("b", { text: L.number(counts.stage_sets.recorded) }),
        " were probed.",
      ]),
    ];
    if (data.contracts) {
      var c = data.contracts.counts;
      lines.push(h("li", {}, [
        "Contract check: ",
        h("b", { text: L.number(c.answers) }),
        " answers of ",
        h("code", { text: "grasppanda compose" }),
        " for one option alone and for two options of two slots (" + L.number(c.accepted) + " accepted, " + L.number(c.refused) +
          " refused). No probe stands behind them: \u201caccepted\u201d says that the contracts fit, not that it trains.",
      ]));
    }
    lines.push(h("li", { text: data.no_accuracy }));
    return h("div", { class: "gx-foot" }, [h("div", { class: "gx-label", text: "Behind the numbers" }), h("ul", {}, lines)]);
  }

  function tabs() {
    /* A phone shows the short names, so that the three tabs fit side by side. */
    var views = [["build", "Build a combination", "Build"], ["grid", "All options of a slot", "All options"], ["run", "Run a method", "Run"]];
    function go(view) {
      state.focusKey = "tab-" + view;
      set({ view: view }, true);
    }
    return h("div", { class: "gx-tabs", role: "tablist", "aria-label": "View" }, views.map(function (item, index) {
      var on = state.view === item[0];
      return h("button", {
        type: "button",
        role: "tab",
        class: "gx-tab",
        "data-key": "tab-" + item[0],
        "aria-selected": on ? "true" : "false",
        tabindex: on ? "0" : "-1",
        "aria-label": item[1],
        onclick: function () {
          go(item[0]);
        },
        onkeydown: function (event) {
          var step = { ArrowRight: 1, ArrowLeft: -1, Home: -index, End: views.length - 1 - index }[event.key];
          if (step === undefined) return;
          event.preventDefault();
          go(views[(index + step + views.length) % views.length][0]);
        },
      }, [h("span", { class: "gx-tab-long", text: item[1] }), h("span", { class: "gx-tab-short", "aria-hidden": "true", text: item[2] })]);
    }));
  }

  // ----------------------------------------------------------------------- live mode

  function post(action, payload) {
    return window
      .fetch(liveBase + action, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })
      .then(function (response) {
        return response.json();
      });
  }

  /* Ask this installation to compose the selection, once per selection. */
  function ensureLive(result) {
    var key = result.method.name + "|" + result.sets.join("&");
    if (!liveBase || !result.sets.length || live.key === key) return;
    live = { key: key, status: "loading", report: null, error: "" };
    saving = { key: key, name: result.output, status: "", message: "" };
    post("compose", { method: result.method.name, sets: result.sets }).then(
      function (report) {
        if (live.key !== key) return;
        live.status = report.error ? "error" : "done";
        live.report = report;
        live.error = report.error || "";
        render();
      },
      function (error) {
        if (live.key !== key) return;
        live.status = "error";
        live.error = "the server did not answer (" + error + "). Is `grasppanda explore` still running?";
        render();
      }
    );
  }

  function saveConfig(result, overwrite) {
    var key = live.key;
    saving.status = "saving";
    post("save", { method: result.method.name, sets: result.sets, name: saving.name, overwrite: overwrite }).then(
      function (answer) {
        if (saving.key !== key) return;
        if (answer.error) {
          saving.status = "error";
          saving.message = answer.error;
        } else if (answer.exists) {
          saving.status = "exists";
          saving.message = answer.name + " exists in the working directory.";
        } else {
          saving.status = "written";
          saving.message = answer.written;
        }
        render();
      },
      function (error) {
        saving.status = "error";
        saving.message = String(error);
        render();
      }
    );
  }

  function liveBlock(result) {
    if (!liveBase || !result.sets.length) return null;
    var title = h("div", { class: "gx-why-title", text: "Contract check, run now by this installation" });
    if (!live.status) return null;
    if (live.status === "loading") {
      return h("div", { class: "gx-live k-none" }, [title, h("p", { text: "composing …" })]);
    }
    if (live.status === "error") {
      return h("div", { class: "gx-live k-fails" }, [title, h("p", { text: live.error })]);
    }
    var report = live.report;
    if (!report) return null;
    var children = [title];
    if (report.accepted) {
      children.push(h("p", {}, [
        mark({ kind: "runs", symbol: "✓" }),
        h("b", { text: " accepted" }),
        ": the contract rules and the config check accept this config (hash ",
        h("code", { text: String(report.config_hash || "").slice(0, 12) }),
        "). Whether it trains and predicts is what the probe shows; this page does not run it.",
      ]));
    } else {
      children.push(h("p", {}, [mark({ kind: "refused", symbol: "✗" }), h("b", { text: " refused" }), ": nothing can be written. The conflict and the fix:"]));
      children.push(h("ul", {}, report.problems.map(function (message) {
        return rich(message, "li");
      }).concat(report.issues.map(function (issue) {
        return h("li", {}, [h("code", { text: issue.path }), " ", rich(issue.message)]);
      }))));
      var kinds = (report.refusal_kinds || []).map(function (item) {
        return typeof item === "string" ? item : item && item.kind;
      }).filter(Boolean);
      if (kinds.length) {
        children.push(h("p", { class: "gx-fine", text: "Kind of the refusal: " + kinds.filter(function (kind, index) { return kinds.indexOf(kind) === index; }).join(", ") + "." }));
      }
    }
    var recorded = result.cell ? result.cell.verdict !== "refused" : null;
    if (recorded !== null && recorded !== report.accepted) {
      children.push(h("p", { class: "gx-note", text: "The record and this installation disagree: the record was measured with another version of the code." }));
    }
    if (report.stages.length) {
      children.push(h("table", { class: "gx-stages" }, [
        h("tbody", {}, report.stages.map(function (stage) {
          return h("tr", { class: /^base config/.test(stage.origin) ? "" : "is-set" }, [
            h("th", { scope: "row", text: stage.slot }),
            h("td", {}, [h("code", { text: stage.component })]),
            h("td", { text: stage.origin.replace(/`/g, "") }),
          ]);
        })),
      ]));
    }
    if (report.actions.length) {
      children.push(h("ul", { class: "gx-did" }, report.actions.map(function (action) {
        return rich(action, "li");
      })));
    }
    if (report.accepted) {
      var busy = saving.status === "saving";
      children.push(h("div", { class: "gx-save" }, [
        h("input", {
          type: "text",
          class: "gx-search",
          "data-key": "save-name",
          "aria-label": "File name of the composed config",
          spellcheck: "false",
          value: saving.name,
          oninput: function (event) {
            saving.name = event.target.value;
            saving.status = "";
          },
        }),
        h("button", {
          type: "button",
          class: "gx-copy",
          "data-key": "save",
          disabled: busy ? true : null,
          text: saving.status === "exists" ? "Overwrite" : "Save the config",
          onclick: function () {
            saveConfig(result, saving.status === "exists");
          },
        }),
      ]));
      if (saving.status === "written") {
        children.push(h("p", {}, [mark({ kind: "runs", symbol: "✓" }), " written ", h("code", { text: saving.message }), " into the directory the server was started in. Next: ", h("code", { text: L.probeCommand(data, saving.message, result.cell) })]));
      } else if (saving.status === "exists" || saving.status === "error") {
        children.push(h("p", { class: "gx-note", text: saving.message }));
      }
    }
    return h("div", { class: "gx-live k-" + (report.accepted ? "runs" : "refused") }, children);
  }

  // ---------------------------------------------------------------------- build view

  function methodPicker() {
    return h("div", { class: "gx-methods", role: "group", "aria-label": "Method" }, [
      h("span", { class: "gx-label", text: "Method" }),
    ].concat(
      data.methods.map(function (m) {
        return h("button", {
          type: "button",
          class: "gx-method",
          "data-key": "method-" + m.name,
          "aria-pressed": state.method === m.name ? "true" : "false",
          title: m.source + " · " + m.config,
          text: m.name,
          onclick: function () {
            set({ method: m.name, picks: {}, open: "", laneQuery: "", notice: [] }, true);
          },
        });
      })
    ));
  }

  function chip(slot, choice, current, isShipped) {
    var badge = isShipped ? L.badge(L.lookup(data, state.method, [])) : L.mark(data, state.method, choice.sets);
    var title = (choice.option && choice.option.summary) || (choice.sets || []).join("  ");
    if (choice.reason) title += "\nNo cell of the grid: " + choice.reason;
    return h(
      "button",
      {
        type: "button",
        class:
          "gx-chip k-" + badge.kind + (badge.check || badge.kind === "accepted" ? " is-check" : "") +
          (slot === "loss" ? " is-long" : "") + (badge.empty ? " is-nograsp" : ""),
        "data-key": "chip-" + slot + "-" + (isShipped ? "" : choice.name),
        "aria-pressed": current ? "true" : "false",
        title: title + (isShipped ? "" : "\n" + badge.label + " (this change alone, under " + state.method + ")"),
        onclick: function () {
          pick(slot, isShipped ? "" : choice.name);
        },
      },
      [
        mark(badge),
        h("span", { class: "gx-chip-name", text: choice.label || choice.name }),
        isShipped ? h("span", { class: "gx-tag", text: "shipped" }) : null,
      ]
    );
  }

  function shippedName(slot, ships) {
    var recipe = (L.lane(data, slot) || {}).recipe;
    return ships.option || ships.label || ships.component || (slot === "post" ? "none added" : slot === "loss" || recipe ? "the method's own" : "built in");
  }

  function laneBody(entry, m, choices, ships) {
    var slot = entry.key;
    var current = state.picks[slot] || "";
    var query = state.laneQuery.trim().toLowerCase();
    function hit(words) {
      return !query || words.join(" ").toLowerCase().indexOf(query) >= 0;
    }
    var others = choices.filter(function (choice) {
      return choice.name !== ships.option;
    });
    var matching = others.filter(function (choice) {
      var option = choice.option || {};
      return hit([choice.name, option.component || choice.component, option.summary, (choice.sets || []).join(" ")]);
    });
    var name = shippedName(slot, ships);
    var folded = !query && !state.expanded[slot] && others.length > FEW + 4;
    var shown = folded
      ? matching.filter(function (choice, index) {
          return index < FEW || choice.name === current;
        })
      : matching;
    var chips = [];
    if (hit([name, ships.component, "shipped"])) {
      chips.push(chip(slot, { name: name, option: L.option(data, slot, ships.option) }, !current, true));
    }
    shown.forEach(function (choice) {
      chips.push(chip(slot, choice, current === choice.name, false));
    });
    var more = null;
    if (!query && others.length > FEW + 4) {
      more = h("button", {
        type: "button",
        class: "gx-more",
        "data-key": "more-" + slot,
        "aria-expanded": folded ? "false" : "true",
        text: folded ? "Show all " + others.length + " options" : "Show fewer",
        onclick: function () {
          var expanded = {};
          Object.keys(state.expanded).forEach(function (key) {
            expanded[key] = state.expanded[key];
          });
          expanded[slot] = folded;
          set({ expanded: expanded });
        },
      });
    }
    var empty = !chips.length
      ? h("p", { class: "gx-empty" }, [
          "No option of this slot matches \u201c" + state.laneQuery.trim() + "\u201d. ",
          h("button", {
            type: "button",
            class: "gx-link",
            "data-key": "lane-clear",
            text: "Clear the filter",
            onclick: function () {
              state.focusKey = "lane-search";
              set({ laneQuery: "" });
            },
          }),
        ])
      : null;
    return h("div", { class: "gx-lane-body", id: "gx-lane-" + slot }, [
      h("p", { class: "gx-fine" }, [rich(entry.task + " "), h("code", { text: entry.config })]),
      docLinks(entry.pages),
      others.length > FEW
        ? h("input", {
            type: "search",
            class: "gx-search",
            "data-key": "lane-search",
            placeholder: "Filter " + others.length + " options",
            "aria-label": "Filter the options of " + entry.title,
            value: state.laneQuery,
            oninput: function (event) {
              set({ laneQuery: event.target.value });
            },
          })
        : null,
      h("div", { class: "gx-chips" + (folded ? " is-folded" : "") }, chips.concat([more])),
      empty,
      h("p", {
        class: "gx-fine",
        text:
          slot === "loss"
            ? "A loss sits on one term of a stage, so a choice names its terms. Each is a set that was probed as a whole under " + m.name + "; the mark is the verdict of that set."
            : "The mark of an option is what happened with this one slot replaced under " + m.name + ". \u25c7 means compose accepts it and nobody probed it; a dashed circle means this page has no answer.",
      }),
    ]);
  }

  function laneRow(entry, m) {
    var slot = entry.key;
    var choices = L.choices(data, m.name, slot);
    var ships = L.shipped(data, m.name, slot);
    var current = state.picks[slot] || "";
    var open = state.open === slot;
    var chosen = null;
    choices.forEach(function (choice) {
      if (current && choice.name === current) chosen = choice;
    });
    var name = chosen ? chosen.label || chosen.name : shippedName(slot, ships);
    var badge = chosen ? L.mark(data, m.name, chosen.sets) : null;
    var head = h(
      "button",
      {
        type: "button",
        class: "gx-lane-head",
        "data-key": "lane-" + slot,
        "aria-expanded": open ? "true" : "false",
        "aria-controls": "gx-lane-" + slot,
        onclick: function () {
          set({ open: open ? "" : slot, laneQuery: "" });
        },
      },
      [
        h("span", { class: "gx-lane-title", text: entry.title }),
        h("span", { class: "gx-lane-choice" }, [
          badge ? mark(badge) : null,
          h("span", { class: "gx-lane-name" + (state.flash === slot ? " is-flash" : ""), title: name, text: name }),
        ]),
        h("span", { class: "gx-tag" + (current ? " is-changed" : ""), text: current ? "changed" : "shipped" }),
        h("span", { class: "gx-caret", "aria-hidden": "true", text: open ? "\u2013" : "+" }),
      ]
    );
    return h("li", { class: "gx-lane" + (badge ? " is-changed k-" + badge.kind : "") + (open ? " is-open" : "") }, [
      head,
      open ? laneBody(entry, m, choices, ships) : null,
    ]);
  }

  /* The lanes of a method. A slot the method does not have is not drawn; a method with
   * few slots says why. The slots of the training recipe are around the model, not a
   * stage between observation and grasps: they are drawn under a heading of their own. */
  function lanes(m) {
    var all = data.lanes.filter(function (entry) {
      return m.slots.indexOf(entry.key) >= 0;
    });
    var own = all.filter(function (entry) {
      return !entry.recipe;
    });
    var around = all.filter(function (entry) {
      return entry.recipe;
    });
    var note = m.seed_based
      ? null
      : h("p", { class: "gx-fine gx-family" }, [
          m.name + " is a detector family of its own (",
          h("code", { text: m.detector }),
          "), not built from the seed-based stages: " +
            own.map(function (entry) { return entry.title.toLowerCase(); }).join(" and ") +
            " are its slots.",
        ]);
    return [
      note,
      h("div", { class: "gx-cap", text: "Observation" }),
      h("ol", { class: "gx-lanes" }, own.map(function (entry) {
        return laneRow(entry, m);
      })),
      h("div", { class: "gx-cap", text: "Grasps" }),
      around.length ? h("div", { class: "gx-cap gx-cap-recipe", text: "Training recipe · around the model" }) : null,
      around.length
        ? h("ol", { class: "gx-lanes" }, around.map(function (entry) {
            return laneRow(entry, m);
          }))
        : null,
    ];
  }

  function examples() {
    var wanted = [
      ["adapter", "an adapter is inserted"],
      ["refused", "a refused swap"],
      ["unstable", "an unstable run"],
      ["fails", "a failure"],
    ];
    var found = [];
    wanted.forEach(function (item) {
      for (var i = 0; i < data.cells.length; i++) {
        var cell = data.cells[i];
        if (L.badge(cell).kind === item[0] && cell.sets.length === 1 && cell.slot) {
          found.push([item[1], cell]);
          return;
        }
      }
    });
    for (var i = 0; i < data.cells.length; i++) {
      var cell = data.cells[i];
      if (cell.group === "stage-set" && cell.key.split("&").length > 2 && !cell.shares) {
        found.push(["several stages at once", cell]);
        break;
      }
    }
    return h("div", { class: "gx-examples" }, [
      h("span", { class: "gx-label", text: "Try" }),
    ].concat(
      found.map(function (item, index) {
        return h("button", {
          type: "button",
          class: "gx-link",
          "data-key": "example-" + index,
          text: item[0],
          onclick: function () {
            loadCell(item[1]);
          },
        });
      })
    ));
  }

  /* Put a record into the builder: its method and the options its sets name. */
  function loadCell(cell) {
    var picks = {};
    if (cell.group === "loss") {
      picks.loss = L.lossName(cell);
    } else {
      cell.sets.forEach(function (item) {
        var at = item.indexOf("=");
        picks[item.slice(0, at)] = item.slice(at + 1);
      });
    }
    openBuilder(cell.method, picks);
  }

  /* Show a selection in the builder, with its answer in view and in focus. */
  function openBuilder(methodName, picks) {
    state.method = methodName;
    set({ view: "build", open: "", cellMenu: "", notice: [], picks: L.normalise(data, methodName, picks) }, true);
    var target = window.innerWidth < 1000 ? document.getElementById("gx-result") : root;
    if (target && target.scrollIntoView) target.scrollIntoView({ block: "start" });
    var title = document.getElementById("gx-title");
    if (title) title.focus({ preventScroll: true });
  }

  function same(a, b) {
    return a.length === b.length && a.every(function (item, index) {
      return item === b[index];
    });
  }

  function messageList(messages) {
    return h("ul", {}, messages.map(function (message) {
      return rich(message, "li");
    }));
  }

  function reasons(cell, methodName) {
    var out = [];
    if (!cell) return out;
    /* What compose answers now, where the page holds it: a record keeps the message of
     * the day it was measured, and the lists of alternatives in a fix grow with the code. */
    var now = cell.sets.length ? L.contract(data, methodName, cell.sets) : null;
    if (cell.refusal) {
      if (now && !now.ok) {
        out.push(h("div", { class: "gx-why" }, [
          h("div", { class: "gx-why-title", text: "The conflict and the fix, as compose prints them" }),
          messageList(now.messages),
          same(now.messages, cell.refusal)
            ? null
            : h("p", { class: "gx-fine", text: "This is the message of the code this page was generated from. The record was refused for the same combination by an earlier version, whose message was worded differently." }),
        ]));
      } else {
        out.push(h("div", { class: "gx-why" }, [
          h("div", { class: "gx-why-title", text: "The conflict and the fix, as compose printed them when this was measured" }),
          messageList(cell.refusal),
          now && now.ok
            ? h("p", { class: "gx-note", text: "compose accepts this combination in the code this page was generated from: the record is older than that and has not been probed again." })
            : null,
        ]));
      }
    } else if (now && !now.ok) {
      out.push(h("div", { class: "gx-why k-refused" }, [
        h("div", { class: "gx-why-title", text: "The record and the code this page was generated from disagree" }),
        h("p", { text: "The record below was measured with an earlier version. compose refuses the combination now:" }),
        messageList(now.messages),
      ]));
    }
    if (cell.failure) {
      var f = cell.failure;
      out.push(h("div", { class: "gx-why" }, [
        h("div", {
          class: "gx-why-title",
          text: "Where it failed: " + [f.phase, f.when, f.component].filter(Boolean).join(" · "),
        }),
        h("p", {}, [h("code", { text: f.error || "" }), " ", rich(f.message || "")]),
      ]));
    }
    if (cell.unstable) {
      out.push(h("div", { class: "gx-why" }, [
        h("div", { class: "gx-why-title", text: "What was unstable" }),
        rich(cell.unstable, "p"),
      ]));
    }
    if (cell.adapter && cell.verdict !== "refused") {
      out.push(h("p", { class: "gx-note" }, [
        "The bare swap is refused by the contract between encoder and proposal. compose wrapped the encoder in the adapter ",
        h("code", { text: cell.adapter }),
        ", said so, and the adapted config is what was probed.",
      ]));
    }
    if (cell.fitted && cell.fitted.length) {
      out.push(h("p", { class: "gx-note" }, ["compose also set "].concat(
        cell.fitted.map(function (item, index) {
          return h("span", {}, [index ? ", " : "", h("code", { text: item })]);
        }),
        [" to fit the method."]
      )));
    }
    return out;
  }

  function factsList(cell) {
    var facts = L.facts(cell);
    if (!facts.length) return null;
    var rows = [];
    facts.forEach(function (fact) {
      rows.push(h("dt", { text: fact[0] }));
      rows.push(h("dd", { text: fact[1] }));
    });
    if (cell.levels) {
      rows.push(h("dt", { text: "levels reached" }));
      rows.push(h("dd", { text: L.levelsText(data, cell) }));
    }
    if (cell.why) {
      rows.push(h("dt", { text: "what kept it from a level" }));
      rows.push(h("dd", { text: cell.why.join("; ") }));
    }
    return h("div", { class: "gx-facts" }, [
      h("dl", {}, rows),
      h("p", {
        class: "gx-fine",
        text:
          "Seen by the probe (" + data.probe.iterations + " iterations on " + data.probe.frames +
          " frames). A falling loss shows that training works, not how good the result is.",
      }),
    ]);
  }

  function optionCard(part) {
    var option = part.option;
    var slotLane = L.lane(data, part.slot);
    if (!option) {
      return h("div", { class: "gx-card" }, [
        h("div", { class: "gx-card-title" }, [h("span", { class: "gx-label", text: slotLane.title }), h("code", { text: part.label || part.name })]),
        h("ul", { class: "gx-plain" }, part.sets.map(function (item) {
          var name = item.slice(item.indexOf("=") + 1);
          var loss = L.option(data, part.slot, name);
          return h("li", {}, [h("code", { text: item }), loss ? " " + loss.summary : ""]);
        })),
        slotLane.pages.length ? h("p", {}, [pageLink(slotLane.pages[0].stem, "How to use it: " + slotLane.pages[0].title.replace(/`/g, ""))]) : null,
      ]);
    }
    var meta = [];
    meta.push(h("dt", { text: "component" }));
    meta.push(h("dd", {}, [h("code", { text: option.component }), option.preset ? " through the preset " : " with its constructor defaults", option.preset ? h("code", { text: option.name }) : null]));
    if (option.source) {
      meta.push(h("dt", { text: "source" }));
      meta.push(h("dd", { text: option.source }));
    }
    meta.push(h("dt", { text: "needs" }));
    meta.push(h("dd", {}, option.extras.length ? [h("code", { text: 'pip install "grasppanda[' + option.extras.join(",") + ']"' })] : ["nothing beyond the core install"]));
    if (!slotLane.recipe) meta.push(h("dt", { text: "weights" }));
    if (!slotLane.recipe) meta.push(h("dd", {}, option.weights ? ["the preset loads released weights (", h("code", { text: option.weights }), ")"] : [option.preset ? "random initialisation in this preset" : "random initialisation (constructor defaults)"]));
    var classes = L.classText(data, option);
    if (classes) {
      meta.push(h("dt", { text: "measured class" }));
      meta.push(h("dd", {}, [classes, h("span", { class: "gx-fine gx-caveat-inline", text: " · " + data.caveat })]));
    }
    var page = option.page || (slotLane.pages[0] || {}).stem;
    return h("div", { class: "gx-card" }, [
      h("div", { class: "gx-card-title" }, [h("span", { class: "gx-label", text: slotLane.title }), h("code", { text: option.name })]),
      rich(option.summary, "p"),
      h("dl", {}, meta),
      page ? h("p", {}, [pageLink(page, "How to use it: the slot page")]) : null,
    ]);
  }

  function yamlBlock(result) {
    if (liveBase && live.status === "done" && live.report && live.report.accepted && result.sets.length) {
      return fold("yaml", "The composed config, as this installation writes it", [codeBlock("Composed config", live.report.config, "yaml", true)]);
    }
    var parts = result.parts.filter(function (part) {
      return part.option || part.sets.length;
    });
    if (!parts.length) return null;
    var active = parts.filter(function (part) {
      return part.slot === state.yamlTab;
    })[0] || parts[0];
    var text;
    if (active.option) {
      text = active.option.yaml;
    } else {
      text = active.sets
        .map(function (item) {
          var at = item.indexOf("=");
          var loss = L.option(data, active.slot, item.slice(at + 1));
          return "# " + item.slice(0, at) + "  <-  preset `" + item.slice(at + 1) + "`\n" + (loss ? loss.yaml : "");
        })
        .join("\n");
    }
    var bar = parts.length > 1
      ? h("div", { class: "gx-subtabs", role: "group", "aria-label": "Fragment of" }, parts.map(function (part) {
          return h("button", {
            type: "button",
            class: "gx-subtab",
            "data-key": "yaml-" + part.slot,
            "aria-pressed": part === active ? "true" : "false",
            text: part.slot,
            onclick: function () {
              set({ yamlTab: part.slot });
            },
          });
        }))
      : null;
    return fold("yaml", "Preset YAML: what the option puts into the config", [
      bar,
      codeBlock("Preset fragment", text, "yaml", true),
      h("p", { class: "gx-fine", text: "compose writes this into a child config of the method, fits it and checks it. The file it writes is the config to train." }),
    ]);
  }

  /* The badge at the top of the answer. A record decides it; without one, the contract
   * check does: the one this installation runs (live mode) or the one stored in the page. */
  function shownBadge(result) {
    if (result.cell || !result.sets.length || !liveBase) return result.badge;
    if (live.status === "done" && live.report) {
      return L.contractBadge({ ok: live.report.accepted, adapter: live.report.adapter || "" });
    }
    if (live.status === "error") return { kind: "fails", symbol: "!", label: "no answer from the server" };
    return { kind: "none", symbol: "…", label: "composing …" };
  }

  function notice() {
    if (!state.notice.length) return null;
    var children = ["This link named "];
    state.notice.forEach(function (text, index) {
      if (index) children.push(index === state.notice.length - 1 ? " and " : ", ");
      children.push(rich(text));
    });
    children.push(
      ", which this version of the page does not know. What you see is " +
        state.method +
        (Object.keys(state.picks).length ? " with the rest of the link." : " as shipped.") +
        " "
    );
    children.push(h("button", {
      type: "button",
      class: "gx-link",
      "data-key": "notice-close",
      text: "Dismiss",
      onclick: function () {
        set({ notice: [] });
      },
    }));
    return h("p", { class: "gx-notice", role: "status" }, children);
  }

  function liveHint() {
    var command = "grasppanda explore";
    var button = h("button", {
      type: "button",
      class: "gx-copy",
      "data-key": "copy-explore",
      "data-label": "Copy",
      "aria-label": "Copy " + command,
      text: "Copy",
      onclick: function () {
        copyText(command, button);
      },
    });
    return h("div", { class: "gx-hint" }, [
      h("div", {}, [
        h("b", { text: "Get the answer in seconds: " }),
        h("code", { text: command }),
        " opens this page on your machine, where every selection is checked by the real compose and an accepted one can be saved as a config.",
      ]),
      button,
    ]);
  }

  function resultPanel(result, badge) {
    var cell = result.cell;
    var title = result.method.name + (result.sets.length ? " with " + result.sets.join(", ") : " as shipped");
    var liveOn = Boolean(liveBase && result.sets.length);
    var checked = result.contract;
    var caption = cell
      ? "record of " + String(cell.run || "").split("@")[0]
      : liveOn
        ? "checked now by this installation · no probe on record"
        : checked
          ? "answer of compose · no probe on record"
          : "no record";
    var children = [
      notice(),
      h("div", { class: "gx-verdict" }, [
        h("span", { class: "gx-badge k-" + badge.kind + (cell ? "" : " is-check") }, [mark(badge, "on-badge"), badge.label]),
        h("span", { class: "gx-fine", text: caption }),
      ]),
      h("p", { class: "gx-title", id: "gx-title", tabindex: "-1", text: title }),
      liveBase
        ? h("p", { class: "gx-run-link" }, [
            h("button", {
              type: "button",
              class: "gx-link",
              "data-key": "go-run",
              text: result.sets.length ? "Run this selection on a scene \u2192" : "Run " + result.method.name + " on a scene \u2192",
              onclick: function () {
                runForm.target = result.sets.length ? "selection" : "method";
                state.focusKey = "tab-run";
                set({ view: "run" }, true);
              },
            }),
          ])
        : null,
    ];
    var coverage =
      L.number(data.counts.grid - data.counts.grid_without_record) + " of the " + L.number(data.counts.grid) +
      " combinations of the probe grid have a record (" + L.number(data.counts.stage_sets.recorded) + " of its " +
      L.number(data.counts.stage_sets.listed) + " sets of several stages).";
    var singles = result.parts.length > 1 || (result.parts.length === 1 && !checked && !liveOn)
      ? h("div", { class: "gx-why k-none" }, [
          h("div", { class: "gx-why-title", text: "Each change alone, under " + result.method.name }),
          h("ul", { class: "gx-plain" }, result.parts.map(function (part) {
            return h("li", {}, [mark(part.badge), h("code", { text: part.sets.join(" ") }), " " + part.badge.label]);
          })),
          result.pairs.length
            ? h("div", { class: "gx-why-title", text: "Two of the changes together" })
            : null,
          result.pairs.length
            ? h("ul", { class: "gx-plain" }, result.pairs.map(function (pair) {
                return h("li", {}, [mark(pair.badge), h("code", { text: pair.sets.join(" ") }), " " + pair.badge.label]);
              }))
            : null,
          h("p", { class: "gx-fine", text: "None of this decides the whole selection: a stage that is refused alone may work together with the stage it depends on (a head bundle with its assigner and its labels), and changes that fit in pairs may not fit all at once." }),
        ])
      : null;
    /* An option the grid has no cell for says why (an adapter, a recipe that names a
     * module the method does not have, a post-processor no dataset can feed). */
    if (result.state === "unmeasured") {
      result.parts.forEach(function (part) {
        if (part.reason) children.push(h("p", { class: "gx-note" }, [h("code", { text: part.sets.join(" ") || part.name }), " has no cell of the grid: ", rich(part.reason)]));
      });
    }
    if (result.state === "unmeasured" && liveOn) {
      children.push(liveBlock(result));
      children.push(h("p", { class: "gx-fine", text: "No record says whether this combination trains: " + coverage }));
      children.push(singles);
    } else if (result.state === "unmeasured" && checked) {
      if (checked.ok) {
        children.push(h("p", {}, [
          h("code", { text: "grasppanda compose" }),
          " accepts this combination: the contract rules and the config check pass",
          checked.adapter ? ", with the scene encoder wrapped in the adapter " : "",
          checked.adapter ? h("code", { text: checked.adapter }) : null,
          ". Nobody has probed it, so whether it trains is not known. The second command below finds out: it trains " +
            data.probe.iterations + " iterations on " + data.probe.frames + " frames.",
        ]));
      } else {
        children.push(h("p", { text: "Nothing can be built: compose refuses this combination and writes no file." }));
        children.push(h("div", { class: "gx-why" }, [
          h("div", { class: "gx-why-title", text: "The conflict and the fix, as compose prints them" }),
          messageList(checked.messages),
        ]));
      }
      children.push(h("p", { class: "gx-fine", text: "This is the answer of compose, stored when the page was generated; it is not a probe. " + coverage }));
      children.push(singles);
    } else if (result.state === "unmeasured") {
      children.push(h("p", {}, [
        "Nobody has probed this combination, so this page does not guess. " + coverage +
          (data.contracts ? " The answers of compose stored here cover one change and two changes of two slots; a stage together with an option of the training recipe is not among them." : "") +
          " The first command below answers at once whether the contracts accept it; the second trains it for " +
          data.probe.iterations + " iterations on " + data.probe.frames + " frames.",
      ]));
      children.push(liveHint());
      children.push(singles);
    } else {
      children.push(rich(data.verdicts[cell.verdict] || "", "p"));
      reasons(cell, result.method.name).forEach(function (node) {
        children.push(node);
      });
      if (cell.shares) {
        children.push(h("p", { class: "gx-note" }, ["Composes to the same config as ", h("code", { text: cell.shares }), "; the two share one probe."]));
      }
      children.push(factsList(cell));
      children.push(liveBlock(result));
    }
    var refused = !result.probe || (liveOn && live.status === "done" && live.report && !live.report.accepted);
    if (result.compose) {
      children.push(codeBlock("1 · compose: a checked config, or the conflict and the fix", result.compose, "compose"));
    }
    if (refused) {
      children.push(h("p", { class: "gx-fine", text: "Nothing to probe: compose writes no file when it refuses." }));
    } else {
      children.push(codeBlock((result.compose ? "2 · " : "") + "probe: build, train a few iterations on real frames, predict", result.probe, "probe"));
    }
    if (cell && cell.probe_opts) {
      children.push(h("p", { class: "gx-fine", text: data.scale_statistic.replace(/`/g, "") }));
    }
    children.push(yamlBlock(result));
    if (result.parts.length) {
      children.push(fold(
        "cards",
        result.parts.length === 1 ? "About this option: source, extra, weights" : "About the " + result.parts.length + " changed options: source, extra, weights",
        result.parts.map(optionCard)
      ));
    }
    if (cell) {
      children.push(h("p", { class: "gx-fine" }, ["Record ", h("code", { text: cell.id }), " of ", h("a", { href: docHref("advanced_guides/measured_combinations"), text: "Measured combinations" }), cell.run ? ", run " + cell.run + "." : "."]));
    }
    return h("section", { class: "gx-result k-" + badge.kind, id: "gx-result", "aria-live": "polite", "aria-label": "Answer" }, children);
  }

  function buildView() {
    var m = L.method(data, state.method);
    var result = L.answer(data, state.method, state.picks);
    ensureLive(result);
    var badge = shownBadge(result);
    var changed = Object.keys(result.picks).length;
    var share = h("button", {
      type: "button",
      class: "gx-link",
      "data-key": "share",
      "data-label": "Copy a link to this selection",
      text: "Copy a link to this selection",
      onclick: function () {
        copyText(String(window.location.href), share);
      },
    });
    return h("div", { class: "gx-build" }, [
      h("div", { class: "gx-pipeline" }, [methodPicker(), examples()].concat(lanes(m), [
        h("div", { class: "gx-actions" }, [
          h("button", {
            type: "button",
            class: "gx-link",
            "data-key": "reset",
            disabled: changed ? null : true,
            text: "Back to " + m.name + " as shipped",
            onclick: function () {
              set({ picks: {}, open: "", notice: [] }, true);
            },
          }),
          share,
        ]),
      ])),
      h("div", { class: "gx-side" }, [resultPanel(result, badge)]),
      h("a", { class: "gx-bar k-" + badge.kind, href: "#gx-result", onclick: function (event) {
        event.preventDefault();
        var target = document.getElementById("gx-result");
        if (target) target.scrollIntoView({ block: "start" });
      } }, [
        mark(badge, "on-badge"),
        h("span", { text: badge.label }),
        h("span", { class: "gx-bar-more", text: "details ↓" }),
      ]),
    ]);
  }

  // ----------------------------------------------------------------------- grid view

  function hideTip() {
    if (tip) tip.hidden = true;
  }

  function showTip(target, lines) {
    if (!tip) {
      tip = h("div", { class: "gx-tip", role: "tooltip", id: "gx-tip" });
      root.appendChild(tip);
    }
    tip.textContent = "";
    lines.forEach(function (line, index) {
      tip.appendChild(h("div", { class: index ? "" : "gx-tip-title", text: line }));
    });
    tip.hidden = false;
    var box = target.getBoundingClientRect();
    var host = root.getBoundingClientRect();
    var left = Math.min(Math.max(box.left - host.left + box.width / 2 - tip.offsetWidth / 2, 0), Math.max(host.width - tip.offsetWidth, 0));
    var top = box.top - host.top - tip.offsetHeight - 8;
    if (top + host.top < 60) top = box.bottom - host.top + 8;
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }

  function tipLines(m, slot, entry, found, badge) {
    var lines = [m.name + " · " + entry.name];
    if (!found.length) {
      lines.push(badge.kind === "none" ? "not measured" : badge.label);
      var reason = slot === "loss" ? "" : L.notProbed(m, slot, entry.name);
      if (reason) lines.push("No cell of the grid: " + reason);
      if (badge.kind !== "none") lines.push("The answer of compose; no probe on record.");
      lines.push("Click to open it in the builder.");
      return lines;
    }
    found.forEach(function (cell) {
      lines.push(L.badge(cell).label + (slot === "loss" ? " as " + cell.sets.join(" ") : ""));
      L.facts(cell).forEach(function (fact) {
        lines.push(fact[0] + ": " + fact[1]);
      });
      if (cell.levels) lines.push(L.levelsText(data, cell));
      if (cell.why) lines.push(cell.why.join("; "));
      if (cell.refusal) lines.push(cell.refusal[0].split(". ")[0] + "." + (cell.kind ? " (" + cell.kind + ")" : ""));
      if (cell.failure) lines.push((cell.failure.error || "") + " in the " + (cell.failure.component || cell.failure.phase));
      if (cell.unstable) lines.push(cell.unstable);
    });
    lines.push(found.length > 1 ? "Click to choose one of the " + found.length + " records." : "Click to open it in the builder.");
    return lines;
  }

  function gridCell(m, slot, entry, found, place, focusable) {
    var loss = slot === "loss";
    var badge = found.length ? L.badgeOf(found) : loss ? L.badge(null) : L.mark(data, m.name, [slot + "=" + entry.name]);
    var inSet = loss && found.some(function (cell) {
      return cell.sets.length > 1;
    });
    var label = m.name + ", " + entry.name + ": " + (loss && found.length
      ? found.map(function (cell) { return L.badge(cell).label + " as " + cell.sets.join(" "); }).join("; ")
      : badge.label);
    var lines = tipLines(m, slot, entry, found, badge);
    var key = "cell-" + slot + "-" + entry.name + "-" + m.name;
    var grasps = found.length === 1 && found[0].grasps && found[0].grasps.length ? found[0].grasps.join("·") : "";
    return h("td", {}, [
      h(
        "button",
        {
          type: "button",
          class: "gx-cell k-" + badge.kind + (found.length ? "" : " is-check") + (badge.empty ? " is-nograsp" : "") + (grasps ? " has-grasps" : ""),
          "data-key": key,
          "data-row": String(place[0]),
          "data-col": String(place[1]),
          tabindex: focusable ? "0" : "-1",
          "aria-label": label,
          onmouseenter: function (event) {
            showTip(event.currentTarget, lines);
          },
          onmouseleave: hideTip,
          onfocus: function (event) {
            state.gridFocus = key;
            showTip(event.currentTarget, lines);
          },
          onblur: hideTip,
          onclick: function () {
            hideTip();
            if (found.length > 1) {
              state.focusKey = "menu-0";
              set({ cellMenu: key });
            } else if (found[0]) {
              loadCell(found[0]);
            } else {
              var picks = {};
              picks[slot] = entry.name;
              openBuilder(m.name, picks);
            }
          },
        },
        [
          h("span", { "aria-hidden": "true", text: badge.symbol + (badge.kind === "adapter" ? "ᴬ" : "") + (inSet ? "⊂" : "") }),
          grasps ? h("span", { class: "gx-cell-grasps", "aria-hidden": "true", text: grasps }) : null,
        ]
      ),
    ]);
  }

  /* The records of a loss cell, each a button: a loss was probed in more than one set. */
  function cellMenu(m, entry, found, key) {
    return h("div", { class: "gx-menu", role: "group", "aria-label": "Records of " + entry.name + " under " + m.name }, [
      h("div", { class: "gx-why-title", text: entry.name + " under " + m.name + " is part of " + found.length + " records. Open one:" }),
      h("ul", { class: "gx-plain" }, found.map(function (cell, index) {
        var badge = L.badge(cell);
        return h("li", {}, [
          mark(badge),
          h("button", { type: "button", class: "gx-link", "data-key": "menu-" + index, text: badge.label + ": " + cell.sets.join(" "), onclick: function () {
            loadCell(cell);
          } }),
        ]);
      })),
      h("button", { type: "button", class: "gx-link", "data-key": "menu-close", text: "Close", onclick: function () {
        state.focusKey = key;
        set({ cellMenu: "" });
      } }),
    ]);
  }

  function moveInGrid(event) {
    var step = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[event.key];
    var from = event.target && event.target.closest ? event.target.closest(".gx-cell") : null;
    if (!step || !from) return;
    var row = Number(from.getAttribute("data-row")) + step[0];
    var col = Number(from.getAttribute("data-col")) + step[1];
    var to = event.currentTarget.querySelector('.gx-cell[data-row="' + row + '"][data-col="' + col + '"]');
    if (!to) return;
    event.preventDefault();
    from.setAttribute("tabindex", "-1");
    to.setAttribute("tabindex", "0");
    to.focus();
  }

  function toggle(key, label, pressed, onclick) {
    return h("button", { type: "button", class: "gx-toggle", "data-key": key, "aria-pressed": pressed ? "true" : "false", text: label, onclick: onclick });
  }

  /* The slot the grid opens on: the one with the most options, so the first look shows
   * every kind of verdict. */
  function firstGridSlot() {
    var best = data.lanes[0].key;
    data.lanes.forEach(function (entry) {
      if ((data.options[entry.key] || []).length > (data.options[best] || []).length) best = entry.key;
    });
    return best;
  }

  function gridView() {
    var slot = state.slot || firstGridSlot();
    var entry = L.lane(data, slot);
    var loss = slot === "loss";
    var grid = L.gridRows(data, slot);
    var filter = state.filter;
    var rows = grid.rows.filter(function (row) {
      return L.matches(row.option, filter, data);
    });
    var level = filter.level || "runs";
    function patch(change) {
      var next = { q: filter.q, level: level, classes: filter.classes.slice(), extra: filter.extra, weights: filter.weights };
      Object.keys(change).forEach(function (name) {
        next[name] = change[name];
      });
      set({ filter: next, cellMenu: "" });
    }
    var slotTabs = h("div", { class: "gx-slots", role: "group", "aria-label": "Slot" }, data.lanes.map(function (item) {
      return h("button", {
        type: "button",
        class: "gx-method",
        "data-key": "slot-" + item.key,
        "aria-pressed": item.key === slot ? "true" : "false",
        onclick: function () {
          state.filter = { q: "", level: level, classes: [], extra: false, weights: false };
          set({ slot: item.key, cellMenu: "", gridFocus: "" }, true);
        },
      }, [item.title, h("span", { class: "gx-count", text: String((data.options[item.key] || []).length) })]);
    }));
    var classes = L.classesOf(data, slot, level);
    var levelPicker = classes.length || (data.options[slot] || []).some(function (option) { return option.classes; })
      ? h("div", { class: "gx-filters gx-levels", role: "group", "aria-label": "Class at the level" }, [
          h("span", { class: "gx-label", text: "Class at the level" }),
        ].concat((data.levels || []).map(function (item) {
          return toggle("level-" + item.key, item.key, item.key === level, function () {
            patch({ level: item.key, classes: [] });
          });
        })))
      : null;
    var filters = h("div", { class: "gx-filters" }, [
      h("input", {
        type: "search",
        class: "gx-search",
        "data-key": "grid-search",
        placeholder: "Search name, component, source",
        "aria-label": "Search the options of " + entry.title,
        value: filter.q,
        oninput: function (event) {
          patch({ q: event.target.value });
        },
      }),
    ].concat(
      classes.map(function (name) {
        var on = filter.classes.indexOf(name) >= 0;
        var label = name === "relative" ? "N of M probed" : name;
        return toggle("class-" + name, label, on, function () {
          patch({ classes: on ? filter.classes.filter(function (other) { return other !== name; }) : filter.classes.concat([name]) });
        });
      }),
      [
        toggle("f-extra", "needs an extra", filter.extra, function () {
          patch({ extra: !filter.extra });
        }),
        toggle("f-weights", "loads released weights", filter.weights, function () {
          patch({ weights: !filter.weights });
        }),
      ]
    ));
    var kinds = [
      [L.badge({ verdict: "runs" }), "runs", ""],
      [L.badge({ verdict: "runs", adapter: "x" }), "runs in the adapter compose inserted", "ᴬ"],
      [L.badge({ verdict: "runs-empty" }), "runs, no grasp predicted", ""],
      [L.badge({ verdict: "runs-unstable" }), "runs, unstable", ""],
      [L.badge({ verdict: "refused" }), "refused by compose", ""],
      [L.badge({ verdict: "fails", failure: {} }), "fails after the check passed", ""],
    ];
    if (loss) {
      kinds.push([{ kind: "runs", symbol: "✓" }, "probed inside a set of several terms", "⊂"]);
      kinds.push([{ kind: "mixed", symbol: "±" }, "its records disagree", ""]);
    } else if (data.contracts) {
      kinds.push([L.contractBadge({ ok: true }), "no record: compose accepts it, not probed", "", " is-check"]);
      kinds.push([L.contractBadge({ ok: false }), "no record: compose refuses it", "", " is-check"]);
    }
    kinds.push([L.badge(null), loss ? "not part of a record" : "no answer on this page", "", " is-check"]);
    kinds.push([{ kind: "runs", symbol: "✓" }, "greyed: no grasp predicted on any probed frame (the grasps per frame are under the mark)", "", " is-nograsp"]);
    var legend = h("ul", { class: "gx-legend" }, kinds.map(function (kind) {
      return h("li", {}, [h("span", { class: "gx-cell is-legend k-" + kind[0].kind + (kind[3] || ""), "aria-hidden": "true", text: kind[0].symbol + kind[2] }), kind[1]]);
    }));
    var withClass = (data.options[slot] || []).some(function (option) { return option.classes; });
    var firstKey = rows.length && grid.columns.length ? "cell-" + slot + "-" + rows[0].option.name + "-" + grid.columns[0].name : "";
    var focusKey = firstKey;
    rows.forEach(function (row) {
      grid.columns.forEach(function (m) {
        if ("cell-" + slot + "-" + row.option.name + "-" + m.name === state.gridFocus) focusKey = state.gridFocus;
      });
    });
    var menu = null;
    var body = rows.map(function (row, r) {
      var option = row.option;
      return h("tr", {}, [
        h("th", { scope: "row", class: "gx-grid-option", title: option.summary }, [
          breakable(option.name),
          h("span", { class: "gx-grid-sub" }, [
            option.component,
            option.extras.length ? h("span", { class: "gx-tag", title: "optional dependency", text: option.extras.join(", ") }) : null,
            option.weights ? h("span", { class: "gx-tag is-changed", title: option.weights, text: "weights" }) : null,
          ]),
        ]),
      ].concat(
        grid.columns.map(function (m, c) {
          var key = "cell-" + slot + "-" + option.name + "-" + m.name;
          if (key === state.cellMenu && row.cells[m.name].length > 1) menu = cellMenu(m, option, row.cells[m.name], key);
          return gridCell(m, slot, option, row.cells[m.name], [r, c], key === focusKey);
        }),
        withClass ? [h("td", { class: "gx-grid-class", text: L.classText(data, option) })] : []
      ));
    });
    if (!rows.length) {
      body = [h("tr", {}, [h("td", { class: "gx-empty", colspan: String(grid.columns.length + 1 + (withClass ? 1 : 0)) }, [
        "No option of this slot matches" + (filter.q.trim() ? " “" + filter.q.trim() + "”" : " these filters") + ". ",
        h("button", { type: "button", class: "gx-link", "data-key": "grid-clear", text: "Clear the search and the filters", onclick: function () {
          state.focusKey = "grid-search";
          set({ filter: { q: "", level: level, classes: [], extra: false, weights: false } });
        } }),
      ])])];
    }
    var table = h("table", { class: "gx-grid", onkeydown: moveInGrid }, [
      h("thead", {}, [
        h("tr", {}, [h("th", { scope: "col", class: "gx-grid-option", text: "--set " + (loss ? "<stage>.<term>" : slot) + "=" })].concat(
          grid.columns.map(function (m) {
            return h("th", { scope: "col", title: m.config }, [h("span", { text: m.name })]);
          }),
          withClass ? [h("th", { scope: "col", class: "gx-grid-class", text: "class, by level" })] : []
        )),
      ]),
      h("tbody", {}, body),
    ]);
    var caption = loss
      ? "A loss sits on one term of a stage (head.width_loss), and most were probed several terms at a time. A cell shows the verdict of the sets this loss was part of under the method of its column, not of the loss alone: the same loss can run on one term and be refused on another. Hover or focus a cell to see the sets."
      : "A cell is this one slot replaced under the method of its column, with the grasps it predicted per frame under the mark." + (withClass ? " The class of an option counts, level by level (composes, runs, learns, predicts), the methods under which it reached the level; the filters use the level picked above. Class: " + data.caveat + "." : "") + " Arrow keys move between cells.";
    /* The scene-encoder rows include the wrappers compose puts around an encoder; the
     * option counts of the README and of the slot index leave those out. */
    var wrappers = grid.rows.filter(function (row) {
      return /^(feature-propagation|furthest-point-subsample|image-painted(-subsampled)?)$/.test(row.option.name);
    }).length;
    var adapterNote = slot === "backbone" && wrappers
      ? (grid.rows.length - wrappers) + " are encoders; " + wrappers + " are adapters and painted wrappers around one, not counted as options elsewhere. "
      : "";
    return h("div", { class: "gx-gridview" }, [
      slotTabs,
      h("p", { class: "gx-fine" }, [rich(entry.task)]),
      docLinks(entry.pages),
      levelPicker,
      filters,
      h("p", { class: "gx-fine", "aria-live": "polite", text: rows.length + " of " + grid.rows.length + " options. " + adapterNote + caption }),
      legend,
      menu,
      h("div", { class: "gx-scroll", role: "region", "aria-label": entry.title + ": option by method" }, [table]),
    ]);
  }

  // ------------------------------------------------------------------------ run view

  /* The Run panel: a ported method on one frame, run by `grasppanda explore` with weights
   * its command line named. On the documentation site it is drawn greyed, with the
   * command that enables it. */
  var TYPE_PATH = "?type"; /* never an entry: a name starts with a letter or a digit, a file ends in .pth, .pt, .tar or .ckpt */
  var runForm = {
    target: "method",
    chosen: {},
    typing: false,
    typed: "",
    input: "example",
    depth: null,
    rgb: null,
    intrinsics: ["", "", "", ""],
    workspace: ["", "", "", "", "", ""],
    collision: true,
  };
  var runLive = { status: "", error: "", setup: null, job: null, sending: false, problem: "", failure: null, urls: null, timer: 0 };
  var FAILURE_TITLES = {
    weights: "The weights cannot be used",
    input: "The input was refused",
    "too-large": "The file is too large",
    busy: "Another run is in progress",
    request: "The request was refused",
  };

  function getJson(path) {
    return window.fetch(liveBase + path, { headers: { Accept: "application/json" } }).then(function (response) {
      return response.json();
    });
  }

  function loadSetup() {
    if (!liveBase || runLive.status) return;
    runLive.status = "loading";
    getJson("run/setup").then(
      function (setup) {
        runLive.setup = setup;
        runLive.status = "ready";
        if (setup.running && !runLive.job) follow(setup.running);
        render();
      },
      function (error) {
        runLive.status = "error";
        runLive.error = "the server did not answer (" + error + "). Is `grasppanda explore` still running?";
        render();
      }
    );
  }

  /* The methods the panel offers: those of the server, or those of the page. */
  function runMethods() {
    if (liveBase && runLive.setup) return runLive.setup.methods;
    return data.methods.map(function (m) {
      return { name: m.name, config: m.config, needs_rgb: !m.seed_based, suggested: "" };
    });
  }

  function runMethod() {
    var methods = runMethods();
    for (var i = 0; i < methods.length; i++) {
      if (methods[i].name === state.method) return methods[i];
    }
    return methods[0];
  }

  function runWeights(m) {
    if (runForm.typing) return runForm.typed.trim();
    return Object.prototype.hasOwnProperty.call(runForm.chosen, m.name) ? runForm.chosen[m.name] : m.suggested || "";
  }

  /* The selection of the builder, when there is one, with what compose said of it. */
  function runSelection() {
    var result = L.answer(data, state.method, state.picks);
    if (!result.sets.length) return null;
    if (liveBase) ensureLive(result);
    var done = live.status === "done" && live.report;
    return { sets: result.sets, accepted: done ? live.report.accepted : null, checking: live.status === "loading" };
  }

  function runSets() {
    var selection = runSelection();
    return runForm.target === "selection" && selection && selection.accepted !== false ? selection.sets : [];
  }

  function numbers(values) {
    var out = [];
    values.forEach(function (text) {
      var value = String(text).trim();
      out.push(value === "" || !isFinite(Number(value)) ? null : Number(value));
    });
    return out;
  }

  function megabytes(size) {
    if (size < 1024 * 1024) return Math.max(1, Math.round(size / 1024)) + " kB";
    return (size / 1024 / 1024).toFixed(1) + " MB";
  }

  function seconds(value) {
    return (value < 10 ? value.toFixed(1) : value.toFixed(0)) + " s";
  }

  /* What stops a run before anything is sent; "" when nothing does. */
  function runProblem() {
    var setup = runLive.setup;
    var m = runMethod();
    if (!setup) return "The server has not said yet which weights it reads.";
    if (!setup.weights.length) return "This server reads no weights: start it again with --weights-dir, as said above.";
    if (!runWeights(m)) return runForm.typing ? "Type the path of a weights file inside " + setup.weights_dir + "." : "Choose a weights file.";
    if (runForm.input !== "depth") return "";
    var limit = setup.limits.upload_bytes;
    if (!runForm.depth) return "Choose a depth image: a 16-bit PNG in millimetres or a .npy in metres.";
    var files = [runForm.depth, runForm.rgb].filter(Boolean);
    for (var i = 0; i < files.length; i++) {
      if (files[i].size > limit) return files[i].name + " is " + megabytes(files[i].size) + "; the server takes files up to " + megabytes(limit) + ".";
    }
    if (numbers(runForm.intrinsics).indexOf(null) >= 0) return "Fill in fx, fy, cx and cy: the intrinsics of the depth camera, in pixels.";
    var box = numbers(runForm.workspace);
    var filled = box.filter(function (value) { return value !== null; }).length;
    var typed = runForm.workspace.filter(function (text) { return String(text).trim() !== ""; }).length;
    if (typed && filled !== 6) return "Fill in all six numbers of the workspace box, or leave all six empty.";
    if (m.needs_rgb && !runForm.rgb) return m.name + " reads the colour image too: choose the colour image registered to the depth image.";
    return "";
  }

  function readBase64(file) {
    return new Promise(function (resolve, reject) {
      var reader = new FileReader();
      reader.onload = function () {
        var text = String(reader.result);
        resolve(text.slice(text.indexOf(",") + 1));
      };
      reader.onerror = function () {
        reject(reader.error);
      };
      reader.readAsDataURL(file);
    });
  }

  function blobUrl(base64, type) {
    var bytes = window.atob(base64);
    var array = new Uint8Array(bytes.length);
    for (var i = 0; i < bytes.length; i++) array[i] = bytes.charCodeAt(i);
    return URL.createObjectURL(new Blob([array], { type: type }));
  }

  function startRun() {
    var problem = runProblem();
    runLive.problem = problem;
    if (problem) {
      state.focusKey = "run-go";
      render();
      return;
    }
    var m = runMethod();
    var payload = { method: m.name, weights: runWeights(m), input: runForm.input, collision_filter: runForm.collision };
    var sets = runSets();
    if (sets.length) payload.sets = sets;
    var uploads = [];
    if (runForm.input === "depth") {
      payload.intrinsics = numbers(runForm.intrinsics);
      if (numbers(runForm.workspace).indexOf(null) < 0) payload.workspace = numbers(runForm.workspace);
      uploads.push(readBase64(runForm.depth.file).then(function (text) {
        payload.depth = { name: runForm.depth.name, data: text };
      }));
      if (runForm.rgb) {
        uploads.push(readBase64(runForm.rgb.file).then(function (text) {
          payload.rgb = { name: runForm.rgb.name, data: text };
        }));
      }
    }
    window.clearTimeout(runLive.timer);
    runLive.sending = true;
    runLive.failure = null;
    runLive.job = null;
    state.focusKey = "run-go";
    render();
    var out = document.getElementById("gx-run-out");
    if (out && window.innerWidth < 1000 && out.scrollIntoView) out.scrollIntoView({ block: "start", behavior: "smooth" });
    Promise.all(uploads)
      .then(function () {
        return post("run", payload);
      })
      .then(
        function (answer) {
          runLive.sending = false;
          if (answer.error) {
            runLive.failure = { title: FAILURE_TITLES[answer.kind] || "The request was refused", message: answer.error, kind: answer.kind || "" };
            render();
            return;
          }
          follow(answer);
        },
        function (error) {
          runLive.sending = false;
          runLive.failure = { title: "No answer from the server", message: String(error) + ". Is `grasppanda explore` still running?", kind: "server" };
          render();
        }
      );
  }

  /* Show a run and ask for its state until it ends; the panel is redrawn alone meanwhile. */
  function follow(status, quiet) {
    runLive.job = status;
    window.clearTimeout(runLive.timer);
    if (status.state !== "running") {
      if (runLive.urls) {
        URL.revokeObjectURL(runLive.urls.png);
        URL.revokeObjectURL(runLive.urls.npy);
      }
      runLive.urls = status.result
        ? { png: blobUrl(status.result.picture, "image/png"), npy: blobUrl(status.result.array, "application/octet-stream") }
        : null;
      render();
      return;
    }
    if (quiet) refreshRunOut();
    else render();
    runLive.timer = window.setTimeout(function () {
      getJson("run/" + status.id).then(
        function (next) {
          if (!runLive.job || runLive.job.id !== status.id) return;
          if (!next.id) {
            runLive.job = null;
            runLive.failure = { title: "The run is gone", message: next.error || "the server does not know this run", kind: "server" };
            render();
            return;
          }
          follow(next, true);
        },
        function (error) {
          runLive.job = null;
          runLive.failure = { title: "No answer from the server", message: String(error) + ". Is `grasppanda explore` still running?", kind: "server" };
          render();
        }
      );
    }, 400);
  }

  function refreshRunOut() {
    var old = document.getElementById("gx-run-out");
    if (old) old.parentNode.replaceChild(runOutput(), old);
  }

  function runStep(n, label, children, id) {
    return h("div", { class: "gx-step", role: "group", "aria-labelledby": id }, [
      h("div", { class: "gx-step-head" }, [h("span", { class: "gx-step-n", "aria-hidden": "true", text: String(n) }), h("span", { class: "gx-step-label", id: id, text: label })]),
    ].concat(children));
  }

  function pickRunMethod(name) {
    if (name === state.method || !L.method(data, name)) return;
    runForm.typing = false;
    runLive.problem = "";
    set({ method: name, picks: {}, notice: [] }, true);
  }

  function methodStep(off) {
    var m = runMethod();
    var children = [
      h("div", { class: "gx-methods" }, runMethods().map(function (item) {
        return h("button", {
          type: "button",
          class: "gx-method",
          "data-key": "run-method-" + item.name,
          "aria-pressed": item.name === m.name ? "true" : "false",
          disabled: off ? true : null,
          text: item.name,
          onclick: function () {
            pickRunMethod(item.name);
          },
        });
      })),
    ];
    var selection = off ? null : runSelection();
    if (selection) {
      var count = selection.sets.length;
      var mine = runForm.target === "selection" && selection.accepted !== false;
      children.push(h("div", { class: "gx-target", role: "group", "aria-label": "What to run" }, [
        toggle("run-target-method", "as shipped", !mine, function () {
          runForm.target = "method";
          render();
        }),
        h("button", {
          type: "button",
          class: "gx-toggle",
          "data-key": "run-target-selection",
          "aria-pressed": mine ? "true" : "false",
          disabled: selection.accepted === false ? true : null,
          text: "with your " + count + " change" + (count > 1 ? "s" : ""),
          onclick: function () {
            runForm.target = "selection";
            render();
          },
        }),
      ]));
      var said = selection.accepted === false
        ? " compose refuses this selection, so there is nothing to run."
        : selection.checking
          ? " compose is checking it …"
          : mine
            ? " The weights must fit the composed model: a change of post-processing or of the training recipe keeps the model, a new stage needs weights trained with it."
            : "";
      children.push(h("p", { class: "gx-fine" }, ["Your changes in Build a combination: "].concat(
        selection.sets.map(function (item, index) {
          return h("span", {}, [index ? ", " : "", h("code", { text: item })]);
        }),
        ["."]
      )));
      if (said) children.push(h("p", { class: "gx-fine", text: said.trim() }));
    }
    children.push(h("p", { class: "gx-fine" }, [
      h("code", { text: m.config }),
      m.needs_rgb ? " · reads a depth image and the colour image registered to it" : " · reads the point cloud of a depth image",
    ]));
    if (m.problem) children.push(h("p", { class: "gx-note", text: m.problem }));
    return runStep(1, "Method", children, "gx-run-l1");
  }

  function weightsStep(off) {
    var setup = runLive.setup;
    if (!off && setup && !setup.weights.length) {
      return runStep(2, "Weights", [
        h("div", { class: "gx-run-none" }, [
          h("p", {}, [h("b", { text: "This server reads no weights. " }), "Stop it (Ctrl+C) and start it again with a folder of weights, or with one file:"]),
          codeBlock("Every .pth, .pt, .tar or .ckpt file under a folder", "grasppanda explore --weights-dir <dir>", "run-dir"),
          codeBlock("One file, under a name of your choice", "grasppanda explore --weights gsnet=<file>", "run-file"),
        ]),
      ], "gx-run-l2");
    }
    var m = runMethod();
    var entries = !off && setup ? setup.weights : [];
    var value = runWeights(m);
    function option(entry) {
      return h("option", { value: entry.id, text: entry.label + " · " + megabytes(entry.bytes) });
    }
    var named = entries.filter(function (entry) { return entry.origin === "command line"; });
    var listed = entries.filter(function (entry) { return entry.origin !== "command line"; });
    var select = h("select", {
      class: "gx-select",
      id: "gx-run-weights",
      "data-key": "run-weights",
      "aria-labelledby": "gx-run-l2",
      disabled: off || !setup ? true : null,
      onchange: function (event) {
        var chosen = event.target.value;
        runForm.typing = chosen === TYPE_PATH;
        if (!runForm.typing) runForm.chosen[m.name] = chosen;
        runLive.problem = "";
        state.focusKey = runForm.typing ? "run-typed" : "run-weights";
        render();
      },
    }, [h("option", { value: "", text: off ? "a weights file of your machine" : entries.length ? "Choose a weights file…" : "…" })]);
    if (named.length) select.appendChild(h("optgroup", { label: "Named on the command line" }, named.map(option)));
    if (listed.length) select.appendChild(h("optgroup", { label: "In " + setup.weights_dir }, listed.map(option)));
    if (setup && setup.weights_dir) select.appendChild(h("option", { value: TYPE_PATH, text: "Another file in this folder: type its path…" }));
    select.value = runForm.typing ? TYPE_PATH : value;
    if (select.value !== (runForm.typing ? TYPE_PATH : value)) select.value = "";
    var children = [select];
    if (runForm.typing) {
      children.push(h("input", {
        type: "text",
        class: "gx-field gx-mono-field",
        "data-key": "run-typed",
        "aria-label": "Path of a weights file inside " + setup.weights_dir,
        placeholder: "a path inside the folder, e.g. run/checkpoints/epoch_0010.pth",
        spellcheck: "false",
        value: runForm.typed,
        oninput: function (event) {
          runForm.typed = event.target.value;
        },
      }));
    }
    var fine = [];
    if (!off && value && value === m.suggested && !runForm.typing) fine.push("Chosen by its name for " + m.name + "; the run says whether it fits. ");
    if (setup && setup.truncated) fine.push("The folder holds more files than are listed; type the path of another one. ");
    if (!off && setup) fine.push("The server reads only the files listed here: the page cannot name any other path.");
    if (fine.length) children.push(h("p", { class: "gx-fine", text: fine.join("") }));
    return runStep(2, "Weights", children, "gx-run-l2");
  }

  /* A file of the visitor's: a button that opens the picker, then the name and the size. */
  function fileRow(role, label, accept, hint, off) {
    var chosen = runForm[role];
    var id = "gx-run-file-" + role;
    var input = h("input", {
      type: "file",
      class: "gx-file-input",
      id: id,
      accept: accept,
      "data-key": "run-file-" + role,
      disabled: off ? true : null,
      onchange: function (event) {
        var file = event.target.files && event.target.files[0];
        if (!file) return;
        runForm[role] = { file: file, name: file.name, size: file.size };
        runLive.problem = "";
        state.focusKey = "run-file-" + role;
        render();
      },
    });
    return h("div", { class: "gx-file" }, [
      h("span", { class: "gx-field-label", text: label }),
      h("div", { class: "gx-file-row" }, [
        input,
        h("label", { for: id, class: "gx-file-button", text: chosen ? "Change" : "Choose a file" }),
        chosen
          ? h("span", { class: "gx-file-name" }, [h("code", { text: chosen.name }), " " + megabytes(chosen.size)])
          : h("span", { class: "gx-fine", text: hint }),
        chosen && role !== "depth"
          ? h("button", {
              type: "button",
              class: "gx-link",
              "data-key": "run-clear-" + role,
              text: "Remove",
              onclick: function () {
                runForm[role] = null;
                render();
              },
            })
          : null,
      ]),
      chosen ? h("p", { class: "gx-fine gx-file-hint", text: hint }) : null,
    ]);
  }

  /* Boxes for several numbers; pasting all of them into one box fills them all. */
  function numberBoxes(key, values, labels, off) {
    function paste(event) {
      var text = (event.clipboardData || window.clipboardData).getData("text");
      var found = String(text).match(/[-+]?(\d+\.?\d*|\.\d+)(e[-+]?\d+)?/gi) || [];
      if (found.length < values.length) return;
      event.preventDefault();
      for (var i = 0; i < values.length; i++) values[i] = found[i];
      runLive.problem = "";
      render();
    }
    return labels.map(function (label, index) {
      return h("label", { class: "gx-num" }, [
        h("span", { text: label }),
        h("input", {
          type: "text",
          inputmode: "decimal",
          class: "gx-field",
          "data-key": key + "-" + index,
          disabled: off ? true : null,
          spellcheck: "false",
          autocomplete: "off",
          value: values[index],
          oninput: function (event) {
            values[index] = event.target.value;
          },
          onpaste: paste,
        }),
      ]);
    });
  }

  function inputStep(off) {
    var m = runMethod();
    var choice = h("div", { class: "gx-target", role: "group", "aria-label": "Input" }, [
      h("button", {
        type: "button", class: "gx-toggle", "data-key": "run-input-example", disabled: off ? true : null,
        "aria-pressed": runForm.input === "example" ? "true" : "false", text: "The example scene",
        onclick: function () { runForm.input = "example"; runLive.problem = ""; render(); },
      }),
      h("button", {
        type: "button", class: "gx-toggle", "data-key": "run-input-depth", disabled: off ? true : null,
        "aria-pressed": runForm.input === "depth" ? "true" : "false", text: "Your depth image",
        onclick: function () { runForm.input = "depth"; runLive.problem = ""; render(); },
      }),
    ]);
    var children = [choice];
    if (runForm.input !== "depth") {
      children.push(h("p", { class: "gx-fine", text: "A synthetic table-top scene that ships with GraspPanda: six primitives on a table, a 1280 x 720 depth image with its colour image and intrinsics, no sensor noise. Nothing to upload." }));
      return runStep(3, "Input", children, "gx-run-l3");
    }
    children.push(fileRow("depth", "Depth image", ".png,.npy,.tif,.tiff", "16-bit PNG in millimetres, or .npy in metres", off));
    children.push(h("div", { class: "gx-field-group" }, [
      h("span", { class: "gx-field-label", text: "Intrinsics, in pixels" }),
      h("div", { class: "gx-nums" }, numberBoxes("run-k", runForm.intrinsics, ["fx", "fy", "cx", "cy"], off)),
      h("p", { class: "gx-fine", text: "Of the depth camera at the resolution of the image. Pasting 925,925,640,360 into one box fills all four." }),
    ]));
    children.push(fileRow("rgb", m.needs_rgb ? "Colour image (" + m.name + " needs it)" : "Colour image (optional)", ".png,.jpg,.jpeg",
      m.needs_rgb ? "registered to the depth image, pixel for pixel" : "registered to the depth image; it colours the picture", off));
    var labels = ["x min", "y min", "z min", "x max", "y max", "z max"];
    children.push(h("div", { class: "gx-field-group" }, [
      h("span", { class: "gx-field-label", text: "Workspace box (optional)" }),
      h("div", { class: "gx-nums gx-box" }, numberBoxes("run-ws", runForm.workspace, labels, off)),
      h("p", { class: "gx-fine", text: "Metres, camera frame: x right, y down, z forward. Only points inside are kept. Empty: every pixel with depth is used, and the run says what that means for this method." }),
    ]));
    return runStep(3, "Input", children, "gx-run-l3");
  }

  function optionsStep(off) {
    return runStep(4, "Post-processing", [
      h("label", { class: "gx-check" }, [
        h("input", {
          type: "checkbox",
          "data-key": "run-collision",
          checked: runForm.collision ? true : null,
          disabled: off ? true : null,
          onchange: function (event) {
            runForm.collision = event.target.checked;
          },
        }),
        h("span", {}, [h("b", { text: "Collision filter" }), " · removes grasps whose gripper would hit the observed points (the config's ", h("code", { text: "cd" }), " variant)"]),
      ]),
    ], "gx-run-l4");
  }

  function runLabel() {
    var m = runMethod();
    var what = m.name + (runSets().length ? " with your changes" : "");
    var where = runForm.input === "depth" ? (runForm.depth ? runForm.depth.name : "your depth image") : "the example scene";
    return "Run " + what + " on " + where;
  }

  function goButton(off) {
    var busy = Boolean(runLive.sending || (runLive.job && runLive.job.state === "running"));
    return h("div", { class: "gx-go" }, [
      h("button", {
        type: "button",
        class: "gx-run-button",
        "data-key": "run-go",
        disabled: off || busy || (runLive.setup && !runLive.setup.weights.length) ? true : null,
        "aria-busy": busy ? "true" : null,
        text: busy ? "Running …" : runLabel(),
        onclick: startRun,
      }),
      runLive.problem ? h("p", { class: "gx-run-problem", role: "alert", text: runLive.problem }) : null,
    ]);
  }

  /* Commands whose lines break between arguments, never inside a short one such as
   * --depth; the copy button copies the text as it is. */
  function commandBlock(label, text, key) {
    var block = codeBlock(label, text, key, false);
    var pre = block.querySelector(".gx-pre");
    pre.textContent = "";
    text.split("\n").forEach(function (line, row) {
      if (row) pre.appendChild(document.createTextNode("\n"));
      line.split(" ").forEach(function (word, column) {
        if (column) pre.appendChild(document.createTextNode(" "));
        pre.appendChild(h("span", { class: "gx-word-unit", text: word }));
      });
    });
    return block;
  }

  function stageList(stages) {
    var symbols = { done: "✓", running: "", waiting: "", failed: "✗" };
    return h("ol", { class: "gx-run-stages" }, stages.map(function (stage) {
      var time = stage.seconds === null || stage.seconds === undefined ? "" : seconds(stage.seconds);
      return h("li", { class: "is-" + stage.state }, [
        h("span", { class: "gx-run-dot", "aria-hidden": "true", text: symbols[stage.state] || "" }),
        h("span", { class: "gx-run-stage", text: stage.name.charAt(0).toUpperCase() + stage.name.slice(1) + (stage.note ? " (" + stage.note + ")" : "") }),
        h("span", { class: "gx-run-time", text: time }),
      ]);
    }));
  }

  function signed(value) {
    return (value < 0 ? "-" : "+") + Math.abs(value).toFixed(3);
  }

  function failureCard(failure, job) {
    var lines = String(failure.message).split("\n");
    var long = lines.length > 6;
    var hints = {
      weights: "Pick another file: weights must have been trained for, or converted to, the model of this config (tools/convert_checkpoint.py converts an author checkpoint).",
      dependency: "Install it, then start grasppanda explore again.",
      busy: "Wait for it to finish, then press Run again.",
    };
    var shown = long ? [lines[0], "…", lines[lines.length - 1]].join("\n") : failure.message;
    return h("div", { class: "gx-run-card k-refused", role: "alert" }, [
      h("div", { class: "gx-run-title" }, [mark({ kind: "refused", symbol: "✗" }), h("span", { text: failure.title })]),
      h("div", { class: "gx-run-message", text: shown }),
      long ? fold("run-message", "The whole message", [h("div", { class: "gx-run-message is-long", text: failure.message })]) : null,
      hints[failure.kind] && (job || failure.kind !== "weights") ? h("p", { class: "gx-fine", text: hints[failure.kind] }) : null,
      job ? stageList(job.stages) : null,
    ]);
  }

  function resultCard(result) {
    var steps = [{ name: "predicted", count: result.predicted }].concat(result.counts.map(function (item) {
      return { name: "after " + item.step, count: item.count };
    }));
    var most = Math.max.apply(null, steps.map(function (step) { return step.count; }).concat([1]));
    var summary = [h("b", { text: L.number(result.predicted) }), " grasps predicted"];
    if (result.counts.length) {
      summary.push(", ", h("b", { text: L.number(result.total) }), " left after " + result.counts.map(function (item) { return item.step; }).join(" and "));
    }
    summary.push(" · " + seconds(result.seconds.total) + (result.reused ? ", model kept from the last run" : ""));
    var caption = "The " + result.drawn + " best grasps as grippers, coloured by score rank from red (lowest shown) to green (highest)." +
      (result.input === "the example scene" ? " A synthetic scene, not a camera image." : "") +
      (result.drawn_frame !== result.frame ? " The picture shows the grasps before the step that changed their frame." : "");
    var rows = result.grasps.map(function (grasp) {
      return h("tr", {}, [
        h("td", { text: String(grasp.rank) }),
        h("td", { text: grasp.score.toFixed(3) }),
        h("td", { text: grasp.width.toFixed(3) }),
        h("td", { text: grasp.depth.toFixed(3) }),
        h("td", { text: grasp.position.map(signed).join("  ") }),
        h("td", { text: grasp.approach.map(function (value) { return (value < 0 ? "-" : "+") + Math.abs(value).toFixed(2); }).join("  ") }),
      ]);
    });
    var lines = [];
    result.lines.forEach(function (line) {
      lines.push(h("dt", { text: line[0] }));
      lines.push(h("dd", { text: line[1] }));
    });
    return h("div", { class: "gx-run-card k-runs" }, [
      h("div", { class: "gx-run-title" }, [mark({ kind: "runs", symbol: "✓" }), h("span", { text: result.target + " on " + result.input })]),
      h("p", { class: "gx-run-sum" }, summary),
      h("figure", { class: "gx-run-picture" }, [
        h("img", {
          src: runLive.urls ? runLive.urls.png : "",
          width: String(result.picture_size[0]),
          height: String(result.picture_size[1]),
          alt: "The " + result.drawn + " best grasps of " + result.target + " on " + result.input + ", drawn as grippers over the point cloud",
        }),
        h("figcaption", { class: "gx-fine", text: caption }),
      ]),
      h("div", { class: "gx-run-downloads" }, [
        h("a", { class: "gx-download", href: runLive.urls ? runLive.urls.npy : "", download: result.array_name, "data-key": "run-npy" }, [
          "Download the grasps ", h("span", { class: "gx-fine", text: result.array_shape[0] + " × 17 .npy" }),
        ]),
        h("a", { class: "gx-download", href: runLive.urls ? runLive.urls.png : "", download: result.array_name.replace(/\.npy$/, ".png"), "data-key": "run-png" }, [
          "Download the picture ", h("span", { class: "gx-fine", text: ".png" }),
        ]),
      ]),
      result.notes.length
        ? h("div", { class: "gx-why k-unstable" }, [h("div", { class: "gx-why-title", text: "Note" })].concat(result.notes.map(function (note) {
            return h("p", { text: note });
          })))
        : null,
      result.total
        ? h("div", {}, [
            h("div", { class: "gx-why-title", text: "The best " + rows.length + " of " + L.number(result.total) }),
            h("div", { class: "gx-run-table", role: "region", "aria-label": "The best grasps", tabindex: "0" }, [
              h("table", {}, [
                h("thead", {}, [h("tr", {}, ["rank", "score", "width m", "depth m", "position x y z, m", "approach x y z"].map(function (name) {
                  return h("th", { scope: "col", text: name });
                }))]),
                h("tbody", {}, rows),
              ]),
            ]),
          ])
        : h("p", { text: "No grasp was returned. That is a valid result: the detector found nothing it would grasp in this cloud" + (result.counts.length ? " or the filter removed everything." : ".") }),
      h("div", { class: "gx-why-title", text: "Through the post-processing" }),
      h("ul", { class: "gx-run-counts" }, steps.map(function (step) {
        return h("li", {}, [
          h("span", { text: step.name }),
          h("span", { class: "gx-run-bar", "aria-hidden": "true" }, [h("span", { style: "width:" + Math.max(1, Math.round((100 * step.count) / most)) + "%" })]),
          h("b", { text: L.number(step.count) }),
        ]);
      })),
      fold("run-ran", "What ran: config, weights, pipeline, input", [h("dl", { class: "gx-run-lines" }, lines)]),
      commandBlock("The same in a terminal", result.command, "run-command"),
      h("p", { class: "gx-fine", text: "Camera frame, metres: x right, y down, z forward. The approach is the first column of the rotation. The .npy holds one row per grasp, best first: score, width, height, depth, rotation (9, row-major), translation (3), object id." }),
    ]);
  }

  function runOutput() {
    var job = runLive.job;
    var body;
    if (!liveBase) {
      body = h("div", { class: "gx-run-card is-idle" }, [h("p", { text: "Here: the picture of the predicted grasps, the best ten in a table, the counts through the post-processing, and every grasp as a (G, 17) array to download." })]);
    } else if (runLive.failure) {
      body = failureCard(runLive.failure, null);
    } else if (runLive.sending) {
      body = h("div", { class: "gx-run-card k-none" }, [h("div", { class: "gx-run-title" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("span", { text: "Sending the request …" })])]);
    } else if (job && job.state === "running") {
      body = h("div", { class: "gx-run-card k-none" }, [
        h("div", { class: "gx-run-title" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("span", { text: "Running " + job.target + " on " + job.input })]),
        h("p", { class: "gx-fine", text: seconds(job.elapsed) + " so far. The first run of a method builds it and loads its weights; a second run reuses them." }),
        stageList(job.stages),
      ]);
    } else if (job && job.state === "failed") {
      body = failureCard(job.error, job);
    } else if (job && job.result) {
      body = resultCard(job.result);
    } else {
      body = h("div", { class: "gx-run-card is-idle" }, [
        h("p", { text: "Press Run: the picture of the predicted grasps, the best ten in a table, the counts through the post-processing and every grasp as a (G, 17) array appear here." }),
        h("p", { class: "gx-fine", text: "The first run of a method builds it and loads its weights; running it again, on another input, reuses them. Nothing is trained." }),
      ]);
    }
    return h("div", { class: "gx-run-out", id: "gx-run-out", "aria-live": "polite" }, [body]);
  }

  function runOff() {
    var command = "grasppanda explore --weights-dir <dir>";
    var button = h("button", {
      type: "button",
      class: "gx-copy",
      "data-key": "copy-run-explore",
      "data-label": "Copy",
      "aria-label": "Copy " + command,
      text: "Copy",
      onclick: function () {
        copyText(command, button);
      },
    });
    return h("div", { class: "gx-hint gx-run-off" }, [
      h("div", {}, [
        h("b", { text: "Runs on your machine. " }),
        "Start ",
        h("code", { text: command }),
        " with a folder of converted weights, and this panel runs the method you pick on the example scene or on your own depth image and shows the grasps. ",
        h("a", { href: docHref("user_guides/inference") + "#run-in-the-browser", text: "How it works" }),
      ]),
      button,
    ]);
  }

  function runView() {
    var off = !liveBase;
    if (!off) loadSetup();
    var top = null;
    if (off) top = runOff();
    else if (runLive.status === "loading") top = h("p", { class: "gx-fine", text: "Asking the server which weights it reads …" });
    else if (runLive.status === "error") top = h("div", { class: "gx-run-card k-fails", role: "alert" }, [h("p", { text: runLive.error })]);
    var form = h("div", { class: "gx-run-form" + (off ? " is-off" : ""), "aria-disabled": off ? "true" : null }, [
      methodStep(off),
      weightsStep(off),
      inputStep(off),
      optionsStep(off),
      goButton(off),
    ]);
    return h("div", { class: "gx-run" }, [top, h("div", { class: "gx-run-grid" }, [form, runOutput()])]);
  }

  // -------------------------------------------------------------------------- render

  function render() {
    var active = document.activeElement;
    var key = state.focusKey || (active && active.getAttribute ? active.getAttribute("data-key") : null);
    var caret = !state.focusKey && key && active.selectionStart !== undefined ? active.selectionStart : null;
    state.focusKey = "";
    hideTip();
    if (barWatch) barWatch.disconnect();
    var view = state.view === "grid" ? gridView() : state.view === "run" ? runView() : buildView();
    var body = h("div", { class: "gx-body" }, [header(), tabs(), state.view !== "build" ? notice() : null, view, state.view === "run" ? null : footer()]);
    var old = root.querySelector(".gx-body");
    if (old) root.replaceChild(body, old);
    else {
      root.textContent = "";
      root.appendChild(body);
    }
    state.flash = "";
    if (key) {
      var again = root.querySelector('[data-key="' + key.replace(/"/g, '\\"') + '"]');
      if (again) {
        again.focus({ preventScroll: true });
        if (caret !== null && again.setSelectionRange) {
          try {
            again.setSelectionRange(caret, caret);
          } catch (error) {
            /* a search field that does not support it */
          }
        }
      }
    }
    /* The floating verdict of a narrow screen stands in for the answer while the answer
     * is out of sight, and leaves when it is not. */
    var bar = root.querySelector(".gx-bar");
    var panel = document.getElementById("gx-result");
    if (bar && panel && window.IntersectionObserver) {
      barWatch = new window.IntersectionObserver(function (entries) {
        bar.classList.toggle("is-away", entries[entries.length - 1].isIntersecting);
      });
      barWatch.observe(panel);
    }
    var hash = "#" + L.toHash(state.method, state.picks, state.view, state.slot);
    try {
      if (window.location.hash !== hash) {
        if (pushNext && window.location.hash) window.history.pushState(null, "", hash);
        else window.history.replaceState(null, "", hash);
      }
    } catch (error) {
      /* file: pages of some browsers refuse it; the link is a convenience */
    }
    pushNext = false;
  }

  /* Take the selection from the address; what it names and the page does not know is
   * said, not dropped silently. */
  function readAddress() {
    var from = L.fromHash(window.location.hash);
    state.notice = L.dropped(data, from);
    state.method = L.method(data, from.method) ? from.method : data.methods[0].name;
    state.picks = L.normalise(data, state.method, from.picks);
    state.view = from.view;
    /* The server says which tab a page without an address opens on (the Run panel when
     * it was given weights). */
    var opens = root.getAttribute("data-view");
    if (!window.location.hash.replace(/^#/, "") && /^(build|grid|run)$/.test(opens || "")) state.view = opens;
    state.slot = L.lane(data, from.slot) ? from.slot : "";
    state.open = "";
    state.cellMenu = "";
  }

  function start(loaded) {
    data = L.prepare(loaded);
    var extra = document.getElementById("gp-explorer-contracts");
    if (extra && extra.textContent.trim()) {
      try {
        data.contracts = JSON.parse(extra.textContent);
      } catch (error) {
        data.contracts = null;
      }
    }
    readAddress();
    if (document.body.classList.contains("gx-standalone") && window.matchMedia) {
      var dark = window.matchMedia("(prefers-color-scheme: dark)");
      var apply = function () {
        root.classList.toggle("is-dark", dark.matches);
        document.body.classList.toggle("is-dark", dark.matches);
      };
      apply();
      if (dark.addEventListener) dark.addEventListener("change", apply);
    }
    root.classList.add("is-ready");
    document.addEventListener("keydown", function (event) {
      if (event.key === "Escape" && state.open) set({ open: "" });
      else if (event.key === "Escape" && state.cellMenu) {
        state.focusKey = state.cellMenu;
        set({ cellMenu: "" });
      }
    });
    /* Back and Forward of the browser walk through the picks. */
    window.addEventListener("popstate", function () {
      readAddress();
      render();
    });
    render();
  }

  var inline = document.getElementById("gp-explorer-data");
  var source = root.getAttribute("data-src");
  if (inline && inline.textContent.trim()) {
    start(JSON.parse(inline.textContent));
  } else if (source && window.fetch) {
    window
      .fetch(source)
      .then(function (response) {
        return response.json();
      })
      .then(start);
  }
})();
