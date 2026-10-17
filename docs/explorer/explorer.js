/* The combination explorer: the page around explorer_logic.js.
 *
 * No framework, no network: the data is the JSON element of the page (or `fetch`ed from
 * `data-src` when the page is served by `grasppanda explore`). What a selection shows is
 * decided in explorer_logic.js; this file only draws it. Every change of the state draws
 * the tool again (render); popovers and tooltips are placed after the drawing.
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
    pickerActive: 0,
    filter: { q: "", level: "runs", classes: [], extra: false, weights: false },
    yamlTab: "",
    repro: "compose",
    folds: {},
    flash: "",
    notice: [],
    cellMenu: "",
    gridFocus: "",
    focusKey: "",
    about: false,
    filters: false,
    docsMenu: false,
  };
  var pushNext = false;
  var watchers = [];
  var docsBase = root.getAttribute("data-docs") || "../";
  /* Set by `grasppanda explore`: the address of the compose check of this installation. */
  var liveBase = root.getAttribute("data-live") || "";
  var live = { key: "", status: "", report: null, error: "" };
  var saving = { key: "", name: "", status: "", message: "" };
  var tip = null;
  var MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // -------------------------------------------------------------------------- icons

  /* One stroke style (1.75 px, round caps, currentColor), drawn on a 24 px grid. */
  var ICONS = {
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    x: '<path d="M6.5 6.5l11 11M17.5 6.5l-11 11"/>',
    minus: '<path d="M6 12h12"/>',
    slash: '<path d="M7.5 16.5l9-9"/>',
    ban: '<circle cx="12" cy="12" r="7.5"/><path d="M6.8 17.2L17.2 6.8"/>',
    alert: '<path d="M10.3 4.2L2.8 17.5A2 2 0 0 0 4.5 20.5h15a2 2 0 0 0 1.7-3L13.7 4.2a2 2 0 0 0-3.4 0z"/><path d="M12 9.5v4"/><path d="M12 17h.01"/>',
    alertCircle: '<circle cx="12" cy="12" r="9"/><path d="M12 7.5v5"/><path d="M12 16.2h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5.5"/><path d="M12 7.6h.01"/>',
    copy: '<rect x="8.5" y="8.5" width="12" height="12" rx="2"/><path d="M15.5 8.5V5.5a2 2 0 0 0-2-2h-8a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h3"/>',
    chevronDown: '<path d="M6 9l6 6 6-6"/>',
    chevronRight: '<path d="M9 6l6 6-6 6"/>',
    chevronsUpDown: '<path d="M8 9.5l4-4 4 4"/><path d="M8 14.5l4 4 4-4"/>',
    search: '<circle cx="11" cy="11" r="6.5"/><path d="M20 20l-4.2-4.2"/>',
    link: '<path d="M10 13.5a4 4 0 0 0 5.7.3l3-3a4 4 0 0 0-5.7-5.7l-1.2 1.2"/><path d="M14 10.5a4 4 0 0 0-5.7-.3l-3 3a4 4 0 0 0 5.7 5.7l1.2-1.2"/>',
    reset: '<path d="M4 12a8 8 0 1 0 2.4-5.7L4 8.7"/><path d="M4 4v4.7h4.7"/>',
    play: '<path d="M7.5 5.5v13l11-6.5z"/>',
    external: '<path d="M14 4h6v6"/><path d="M20 4l-9 9"/><path d="M18 14v4a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4"/>',
    dot: '<circle cx="12" cy="12" r="3.5" fill="currentColor"/>',
    circle: '<circle cx="12" cy="12" r="7"/>',
    arrowRight: '<path d="M5 12h14"/><path d="M13 6l6 6-6 6"/>',
    filter: '<path d="M4 6h16"/><path d="M7 12h10"/><path d="M10 18h4"/>',
    download: '<path d="M12 4v11"/><path d="M7 10.5l5 5 5-5"/><path d="M5 20h14"/>',
    upload: '<path d="M12 20V9"/><path d="M7 13.5l5-5 5 5"/><path d="M5 4h14"/>',
    image: '<rect x="3.5" y="4.5" width="17" height="15" rx="2"/><circle cx="9" cy="10" r="1.8"/><path d="M20.5 16l-5-5-9 8.5"/>',
    layers: '<path d="M12 3.5l8.5 4.5L12 12.5 3.5 8z"/><path d="M3.5 12L12 16.5l8.5-4.5"/><path d="M3.5 16L12 20.5l8.5-4.5"/>',
    grid: '<rect x="4" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="4" width="6.5" height="6.5" rx="1.5"/><rect x="4" y="13.5" width="6.5" height="6.5" rx="1.5"/><rect x="13.5" y="13.5" width="6.5" height="6.5" rx="1.5"/>',
    terminal: '<rect x="3" y="4.5" width="18" height="15" rx="2"/><path d="M7 9.5l3 2.5-3 2.5"/><path d="M12.5 15h4.5"/>',
    book: '<path d="M4.5 5.5A2 2 0 0 1 6.5 3.5h13v14h-13a2 2 0 0 0-2 2z"/><path d="M4.5 19.5a2 2 0 0 0 2 2h13v-4"/>',
  };

  function icon(name, size) {
    var px = String(size || 16);
    var holder = document.createElement("span");
    holder.innerHTML =
      '<svg class="gx-i" viewBox="0 0 24 24" width="' + px + '" height="' + px + '" style="width:' + px + "px;height:" + px + 'px" fill="none" stroke="currentColor" ' +
      'stroke-width="1.75" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true" focusable="false">' +
      (ICONS[name] || ICONS.dot) + "</svg>";
    return holder.firstChild;
  }

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
  function rich(text, tag, attrs) {
    var node = h(tag || "span", attrs || {});
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

  function plain(text) {
    return String(text || "").replace(/`/g, "");
  }

  function capital(text) {
    text = String(text || "");
    return text.charAt(0).toUpperCase() + text.slice(1);
  }

  var num = L.thousands;

  /* 15353775 -> "15.4 M". */
  function compact(value) {
    if (value >= 1e6) return (value / 1e6).toFixed(value >= 1e8 ? 0 : value >= 1e7 ? 1 : 2) + " M";
    if (value >= 1e3) return Math.round(value / 1e3) + " k";
    return String(value);
  }

  /* A loss value with three significant digits. */
  function short(value) {
    if (Math.abs(value) >= 1000) return num(Math.round(value));
    return String(Number(Number(value).toPrecision(3)));
  }

  function percent(value) {
    var rounded = Math.round(value * 100);
    return (rounded < 0 ? "−" : rounded > 0 ? "+" : "") + Math.abs(rounded) + "%";
  }

  /* "2026-10-15@0ddca31" -> "15 Oct 2026". */
  function day(run) {
    var found = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(run || ""));
    if (!found) return String(run || "");
    return Number(found[3]) + " " + MONTHS[Number(found[2]) - 1] + " " + found[1];
  }

  /* A page of the documentation. The site uses directory addresses, which a browser does
   * not resolve when the site is opened from a file. */
  function docHref(path) {
    return docsBase + path + "/" + (window.location.protocol === "file:" ? "index.html" : "");
  }

  function slotHref(stem) {
    return docHref("slots/" + stem);
  }

  /* The task of a slot without the slot's name in front: "Scene encoder: points to ..." */
  function task(entry) {
    var text = plain(entry.task);
    var at = text.indexOf(": ");
    if (at > 0 && at < 30 && text.slice(0, at).toLowerCase() === entry.title.toLowerCase()) text = capital(text.slice(at + 2));
    return text;
  }

  /* Change the state and draw. `push` makes the change a step of the browser history. */
  function set(patch, push) {
    Object.keys(patch).forEach(function (name) {
      state[name] = patch[name];
    });
    if (push) pushNext = true;
    render();
  }

  function closePopovers() {
    state.open = "";
    state.cellMenu = "";
    state.about = false;
    state.filters = false;
    state.docsMenu = false;
  }

  function pick(slot, name) {
    var picks = {};
    Object.keys(state.picks).forEach(function (key) {
      picks[key] = state.picks[key];
    });
    if (name) picks[slot] = name;
    else delete picks[slot];
    state.focusKey = "lane-" + slot;
    set({ picks: L.normalise(data, state.method, picks), flash: slot, notice: [], open: "", laneQuery: "" }, true);
  }

  // ---------------------------------------------------------------- shared pieces

  function copyText(text, button) {
    var label = button.querySelector(".gx-copy-label");
    var before = label ? label.textContent : "";
    function done(ok) {
      button.classList.add(ok ? "is-copied" : "is-failed");
      var old = button.querySelector(".gx-i");
      if (old) old.parentNode.replaceChild(icon(ok ? "check" : "x", 14), old);
      if (label) label.textContent = ok ? "Copied" : "Press Ctrl+C";
      window.setTimeout(function () {
        button.classList.remove("is-copied", "is-failed");
        var now = button.querySelector(".gx-i");
        if (now) now.parentNode.replaceChild(icon(button.getAttribute("data-icon") || "copy", 14), now);
        if (label) label.textContent = before;
      }, 1500);
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

  /* A button that copies `text`: an icon, and a label that may be visually hidden. */
  function copyButton(text, key, what, opts) {
    opts = opts || {};
    var button = h("button", {
      type: "button",
      class: "gx-copy" + (opts.cls ? " " + opts.cls : ""),
      "data-key": key,
      "data-icon": opts.icon || "copy",
      "aria-label": "Copy " + what,
      title: opts.title || "Copy",
      onclick: function () {
        copyText(text, button);
      },
    }, [icon(opts.icon || "copy", 14), h("span", { class: "gx-copy-label" + (opts.showLabel ? "" : " gx-sr"), text: opts.label || "Copy" })]);
    return button;
  }

  /* A command in a terminal: always dark, in both themes. */
  function terminal(text, key, what) {
    /* Lines break between arguments, never inside a short one such as --weights-dir; a
     * long path may break where it must. The copy button copies the text as it is. */
    var words = [];
    String(text).split("\n").forEach(function (line, row) {
      if (row) words.push("\n");
      line.split(" ").forEach(function (word, column) {
        if (column) words.push(" ");
        words.push(h("span", { class: word.length > 28 ? "gx-tok is-long" : "gx-tok", text: word }));
      });
    });
    return h("div", { class: "gx-term" }, [
      h("pre", { class: "gx-term-pre", tabindex: "0", role: "group", "aria-label": what }, [
        h("span", { class: "gx-term-prompt", "aria-hidden": "true", text: "$ " }),
      ].concat(words)),
      copyButton(text, "copy-" + key, what, { cls: "is-term", showLabel: false }),
    ]);
  }

  /* A file fragment (YAML): a light code box. */
  function codeBox(text, key, what) {
    return h("div", { class: "gx-codebox" }, [
      h("pre", { class: "gx-codebox-pre", tabindex: "0", role: "group", "aria-label": what, text: text }),
      copyButton(text, "copy-" + key, what, { cls: "is-box" }),
    ]);
  }

  var TONES = {
    runs: "ok",
    adapter: "ok",
    accepted: "ok",
    empty: "empty",
    unstable: "unstable",
    refused: "refused",
    fails: "fails",
    mixed: "empty",
    none: "none",
  };

  /* How a mark of the logic is drawn: tone (its colour), label, icon, and whether it is the
   * answer of the contract check (an outline, never filled). */
  function status(badge, cell) {
    var kind = badge.kind;
    var contract = Boolean(badge.check || kind === "accepted");
    var label = {
      runs: "Runs",
      adapter: "Runs",
      empty: "Runs, no grasps",
      unstable: "Unstable",
      refused: contract ? "Refused by compose" : "Refused",
      fails: "Fails",
      accepted: "Accepted by compose",
      mixed: "Mixed",
      none: "Not measured",
    }[kind] || capital(badge.label);
    if (kind === "refused" && !contract && cell && cell.kind) label += " · " + cell.kind.replace(/-/g, " ");
    var glyph = { ok: "check", empty: "minus", unstable: "alert", refused: "ban", fails: "alertCircle", none: "circle" }[TONES[kind] || "none"];
    return { tone: TONES[kind] || "none", label: label, icon: glyph, contract: contract };
  }

  function pill(badge, cell, extra) {
    var s = status(badge, cell);
    return h("span", { class: "gx-pill s-" + s.tone + (s.contract ? " is-contract" : "") + (extra ? " " + extra : "") }, [icon(s.icon, 14), h("span", { text: s.label })]);
  }

  /* The 8 px mark of the picker and of the small lists. */
  function dot(badge, title) {
    var s = status(badge);
    return h("span", { class: "gx-dot s-" + s.tone + (s.contract ? " is-contract" : ""), title: title || null, "aria-hidden": "true" });
  }

  function alertBox(tone, glyph, title, body, extra) {
    return h("div", { class: "gx-alert s-" + tone + (extra ? " " + extra : ""), role: tone === "refused" || tone === "fails" ? "note" : null }, [
      h("span", { class: "gx-alert-icon" }, [icon(glyph, 16)]),
      h("div", { class: "gx-alert-body" }, [h("div", { class: "gx-alert-title", text: title })].concat(body || [])),
    ]);
  }

  function messageList(messages) {
    return h("ul", { class: "gx-msgs" }, messages.map(function (message) {
      return rich(message, "li");
    }));
  }

  /* An "i" that explains: a tooltip on hover and focus, the same text on a tap. */
  function infoButton(text, what) {
    var button = h("button", {
      type: "button",
      class: "gx-info",
      "aria-label": what || "More",
      onmouseenter: function (event) {
        showTip(event.currentTarget, text);
      },
      onmouseleave: hideTip,
      onfocus: function (event) {
        showTip(event.currentTarget, text);
      },
      onblur: hideTip,
      onclick: function (event) {
        showTip(event.currentTarget, text);
      },
    }, [icon("info", 14)]);
    return button;
  }

  /* A part that is closed until asked for; it stays as the visitor left it. */
  function fold(key, summary, children, meta) {
    var open = Boolean(state.folds[key]);
    return h("div", { class: "gx-fold" + (open ? " is-open" : "") }, [
      h("button", {
        type: "button",
        class: "gx-fold-head",
        "data-key": "fold-" + key,
        "aria-expanded": open ? "true" : "false",
        "aria-controls": "gx-fold-" + key,
        onclick: function () {
          var folds = {};
          Object.keys(state.folds).forEach(function (name) {
            folds[name] = state.folds[name];
          });
          folds[key] = !open;
          state.focusKey = "fold-" + key;
          set({ folds: folds });
        },
      }, [h("span", { class: "gx-fold-title", text: summary }), meta ? h("span", { class: "gx-fold-meta", text: meta }) : null, icon("chevronDown", 16)]),
      open ? h("div", { class: "gx-fold-body", id: "gx-fold-" + key }, children) : null,
    ]);
  }

  /* A segmented control: [[value, label, key]], one pressed. */
  function segmented(items, value, onpick, label, extra) {
    return h("div", { class: "gx-seg" + (extra ? " " + extra : ""), role: "group", "aria-label": label }, items.map(function (item) {
      return h("button", {
        type: "button",
        class: "gx-seg-item",
        "data-key": item[2],
        "aria-pressed": item[0] === value ? "true" : "false",
        title: item[3] || null,
        disabled: item[4] ? true : null,
        onclick: function () {
          onpick(item[0]);
        },
      }, [item[1]]);
    }));
  }

  function keyValues(rows) {
    var items = [];
    rows.forEach(function (row) {
      if (!row) return;
      items.push(h("div", { class: "gx-kv-row" }, [h("dt", { text: row[0] }), h("dd", {}, [].concat(row[1]))]));
    });
    return h("dl", { class: "gx-kv" }, items);
  }

  function external(href, text, key) {
    return h("a", { class: "gx-link", href: href, "data-key": key || null }, [h("span", { text: text }), icon("external", 14)]);
  }

  // ---------------------------------------------------------------------- tooltip

  function hideTip() {
    if (tip) tip.hidden = true;
  }

  /* A tooltip above the target (below it when there is no room); `content` is text or a
   * node. A rich one (`side`) stands beside its target instead, right of it or left when
   * there is no room, centred on it and never above the header of its table. It lives
   * outside the drawn body, so drawing again does not drop it mid-hover. */
  function showTip(target, content, side) {
    if (!tip) {
      tip = h("div", { class: "gx-tip", role: "tooltip", id: "gx-tip" });
      root.appendChild(tip);
    }
    tip.textContent = "";
    tip.className = "gx-tip" + (side ? " is-rich" : "");
    tip.appendChild(typeof content === "string" ? rich(content, "div") : content);
    tip.hidden = false;
    var box = target.getBoundingClientRect();
    var host = root.getBoundingClientRect();
    var page = document.documentElement.clientWidth;
    var width = tip.offsetWidth;
    var height = tip.offsetHeight;
    var left;
    var top;
    if (side) {
      left = box.right + 10;
      if (left + width > page - 8) left = box.left - 10 - width;
      left = Math.max(left, 8);
      top = box.top + box.height / 2 - height / 2;
      var table = target.closest ? target.closest("table") : null;
      var head = table ? table.querySelector("thead") : null;
      var floor = Math.max(head ? head.getBoundingClientRect().bottom + 4 : 0, 8);
      top = Math.max(top, floor);
      tip.style.left = left - host.left + "px";
      tip.style.top = top - host.top + "px";
    } else {
      left = box.left - host.left + box.width / 2 - width / 2;
      var limit = Math.max(page - host.left - width - 8, 0);
      left = Math.min(Math.max(left, 8 - host.left), limit);
      top = box.top - host.top - height - 8;
      if (box.top - height - 8 < 56) top = box.bottom - host.top + 8;
      tip.style.left = left + "px";
      tip.style.top = top + "px";
    }
    target.setAttribute("aria-describedby", "gx-tip");
  }

  // --------------------------------------------------------------------- popovers

  /* Place a popover under (or over) its anchor, inside the page width. */
  function place(pop, anchor, align) {
    if (!pop || !anchor) return;
    var sheet = pop.classList.contains("gx-picker") && window.matchMedia && window.matchMedia("(max-width: 599px)").matches;
    var list = pop.querySelector(".gx-picker-list");
    if (sheet) {
      pop.classList.add("is-sheet");
      pop.style.left = "";
      pop.style.top = "";
      if (list) list.style.maxHeight = "";
      pop.hidden = false;
      return;
    }
    pop.classList.remove("is-sheet");
    pop.hidden = false;
    var box = anchor.getBoundingClientRect();
    var host = (pop.offsetParent || root).getBoundingClientRect();
    var page = document.documentElement.clientWidth;
    if (pop.classList.contains("gx-picker")) pop.style.width = Math.min(Math.max(box.width, 380), page - 24) + "px";
    var width = pop.offsetWidth;
    var left = align === "end" ? box.right - host.left - width : box.left - host.left;
    left = Math.min(Math.max(left, 12 - host.left), page - host.left - width - 12);
    var below = window.innerHeight - box.bottom - 16;
    var above = box.top - 16 - 56;
    if (list) {
      var chrome = pop.offsetHeight - list.offsetHeight;
      var room = Math.max(below, above);
      list.style.maxHeight = Math.max(Math.min(room - chrome, 340), 168) + "px";
    }
    var height = pop.offsetHeight;
    var top = box.bottom - host.top + 6;
    /* Below the anchor when it fits (or when the list can shrink to a usable height there),
     * above it otherwise. */
    if (height > below && above > below && below < 320) top = box.top - host.top - height - 6;
    else if (list && height > below) {
      list.style.maxHeight = Math.max(below - (height - list.offsetHeight), 168) + "px";
    }
    pop.style.left = left + "px";
    pop.style.top = top + "px";
  }

  function anchorOf(key) {
    return root.querySelector('[data-key="' + key + '"]');
  }

  function placeAll() {
    var pops = root.querySelectorAll(".gx-pop[data-anchor]");
    for (var i = 0; i < pops.length; i++) {
      place(pops[i], anchorOf(pops[i].getAttribute("data-anchor")), pops[i].getAttribute("data-align"));
    }
  }

  // -------------------------------------------------------------------------- header

  var VERDICT_LABEL = { runs: "Runs", "runs-empty": "Runs, no grasps", "runs-unstable": "Unstable", refused: "Refused", fails: "Fails" };
  var VERDICT_TONE = { runs: "ok", "runs-empty": "empty", "runs-unstable": "unstable", refused: "refused", fails: "fails" };

  function header() {
    var counts = data.counts;
    var parts = data.verdict_order.map(function (name) {
      return [name, counts.by_verdict[name] || 0];
    });
    var summary = parts.map(function (part) {
      return num(part[1]) + " " + VERDICT_LABEL[part[0]].toLowerCase();
    }).join(", ");
    var bar = h("div", { class: "gx-mix", role: "img", "aria-label": "Verdicts of the records: " + summary }, parts.map(function (part) {
      return h("span", {
        class: "gx-mix-seg s-" + VERDICT_TONE[part[0]],
        style: "flex-grow:" + part[1],
        onmouseenter: function (event) {
          showTip(event.currentTarget, num(part[1]) + " · " + VERDICT_LABEL[part[0]]);
        },
        onmouseleave: hideTip,
      });
    }));
    var about = h("button", {
      type: "button",
      class: "gx-iconbtn",
      "data-key": "about",
      "aria-label": "About these records",
      "aria-expanded": state.about ? "true" : "false",
      "aria-controls": "gx-about",
      onclick: function () {
        var open = !state.about;
        closePopovers();
        state.focusKey = "about";
        set({ about: open });
      },
    }, [icon("info", 16)]);
    return h("header", { class: "gx-head" }, [
      h("div", { class: "gx-head-text", "aria-hidden": "true" }, [
        h("div", { class: "gx-head-title" }, [h("span", { text: "Combination explorer" }), liveBase ? h("span", { class: "gx-live-chip" }, [h("span", { class: "gx-live-dot" }), "Live"]) : null]),
        h("p", { class: "gx-head-sub", text: "Swap any stage of a grasp detector and see what happened when it was built and trained." }),
      ]),
      h("div", { class: "gx-head-data" }, [
        h("div", { class: "gx-head-stat" }, [
          h("div", { class: "gx-head-count" }, [h("b", { text: num(counts.records) }), h("span", { text: " records" })]),
          bar,
        ]),
        about,
      ]),
      state.about ? aboutPop() : null,
    ]);
  }

  /* What stands behind the numbers of the page, computed from the data. */
  function aboutPop() {
    var counts = data.counts;
    var rows = [
      ["With a probe of their own", num(counts.probed)],
      ["Sharing the probe of another record", num(counts.shared_probe)],
      ["Refused by compose, nothing built", num(counts.by_verdict.refused || 0)],
      ["Stage sets probed", num(counts.stage_sets.recorded) + " of " + num(counts.stage_sets.listed)],
      ["Grid combinations with a record", num(counts.grid - counts.grid_without_record) + " of " + num(counts.grid)],
    ];
    if (data.contracts) {
      var c = data.contracts.counts;
      rows.push(["Answers of compose, no probe", num(c.answers)]);
    }
    return h("div", { class: "gx-pop gx-about", id: "gx-about", role: "dialog", "aria-label": "About these records", "data-anchor": "about", "data-align": "end", hidden: true }, [
      h("div", { class: "gx-pop-title", text: "About these records" }),
      h("p", { class: "gx-pop-lead", text: data.no_accuracy }),
      h("ul", { class: "gx-about-mix" }, data.verdict_order.map(function (name) {
        return h("li", {}, [
          h("span", { class: "gx-dot s-" + VERDICT_TONE[name], "aria-hidden": "true" }),
          h("span", { text: VERDICT_LABEL[name] }),
          h("b", { text: num(counts.by_verdict[name] || 0) }),
        ]);
      })),
      h("div", { class: "gx-label", text: "Behind the numbers" }),
      h("dl", { class: "gx-about-rows" }, rows.map(function (row) {
        return h("div", {}, [h("dt", { text: row[0] }), h("dd", { text: row[1] })]);
      })),
      data.caveat ? h("p", { class: "gx-pop-fine", text: "Class: " + data.caveat + "." }) : null,
      h("div", { class: "gx-pop-foot" }, [external(docHref("advanced_guides/measured_combinations"), "Measured combinations")]),
    ]);
  }

  function tabs() {
    var views = [["build", "Build a combination", "Build"], ["grid", "Compare all options of a slot", "Compare"], ["run", "Run a method", "Run"]];
    var glyphs = { build: "layers", grid: "grid", run: "play" };
    function go(view) {
      state.focusKey = "tab-" + view;
      closePopovers();
      set({ view: view }, true);
    }
    return h("div", { class: "gx-tabs", role: "tablist", "aria-label": "View" }, views.map(function (item, index) {
      var on = state.view === item[0];
      return h("button", {
        type: "button",
        role: "tab",
        class: "gx-tab",
        id: "gx-tab-" + item[0],
        "data-key": "tab-" + item[0],
        "aria-selected": on ? "true" : "false",
        "aria-controls": "gx-panel",
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
      }, [
        icon(glyphs[item[0]], 14),
        h("span", { text: item[2] }),
        item[0] === "run" && !liveBase ? h("span", { class: "gx-tab-tag", text: "Local" }) : null,
      ]);
    }));
  }

  /* What a link named and this page does not know: said, not dropped silently. */
  function notice() {
    if (!state.notice.length) return null;
    var children = [h("span", { text: "This link named " })];
    state.notice.forEach(function (text, index) {
      if (index) children.push(h("span", { text: index === state.notice.length - 1 ? " and " : ", " }));
      children.push(rich(text));
    });
    children.push(h("span", { text: ", which this version does not know. Showing " + state.method + (Object.keys(state.picks).length ? " with the rest of the link." : " as shipped.") }));
    return h("div", { class: "gx-notice", role: "status" }, [
      icon("info", 16),
      h("p", {}, children),
      h("button", {
        type: "button",
        class: "gx-ghost is-small",
        "data-key": "notice-close",
        text: "Dismiss",
        onclick: function () {
          set({ notice: [] });
        },
      }),
    ]);
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
        live.error = "The server did not answer (" + error + "). Is `grasppanda explore` still running?";
        render();
      }
    );
  }

  function saveConfig(result, overwrite) {
    var key = live.key;
    saving.status = "saving";
    render();
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

  /* The compose answer of this installation, for a selection with changes. */
  function liveBlock(result) {
    if (!liveBase || !result.sets.length || !live.status) return null;
    var head = h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "This installation" })]);
    if (live.status === "loading") {
      return h("div", { class: "gx-live" }, [head, h("p", { class: "gx-live-wait" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), "Composing with this installation…"])]);
    }
    if (live.status === "error") {
      return h("div", { class: "gx-live" }, [head, alertBox("fails", "alertCircle", "No answer from the server", [rich(live.error, "p")])]);
    }
    var report = live.report;
    if (!report) return null;
    var children = [head];
    var recorded = result.cell ? result.cell.verdict !== "refused" : null;
    if (report.accepted) {
      children.push(h("p", { class: "gx-live-line" }, [
        h("span", { class: "gx-muted", text: "Composed config, hash" }),
        h("code", { text: String(report.config_hash || "").slice(0, 12) }),
      ]));
    } else {
      var kinds = (report.refusal_kinds || []).map(function (item) {
        return typeof item === "string" ? item : item && item.kind;
      }).filter(function (kind, index, all) {
        return kind && all.indexOf(kind) === index;
      });
      var items = report.problems.map(function (message) {
        return rich(message, "li");
      }).concat(report.issues.map(function (issue) {
        return h("li", {}, [h("code", { text: issue.path }), " ", rich(issue.message)]);
      }));
      children.push(alertBox("refused", "ban", "Why compose refuses it" + (kinds.length ? " · " + kinds.join(", ").replace(/-/g, " ") : ""), [h("ul", { class: "gx-msgs" }, items)]));
    }
    if (recorded !== null && recorded !== report.accepted) {
      children.push(alertBox("unstable", "alert", "The record and this installation disagree", [h("p", { text: "The record was measured with another version of the code." })]));
    }
    if (report.stages.length) {
      children.push(h("ul", { class: "gx-stages" }, report.stages.map(function (stage) {
        var changed = !/^base config/.test(stage.origin);
        return h("li", { class: changed ? "is-set" : "" }, [
          h("span", { class: "gx-stages-slot", text: stage.slot }),
          h("code", { text: stage.component }),
          h("span", { class: "gx-stages-origin", title: plain(stage.origin), text: changed ? plain(stage.origin) : "Default" }),
        ]);
      })));
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
          class: "gx-input is-mono",
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
          class: "gx-btn",
          "data-key": "save",
          disabled: busy ? true : null,
          onclick: function () {
            saveConfig(result, saving.status === "exists");
          },
        }, [busy ? h("span", { class: "gx-spin", "aria-hidden": "true" }) : icon("download", 14), h("span", { text: saving.status === "exists" ? "Overwrite" : "Save config" })]),
      ]));
      if (saving.status === "written") {
        children.push(h("p", { class: "gx-live-saved" }, [icon("check", 14), h("span", { text: "Saved " }), h("code", { text: saving.message }), h("span", { text: " next to the server. Next: " }), h("code", { text: L.probeCommand(data, saving.message, result.cell) })]));
      } else if (saving.status === "exists" || saving.status === "error") {
        children.push(h("p", { class: "gx-live-problem", role: "alert", text: saving.message }));
      }
    }
    return h("div", { class: "gx-live" }, children);
  }

  // ---------------------------------------------------------------------- build view

  /* "GSNet (Wang et al., ICCV 2021)" -> "ICCV 2021". */
  function venue(m) {
    var found = /,\s*([^,()]+\d{4})\)\s*$/.exec(m.source || "");
    return found ? found[1] : "";
  }

  function methodTiles(methods, current, keyOf, onpick, off) {
    return h("div", { class: "gx-mtiles", role: "group", "aria-label": "Method" }, methods.map(function (m) {
      var known = L.method(data, m.name) || m;
      var on = m.name === current;
      return h("button", {
        type: "button",
        class: "gx-mtile",
        "data-key": keyOf(m),
        "aria-pressed": on ? "true" : "false",
        disabled: off ? true : null,
        title: (known.source || m.name) + (known.config ? " · " + known.config : ""),
        onclick: function () {
          onpick(m.name);
        },
      }, [
        h("span", { class: "gx-mtile-name", text: m.name }),
        venue(known) ? h("span", { class: "gx-mtile-venue", text: venue(known) }) : null,
        on ? h("span", { class: "gx-mtile-check", "aria-hidden": "true" }, [icon("check", 12)]) : null,
      ]);
    }));
  }

  /* The name a slot shows when the method keeps its own option. */
  function shippedName(slot, ships) {
    return ships.option || ships.label || ships.component || "";
  }

  function examples() {
    var wanted = [
      ["adapter", "Adapter inserted"],
      ["refused", "Refused swap"],
      ["unstable", "Unstable run"],
      ["fails", "Failure"],
    ];
    /* A structural refusal that compose still refuses: the example must show the conflict. */
    var preferred = { refused: "stage/EconomicGrasp/view/objectness-view" };
    var found = [];
    wanted.forEach(function (item) {
      var best = null;
      for (var i = 0; i < data.cells.length; i++) {
        var cell = data.cells[i];
        if (L.badge(cell).kind !== item[0] || cell.sets.length !== 1 || !cell.slot) continue;
        if (item[0] === "refused") {
          var now = L.contract(data, cell.method, cell.sets);
          if (now && now.ok) continue;
        }
        if (cell.id === preferred[item[0]]) {
          best = cell;
          break;
        }
        if (!best) best = cell;
      }
      if (best) found.push([item[1], best]);
    });
    for (var i = 0; i < data.cells.length; i++) {
      var cell = data.cells[i];
      if (cell.group === "stage-set" && cell.key.split("&").length > 2 && !cell.shares) {
        found.push(["Several stages", cell]);
        break;
      }
    }
    return h("div", { class: "gx-sec gx-examples" }, [h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Examples" })]), h("div", { class: "gx-examples-row" }, found.map(function (item, index) {
        return h("button", {
          type: "button",
          class: "gx-chipbtn",
          "data-key": "example-" + index,
          title: item[1].method + (item[1].sets.length ? " with " + item[1].sets.join(", ") : ""),
          text: item[0],
          onclick: function () {
            loadCell(item[1]);
          },
        });
      }))]);
  }

  function laneRow(entry, m) {
    var slot = entry.key;
    var current = state.picks[slot] || "";
    var chosen = null;
    if (current) {
      L.choices(data, m.name, slot).forEach(function (choice) {
        if (choice.name === current) chosen = choice;
      });
    }
    var ships = L.shipped(data, m.name, slot);
    var name = chosen ? chosen.label || chosen.name : shippedName(slot, ships);
    var open = state.open === slot;
    var main = h("button", {
      type: "button",
      class: "gx-row-main",
      "data-key": "lane-" + slot,
      "aria-haspopup": "dialog",
      "aria-expanded": open ? "true" : "false",
      "aria-controls": open ? "gx-picker" : null,
      "aria-label": entry.title + ": " + (name || (slot === "post" ? "none" : "default")) + (current ? ", modified" : ""),
      onclick: function () {
        var next = open ? "" : slot;
        closePopovers();
        state.focusKey = next ? "lane-search" : "lane-" + slot;
        set({ open: next, laneQuery: "", pickerActive: 0 });
      },
    }, [
      h("span", { class: "gx-node", "aria-hidden": "true" }),
      h("span", { class: "gx-row-slot", text: entry.title }),
      name
        ? h("span", { class: "gx-row-opt", title: name, text: name })
        : h("span", { class: "gx-row-opt is-default", text: slot === "post" ? "None" : "Default" }),
      h("span", { class: "gx-row-caret", "aria-hidden": "true" }, [icon("chevronDown", 14)]),
    ]);
    var reset = current
      ? h("button", {
          type: "button",
          class: "gx-iconbtn is-row",
          "data-key": "lane-reset-" + slot,
          "aria-label": "Reset " + entry.title + " to the default",
          title: "Reset to default",
          onclick: function () {
            pick(slot, "");
          },
        }, [icon("reset", 14)])
      : null;
    return h("li", { class: "gx-row" + (current ? " is-mod" : "") + (open ? " is-open" : "") + (state.flash === slot ? " is-flash" : "") }, [main, reset]);
  }

  /* The slots of a method, as a pipeline from the observation to the grasps, then the
   * training recipe around the model. A slot the method does not have is not drawn. */
  function pipeline(m) {
    var all = data.lanes.filter(function (entry) {
      return m.slots.indexOf(entry.key) >= 0;
    });
    var own = all.filter(function (entry) {
      return !entry.recipe;
    });
    var around = all.filter(function (entry) {
      return entry.recipe;
    });
    var changed = Object.keys(state.picks).length;
    return h("div", { class: "gx-sec" }, [
      h("div", { class: "gx-sec-head" }, [
        h("span", { class: "gx-label", text: "Pipeline" }),
        changed ? h("span", { class: "gx-sec-count", text: changed + " modified" }) : null,
      ]),
      m.seed_based ? null : h("p", { class: "gx-family" }, [m.name + " is a detector family of its own (", h("code", { text: m.detector }), ")."]),
      h("div", { class: "gx-pipe" }, [
        h("div", { class: "gx-pipe-cap is-start" }, [h("span", { class: "gx-pipe-mark", "aria-hidden": "true" }), m.seed_based ? "Depth cloud" : "RGB-D image"]),
        h("ol", { class: "gx-rows", "aria-label": "Stages" }, own.map(function (entry) {
          return laneRow(entry, m);
        })),
        h("div", { class: "gx-pipe-cap is-end" }, [h("span", { class: "gx-pipe-mark", "aria-hidden": "true" }), "Grasps"]),
      ]),
      around.length ? h("div", { class: "gx-sec-head is-recipe" }, [h("span", { class: "gx-label", text: "Training recipe" })]) : null,
      around.length
        ? h("ol", { class: "gx-rows is-recipe", "aria-label": "Training recipe" }, around.map(function (entry) {
            return laneRow(entry, m);
          }))
        : null,
    ]);
  }

  /* The options of the open slot, default first, as the picker lists them. */
  function pickerItems(entry, m) {
    var slot = entry.key;
    var ships = L.shipped(data, m.name, slot);
    var choices = L.choices(data, m.name, slot);
    var items = [];
    var name = shippedName(slot, ships);
    var shippedOption = L.option(data, slot, ships.option);
    items.push({
      key: "chip-" + slot + "-",
      name: name || (slot === "post" ? "None" : "Default"),
      mono: Boolean(name),
      value: "",
      isDefault: true,
      option: shippedOption,
      component: ships.component || (ships.about || ""),
      badge: L.badge(L.lookup(data, m.name, [])),
      words: [name, ships.component, "default"],
    });
    choices.forEach(function (choice) {
      if (choice.name === ships.option) return;
      var option = choice.option || {};
      items.push({
        key: "chip-" + slot + "-" + choice.name,
        name: choice.label || choice.name,
        mono: true,
        value: choice.name,
        option: choice.option,
        component: option.component || choice.component || "",
        reason: choice.reason || "",
        sets: choice.sets,
        badge: L.mark(data, m.name, choice.sets),
        words: [choice.name, option.component || choice.component, option.summary, (choice.sets || []).join(" ")],
      });
    });
    var query = state.laneQuery.trim().toLowerCase();
    if (!query) return { all: items.length, items: items };
    return {
      all: items.length,
      items: items.filter(function (item) {
        var hay = item.words.join(" ").toLowerCase();
        return query.split(/\s+/).every(function (word) {
          return hay.indexOf(word) >= 0;
        });
      }),
    };
  }

  function picker(entry, m) {
    var slot = entry.key;
    var listed = pickerItems(entry, m);
    var items = listed.items;
    var current = state.picks[slot] || "";
    var active = Math.min(Math.max(state.pickerActive, 0), Math.max(items.length - 1, 0));
    var searchable = listed.all > 7;
    function move(list, to) {
      var options = list.querySelectorAll(".gx-opt");
      if (!options.length) return;
      to = Math.min(Math.max(to, 0), options.length - 1);
      for (var i = 0; i < options.length; i++) options[i].classList.toggle("is-active", i === to);
      state.pickerActive = to;
      var owner = root.querySelector('[data-key="lane-search"]') || list;
      owner.setAttribute("aria-activedescendant", options[to].id);
      options[to].scrollIntoView({ block: "nearest" });
    }
    function keys(event) {
      var list = document.getElementById("gx-picker-list");
      if (!list) return;
      var step = { ArrowDown: 1, ArrowUp: -1, PageDown: 6, PageUp: -6 }[event.key];
      if (step !== undefined) {
        event.preventDefault();
        move(list, state.pickerActive + step);
      } else if (event.key === "Home" && event.currentTarget === list) {
        event.preventDefault();
        move(list, 0);
      } else if (event.key === "End" && event.currentTarget === list) {
        event.preventDefault();
        move(list, items.length - 1);
      } else if (event.key === "Enter") {
        event.preventDefault();
        var item = items[state.pickerActive];
        if (item) pick(slot, item.value);
      }
    }
    var options = items.map(function (item, index) {
      var selected = item.value === current;
      var tags = [];
      if (item.isDefault) tags.push(h("span", { class: "gx-tag", text: "Default" }));
      if (item.option && item.option.extras && item.option.extras.length) tags.push(h("span", { class: "gx-tag", title: 'pip install "grasppanda[' + item.option.extras.join(",") + ']"', text: "extra" }));
      if (item.option && item.option.weights) tags.push(h("span", { class: "gx-tag", title: item.option.weights, text: "weights" }));
      var s = status(item.badge);
      var title = (item.option && item.option.summary ? plain(item.option.summary) + "\n" : "") + s.label + (item.isDefault ? " (" + m.name + " as shipped)" : " (this change alone, under " + m.name + ")") + (item.reason ? "\nNo cell of the grid: " + plain(item.reason) : "");
      return h("li", {
        class: "gx-opt" + (index === active ? " is-active" : "") + (selected ? " is-selected" : "") + (slot === "loss" ? " is-long" : ""),
        id: "gx-opt-" + index,
        role: "option",
        "aria-selected": selected ? "true" : "false",
        "data-key": item.key,
        title: title,
        onclick: function () {
          pick(slot, item.value);
        },
        onmousemove: function () {
          if (state.pickerActive !== index) {
            var list = document.getElementById("gx-picker-list");
            if (list) move(list, index);
          }
        },
      }, [
        dot(item.badge),
        h("span", { class: "gx-opt-name" + (item.mono ? "" : " is-plain"), text: item.name }),
        tags.length ? h("span", { class: "gx-opt-tags" }, tags) : null,
        h("span", { class: "gx-opt-comp", text: item.isDefault && !item.mono ? item.component : item.component }),
        selected ? h("span", { class: "gx-opt-check" }, [icon("check", 14)]) : null,
      ]);
    });
    var list = h("ul", {
      class: "gx-picker-list",
      id: "gx-picker-list",
      role: "listbox",
      "aria-label": entry.title,
      tabindex: searchable ? "-1" : "0",
      "data-key": searchable ? null : "lane-search",
      "aria-activedescendant": !searchable && items.length ? "gx-opt-" + active : null,
      onkeydown: searchable ? null : keys,
    }, options);
    var empty = items.length
      ? null
      : h("div", { class: "gx-picker-empty" }, [
          h("span", { text: "No option matches “" + state.laneQuery.trim() + "”." }),
          h("button", {
            type: "button",
            class: "gx-ghost is-small",
            "data-key": "lane-clear",
            text: "Clear search",
            onclick: function () {
              state.focusKey = "lane-search";
              set({ laneQuery: "", pickerActive: 0 });
            },
          }),
        ]);
    var page = entry.pages[0];
    var legend = [["ok", "", "Runs"], ["empty", "", "No grasps"], ["unstable", "", "Unstable"], ["refused", "", "Refused"], ["fails", "", "Fails"], ["ok", " is-contract", "Compose only"], ["none", "", "No answer"]];
    return h("div", {
      class: "gx-pop gx-picker",
      id: "gx-picker",
      role: "dialog",
      "aria-label": "Choose the " + entry.title.toLowerCase(),
      "data-anchor": "lane-" + slot,
      hidden: true,
      onfocusout: function (event) {
        var next = event.relatedTarget;
        var pop = document.getElementById("gx-picker");
        if (next && pop && !pop.contains(next) && state.open === slot && !(next.getAttribute && next.getAttribute("data-key") === "lane-" + slot)) {
          window.setTimeout(function () {
            if (state.open === slot && document.activeElement === next) set({ open: "", laneQuery: "" });
          }, 0);
        }
      },
    }, [
      h("div", { class: "gx-picker-head" }, [
        h("div", { class: "gx-picker-text" }, [
          h("div", { class: "gx-pop-title", text: entry.title }),
          h("p", { class: "gx-picker-task", text: task(entry) }),
        ]),
        page ? external(slotHref(page.stem), "Docs") : null,
      ]),
      searchable
        ? h("label", { class: "gx-search is-picker" }, [
            icon("search", 16),
            h("input", {
              type: "text",
              class: "gx-search-input",
              "data-key": "lane-search",
              role: "combobox",
              "aria-expanded": "true",
              "aria-controls": "gx-picker-list",
              "aria-autocomplete": "list",
              "aria-activedescendant": items.length ? "gx-opt-" + active : null,
              "aria-label": "Search the options of " + entry.title,
              placeholder: "Search " + listed.all + " options",
              autocomplete: "off",
              spellcheck: "false",
              value: state.laneQuery,
              oninput: function (event) {
                set({ laneQuery: event.target.value, pickerActive: 0 });
              },
              onkeydown: keys,
            }),
          ])
        : null,
      list,
      empty,
      h("div", { class: "gx-picker-foot", "aria-hidden": "true" }, legend.map(function (item) {
        return h("span", {}, [h("span", { class: "gx-dot s-" + item[0] + item[1] }), item[2]]);
      })),
    ]);
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
    closePopovers();
    set({ view: "build", notice: [], picks: L.normalise(data, methodName, picks) }, true);
    var stacked = window.matchMedia && window.matchMedia("(max-width: 959px)").matches;
    var target = stacked ? document.getElementById("gx-result") : root;
    if (target && target.getBoundingClientRect().top < 0 && target.scrollIntoView) target.scrollIntoView({ block: "start" });
    else if (stacked && target && target.scrollIntoView) target.scrollIntoView({ block: "start" });
    var title = document.getElementById("gx-title");
    if (title) title.focus({ preventScroll: true });
  }

  function same(a, b) {
    return a.length === b.length && a.every(function (item, index) {
      return item === b[index];
    });
  }

  /* The alerts of a record: why it was refused, where it failed, what was unstable, what
   * compose adjusted. */
  function recordAlerts(cell, methodName) {
    var out = [];
    /* What compose answers now, where the page holds it: a record keeps the message of
     * the day it was measured, and the lists of alternatives in a fix grow with the code. */
    var now = cell.sets.length ? L.contract(data, methodName, cell.sets) : null;
    if (cell.refusal) {
      if (now && !now.ok) {
        out.push(alertBox("refused", "ban", "Why compose refuses it", [
          messageList(now.messages),
          same(now.messages, cell.refusal) ? null : h("p", { class: "gx-alert-fine", text: "Today's wording; the record was refused by an earlier version with other words." }),
        ]));
      } else {
        out.push(alertBox("refused", "ban", "Why compose refused it", [messageList(cell.refusal)]));
        if (now && now.ok) out.push(alertBox("none", "info", "compose accepts this today", [h("p", { text: "The record is older than the code of this page and has not been probed again." })]));
      }
    } else if (now && !now.ok) {
      out.push(alertBox("unstable", "alert", "Today's compose refuses this record", [h("p", { text: "It was measured with an earlier version. compose now says:" }), messageList(now.messages)]));
    }
    if (cell.failure) {
      var f = cell.failure;
      out.push(alertBox("fails", "alertCircle", "Failed at " + (f.phase || "an unknown phase"), [
        h("p", { class: "gx-alert-meta", text: [f.when, f.component].filter(Boolean).join(" · ") }),
        h("p", { class: "gx-alert-error" }, [h("code", { text: f.error || "Error" }), " ", rich(f.message || "")]),
      ]));
    }
    if (cell.unstable) out.push(alertBox("unstable", "alert", "What was unstable", [rich(cell.unstable, "p")]));
    var adjusted = [];
    if (cell.adapter && cell.verdict !== "refused") {
      adjusted.push(h("p", {}, ["Wrapped the encoder in ", h("code", { text: cell.adapter }), "; the bare swap breaks the contract between encoder and proposal."]));
    }
    if (cell.fitted && cell.fitted.length) {
      adjusted.push(h("p", {}, ["Set "].concat(cell.fitted.map(function (item, index) {
        return h("span", {}, [index ? ", " : "", h("code", { text: item })]);
      }), [" to fit the method."])));
    }
    if (adjusted.length) out.push(alertBox("none", "info", cell.adapter && cell.verdict !== "refused" ? "Adapter inserted by compose" : "Fitted by compose", adjusted));
    return out;
  }

  /* The four levels of a record (or of a contract answer) as a tracker. */
  function levelTracker(cell, checked) {
    var steps = data.levels.map(function (level, index) {
      var said;
      var contract = false;
      if (cell) {
        said = L.levelState(data, cell, level.key);
        if (cell.verdict === "refused" && index > 0) said = "not reached";
      } else if (checked) {
        contract = index === 0;
        said = index === 0 ? (checked.ok ? "yes" : "no") : checked.ok ? "not probed" : "not reached";
      } else {
        said = "not probed";
      }
      return { key: level.key, said: said, contract: contract, meaning: level.meaning };
    });
    var reached = 0;
    while (reached < steps.length && steps[reached].said === "yes" && !steps[reached].contract) reached += 1;
    var whys = {};
    var loose = [];
    (cell && cell.why && cell.verdict !== "refused" ? cell.why : []).forEach(function (line) {
      var at = line.indexOf(": ");
      var head = at > 0 ? line.slice(0, at) : "";
      var known = data.levels.some(function (level) {
        return level.key === head;
      });
      if (known) whys[head] = (whys[head] ? whys[head] + "; " : "") + line.slice(at + 2);
      else loose.push(line);
    });
    var track = h("ol", { class: "gx-levels", style: "--gx-reached:" + reached, "aria-label": "Levels" }, steps.map(function (step, index) {
      var tone = step.said === "yes" ? "ok" : step.said === "no" ? "refused" : "none";
      var glyph = step.said === "yes" ? "check" : step.said === "no" ? "x" : null;
      var undecided = tone === "none";
      return h("li", { class: "gx-level s-" + tone + (step.contract ? " is-contract" : "") + (undecided ? " is-undecided" : "") + (index < reached ? " is-reached" : "") }, [
        h("span", { class: "gx-level-mark", "aria-hidden": "true" }, [glyph ? icon(glyph, 12) : null]),
        h("span", { class: "gx-level-name", title: plain(step.meaning) }, [capital(step.key)]),
        h("span", { class: "gx-level-state", text: undecided ? step.said : step.said === "yes" ? (step.contract ? "compose says yes" : "yes") : step.contract ? "compose says no" : "no" }),
      ]);
    }));
    var lines = steps.filter(function (step) {
      return whys[step.key];
    }).map(function (step) {
      return h("li", {}, [h("span", { class: "gx-why-level", text: capital(step.key) }), rich(whys[step.key], "span", { class: "gx-why-text", title: plain(whys[step.key]) })]);
    }).concat(loose.map(function (line) {
      return h("li", {}, [rich(line, "span", { class: "gx-why-text", title: plain(line) })]);
    }));
    return h("div", { class: "gx-block" }, [
      track,
      lines.length ? h("ul", { class: "gx-whys" }, lines) : null,
    ]);
  }

  function statTiles(cell) {
    if (!cell || cell.parameters === undefined) return null;
    var tiles = [];
    tiles.push(["Parameters", [compact(cell.parameters)], num(cell.parameters), ""]);
    if (cell.loss) {
      var change = (cell.loss[1] - cell.loss[0]) / Math.abs(cell.loss[0] || 1);
      tiles.push(["Total loss", [short(cell.loss[0]), icon("arrowRight", 14), short(cell.loss[1])], percent(change), change > 0 ? "is-bad" : ""]);
    }
    if (cell.grasps && cell.grasps.length) {
      tiles.push(["Grasps per frame", [cell.grasps.map(num).join(" · ")], cell.grasps.length + " frames", ""]);
    }
    tiles.push(["Probe", [data.probe.iterations + " it"], "on " + data.probe.frames + " frames", ""]);
    return h("div", { class: "gx-stats" }, tiles.map(function (tile) {
      return h("div", { class: "gx-stat" }, [
        h("div", { class: "gx-stat-label", text: tile[0] }),
        h("div", { class: "gx-stat-value" }, tile[1]),
        tile[2] ? h("div", { class: "gx-stat-sub " + tile[3], text: tile[2] }) : null,
      ]);
    }));
  }

  function lossFalls(cell) {
    if (!cell || !cell.fall) {
      if (cell && cell.levels && cell.levels.indexOf("C") >= 0) {
        return h("p", { class: "gx-muted gx-small", text: "Loss terms not checked: a record of the first judge." });
      }
      return null;
    }
    var flags = cell.flags || {};
    var rows = Object.keys(cell.fall).map(function (term) {
      var value = cell.fall[term];
      var marks = (flags["loss/" + term] || []).map(function (flag) {
        return h("span", { class: "gx-tag is-flag" + (flag === "rising" ? " s-refused" : ""), text: flag.replace(/_/g, " ") });
      });
      var rising = value !== null && (value < 0 || (flags["loss/" + term] || []).indexOf("rising") >= 0);
      return h("li", { class: "gx-fall" + (rising ? " is-rising" : "") }, [
        h("span", { class: "gx-fall-name" }, [h("code", { text: term })].concat(marks)),
        value === null
          ? h("span", { class: "gx-fall-none", text: "never active" })
          : h("span", { class: "gx-fall-track", "aria-hidden": "true" }, [h("span", { class: "gx-fall-bar", style: "width:" + Math.max(2, Math.min(Math.abs(value), 1) * 100) + "%" })]),
        h("span", { class: "gx-fall-value", text: value === null ? "" : percent(-value) }),
      ]);
    });
    return h("div", { class: "gx-block" }, [
      h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Loss fall by term" }), infoButton("The fall of a term: (first − mean of the last quarter) / |first|, over its active iterations. A falling loss shows that training works, not how good the result is.", "How the fall is measured")]),
      h("ul", { class: "gx-falls" }, rows),
    ]);
  }

  /* Each change alone, and pairs of changes: what is known of the parts of a selection
   * that has no answer of its own. */
  function partsBlock(result, liveOn) {
    var checked = result.contract;
    if (!(result.parts.length > 1 || (result.parts.length === 1 && !checked && !liveOn))) return null;
    function row(sets, badge) {
      return h("li", {}, [dot(badge), h("code", { text: sets.join(" ") }), h("span", { class: "gx-parts-state", text: status(badge).label })]);
    }
    return h("div", { class: "gx-block" }, [
      h("div", { class: "gx-sec-head" }, [
        h("span", { class: "gx-label", text: "Each change alone" }),
        infoButton("None of this decides the whole selection: a stage refused alone may work with the stage it depends on, and changes that fit in pairs may not fit all at once.", "What the parts say"),
      ]),
      h("ul", { class: "gx-parts" }, result.parts.map(function (part) {
        return row(part.sets, part.badge);
      })),
      result.pairs.length ? h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Pairs" })]) : null,
      result.pairs.length ? h("ul", { class: "gx-parts" }, result.pairs.map(function (pair) {
        return row(pair.sets, pair.badge);
      })) : null,
    ]);
  }

  function optionDetails(part) {
    var option = part.option;
    var slotLane = L.lane(data, part.slot);
    if (!option) {
      return h("div", { class: "gx-detail" }, [
        h("div", { class: "gx-detail-head" }, [h("span", { class: "gx-muted", text: slotLane.title }), h("code", { text: part.label || part.name })]),
        keyValues(part.sets.map(function (item) {
          var name = item.slice(item.indexOf("=") + 1);
          var loss = L.option(data, part.slot, name);
          return [item.slice(0, item.indexOf("=")), [h("code", { text: name }), loss ? " " + plain(loss.summary) : ""]];
        }).concat(slotLane.pages.length ? [["Docs", [external(slotHref(slotLane.pages[0].stem), plain(slotLane.pages[0].title))]]] : [])),
      ]);
    }
    var page = option.page || (slotLane.pages[0] || {}).stem;
    var classes = L.classText(data, option);
    return h("div", { class: "gx-detail" }, [
      h("div", { class: "gx-detail-head" }, [h("span", { class: "gx-muted", text: slotLane.title }), h("code", { text: option.name })]),
      rich(option.summary, "p", { class: "gx-detail-sum" }),
      keyValues([
        ["Component", [h("code", { text: option.component }), option.preset ? " through its preset" : " with its defaults"]],
        option.source ? ["Source", [option.source]] : null,
        ["Needs", option.extras.length ? [h("code", { text: 'pip install "grasppanda[' + option.extras.join(",") + ']"' })] : ["The core install"]],
        slotLane.recipe ? null : ["Weights", option.weights ? ["Released, ", h("code", { text: option.weights })] : ["Random initialisation"]],
        classes ? ["Class", [classes, " ", infoButton("Class: " + data.caveat + ".", "What a class counts")]] : null,
        ["Config", [h("code", { text: slotLane.config })]],
        page ? ["Docs", [external(slotHref(page), "Slot page")]] : null,
      ]),
    ]);
  }

  function yamlFold(result) {
    if (liveBase && live.status === "done" && live.report && live.report.accepted && result.sets.length) {
      return fold("yaml", "Composed config", [codeBox(live.report.config, "yaml", "the composed config")], "this installation");
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
      ? segmented(parts.map(function (part) {
          return [part.slot, part.slot, "yaml-" + part.slot];
        }), active.slot, function (slot) {
          state.focusKey = "yaml-" + slot;
          set({ yamlTab: slot });
        }, "Fragment of", "is-small")
      : null;
    return fold("yaml", "Preset YAML", [bar, codeBox(text, "yaml", "the preset fragment")], parts.length > 1 ? parts.length + " fragments" : active.slot);
  }

  function recordFold(result) {
    var cell = result.cell;
    var counts = data.counts;
    var coverage = num(counts.grid - counts.grid_without_record) + " of " + num(counts.grid) + " grid combinations have a record";
    if (!cell) {
      return fold("record", "Record", [
        keyValues([
          ["Record", ["None for this selection"]],
          ["Coverage", [coverage]],
          ["Method", [h("code", { text: result.method.config })]],
        ]),
        h("p", { class: "gx-fold-foot" }, [external(docHref("advanced_guides/measured_combinations"), "Measured combinations")]),
      ], "none");
    }
    var rows = [
      ["Record", [h("code", { text: cell.id })]],
      cell.run ? ["Run", [h("code", { text: cell.run })]] : null,
      cell.judge ? ["Judge", ["Judge " + cell.judge]] : null,
      cell.shares ? ["Probe", ["Shared with ", h("code", { text: cell.shares }), ": the same config"]] : null,
      cell.post && cell.post.grasps
        ? ["After " + cell.post.steps.join(", "), [cell.post.grasps.map(function (frame) {
            return frame.map(num).join(" → ");
          }).join(" · ") + " grasps"]]
        : null,
      cell.unchanged && cell.unchanged.length ? ["Unchanged parts", [cell.unchanged.join(", ")]] : null,
      cell.probe_opts ? ["Probe options", [plain(data.scale_statistic)]] : null,
      ["Method", [h("code", { text: result.method.config })]],
    ];
    return fold("record", "Record", [
      keyValues(rows),
      h("p", { class: "gx-fold-foot" }, [external(docHref("advanced_guides/measured_combinations"), "Measured combinations")]),
    ], day(cell.run));
  }

  /* The badge at the top of the answer. A record decides it; without one, the contract
   * check does: the one this installation runs (live mode) or the one stored in the page. */
  function shownBadge(result) {
    if (result.cell || !result.sets.length || !liveBase) return result.badge;
    if (live.status === "done" && live.report) {
      return L.contractBadge({ ok: live.report.accepted, adapter: live.report.adapter || "" });
    }
    if (live.status === "error") return { kind: "fails", symbol: "", label: "no answer from the server" };
    return { kind: "none", symbol: "", label: "composing", loading: true };
  }

  function summaryLine(result, liveOn) {
    var cell = result.cell;
    var it = data.probe.iterations;
    var fr = data.probe.frames;
    var text;
    var full;
    if (cell) {
      text = {
        runs: "Built, trained " + it + " iterations on " + fr + " real frames, and predicted grasps.",
        "runs-empty": "Built, trained and asked to predict, but no probed frame got a grasp.",
        "runs-unstable": "Built and trained, but a loss, a weight, a seed or a gradient went wrong.",
        refused: "The config check refused it, so nothing was built.",
        fails: "It passed the config check, then raised an error.",
      }[cell.verdict] || capital(L.badge(cell).label) + ".";
      full = data.verdicts[cell.verdict] || "";
    } else if (liveOn) {
      if (live.status === "done" && live.report) {
        text = live.report.accepted ? "This installation's compose accepts it; whether it trains is what the probe shows." : "This installation's compose refuses it and writes no file.";
      } else if (live.status === "error") {
        text = "The server gave no answer.";
      } else {
        text = "Composing it with this installation…";
      }
      full = "The contract check runs now, on this machine. It is no probe: nothing is trained here.";
    } else if (result.contract) {
      text = result.contract.ok ? "compose accepts it; nobody has probed whether it trains." : "compose refuses it, so nothing can be built.";
      full = "The answer of `grasppanda compose`, stored when this page was generated, for one change or two changes of two slots. It is the check of the contract rules and of the config, not a probe: “accepted” says that the contracts fit, not that it trains.";
    } else {
      text = "Nobody has probed this combination, so this page does not guess.";
      full = "The page stores answers of compose for one change and for two changes of two slots; a stage together with an option of the training recipe is not among them. The commands below find out: compose answers at once, the probe trains " + it + " iterations on " + fr + " frames.";
    }
    return h("p", { class: "gx-res-sum" }, [rich(text, "span"), full ? infoButton(full, "What this means") : null]);
  }

  function reproduce(result, liveOn) {
    var refused = !result.probe || (liveOn && live.status === "done" && live.report && !live.report.accepted);
    var items = [];
    if (result.compose) items.push(["compose", "Compose", "repro-compose"]);
    if (!refused) items.push(["probe", "Probe", "repro-probe"]);
    if (!items.length) return null;
    var chosen = items.some(function (item) {
      return item[0] === state.repro;
    }) ? state.repro : items[0][0];
    var text = chosen === "compose" ? result.compose : result.probe;
    var what = chosen === "compose" ? "the compose command" : "the probe command";
    return h("div", { class: "gx-block gx-repro" }, [
      h("div", { class: "gx-sec-head" }, [
        h("span", { class: "gx-label", text: "Reproduce" }),
        items.length > 1
          ? segmented(items, chosen, function (value) {
              state.focusKey = "repro-" + value;
              set({ repro: value });
            }, "Command", "is-small")
          : h("span", { class: "gx-sec-count", text: items[0][1] }),
      ]),
      terminal(text, chosen, what),
      refused && result.compose ? h("p", { class: "gx-muted gx-small", text: "Nothing to probe: compose writes no file when it refuses." }) : null,
    ]);
  }

  function changeChips(result) {
    if (!result.sets.length) return h("p", { class: "gx-changes is-none", text: "As shipped" });
    return h("ul", { class: "gx-changes", "aria-label": "Changes" }, result.sets.map(function (item) {
      var at = item.indexOf("=");
      return h("li", { class: "gx-change" }, [h("code", { text: item.slice(0, at) }), icon("arrowRight", 12), h("code", { class: "is-to", text: item.slice(at + 1) })]);
    }));
  }

  function resultCard(result, badge) {
    var cell = result.cell;
    var liveOn = Boolean(liveBase && result.sets.length);
    var checked = result.contract;
    var meta = cell
      ? "Record of " + day(cell.run)
      : liveOn
        ? "Checked by this installation"
        : checked
          ? "Contract check, no probe"
          : "No record";
    var top = h("div", { class: "gx-res-top" }, [
      badge.loading
        ? h("span", { class: "gx-pill s-none is-contract" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("span", { text: "Checking…" })])
        : pill(badge, cell),
      cell && cell.adapter && cell.verdict !== "refused" ? h("span", { class: "gx-tag", title: "compose inserted the adapter " + cell.adapter, text: "Adapter" }) : null,
      h("span", { class: "gx-res-meta", text: meta }),
    ]);
    var titleRow = h("div", { class: "gx-res-titlerow" }, [
      h("h2", { class: "gx-res-title", id: "gx-title", tabindex: "-1", text: result.method.name }),
      liveBase
        ? h("button", {
            type: "button",
            class: "gx-btn is-small",
            "data-key": "go-run",
            onclick: function () {
              runForm.target = result.sets.length ? "selection" : "method";
              state.focusKey = "tab-run";
              set({ view: "run" }, true);
            },
          }, [icon("play", 14), h("span", { text: result.sets.length ? "Run this selection" : "Run on a scene" })])
        : null,
    ]);
    var children = [top, titleRow, changeChips(result), summaryLine(result, liveOn)];
    if (cell) {
      children.push(levelTracker(cell, null));
      children.push(statTiles(cell));
      children.push(lossFalls(cell));
      recordAlerts(cell, result.method.name).forEach(function (node) {
        children.push(node);
      });
      children.push(liveBlock(result));
    } else if (liveOn) {
      result.parts.forEach(function (part) {
        if (part.reason) children.push(alertBox("none", "info", "No grid cell for " + part.name, [rich(part.reason, "p")]));
      });
      children.push(liveBlock(result));
      children.push(partsBlock(result, liveOn));
    } else {
      if (checked) {
        children.push(levelTracker(null, checked));
        if (!checked.ok) children.push(alertBox("refused", "ban", "Why compose refuses it", [messageList(checked.messages)]));
        else if (checked.adapter) children.push(alertBox("none", "info", "Adapter inserted by compose", [h("p", {}, ["compose wraps the scene encoder in ", h("code", { text: checked.adapter }), "."])]));
      }
      result.parts.forEach(function (part) {
        if (part.reason) children.push(alertBox("none", "info", "No grid cell for " + part.name, [rich(part.reason, "p")]));
      });
      children.push(partsBlock(result, liveOn));
      if (!checked) {
        children.push(h("div", { class: "gx-hint" }, [
          icon("terminal", 16),
          h("p", {}, [h("span", { text: "Get an answer in seconds: " }), h("code", { text: "grasppanda explore" }), h("span", { text: " checks any selection with the real compose." })]),
          copyButton("grasppanda explore", "copy-explore", "grasppanda explore", { cls: "is-ghost" }),
        ]));
      }
    }
    children.push(reproduce(result, liveOn));
    var folds = [];
    if (result.parts.length) folds.push(fold("cards", "Option details", result.parts.map(optionDetails), result.parts.length > 1 ? result.parts.length + " options" : result.parts[0].name));
    folds.push(yamlFold(result));
    folds.push(recordFold(result));
    children.push(h("div", { class: "gx-folds" }, folds));
    return h("section", { class: "gx-card gx-result", id: "gx-result", "aria-live": "polite", "aria-label": "Result" }, children);
  }

  function buildView() {
    var m = L.method(data, state.method);
    var result = L.answer(data, state.method, state.picks);
    ensureLive(result);
    var badge = shownBadge(result);
    var changed = Object.keys(result.picks).length;
    var share = h("button", {
      type: "button",
      class: "gx-ghost",
      "data-key": "share",
      "data-icon": "link",
      onclick: function () {
        copyText(String(window.location.href), share);
      },
    }, [icon("link", 14), h("span", { class: "gx-copy-label", text: "Copy link" })]);
    var openEntry = state.open ? L.lane(data, state.open) : null;
    var s = status(badge, result.cell);
    return h("div", { class: "gx-build" }, [
      h("section", { class: "gx-config", "aria-label": "Configuration" }, [
        h("div", { class: "gx-sec" }, [
          h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Method" })]),
          methodTiles(data.methods, state.method, function (item) {
            return "method-" + item.name;
          }, function (name) {
            closePopovers();
            state.focusKey = "method-" + name;
            set({ method: name, picks: {}, laneQuery: "", notice: [] }, true);
          }),
        ]),
        examples(),
        pipeline(m),
        h("div", { class: "gx-config-actions" }, [
          h("button", {
            type: "button",
            class: "gx-ghost",
            "data-key": "reset",
            disabled: changed ? null : true,
            onclick: function () {
              closePopovers();
              set({ picks: {}, notice: [] }, true);
            },
          }, [icon("reset", 14), h("span", { text: "Reset" })]),
          share,
        ]),
      ]),
      h("div", { class: "gx-side" }, [h("div", { class: "gx-side-inner" }, [resultCard(result, badge)])]),
      openEntry && m.slots.indexOf(state.open) >= 0 ? picker(openEntry, m) : null,
      openEntry && m.slots.indexOf(state.open) >= 0 ? h("div", { class: "gx-scrim", onclick: function () { set({ open: "" }); } }) : null,
      h("a", {
        class: "gx-bar",
        href: "#gx-result",
        onclick: function (event) {
          event.preventDefault();
          var target = document.getElementById("gx-result");
          if (target) target.scrollIntoView({ block: "start", behavior: "smooth" });
        },
      }, [
        h("span", { class: "gx-pill s-" + s.tone + (s.contract ? " is-contract" : "") }, [icon(s.icon, 14), h("span", { text: s.label })]),
        h("span", { class: "gx-bar-text", text: result.method.name + (changed ? " · " + changed + " modified" : " as shipped") }),
        h("span", { class: "gx-bar-more" }, [h("span", { text: "Result" }), icon("chevronDown", 14)]),
      ]),
    ]);
  }

  // -------------------------------------------------------------------- compare view

  var TILE_ICON = { yes: "check", no: "x", refused: "ban", unstable: "minus", "check-refused": "ban", mixed: "minus" };

  function tipFor(m, slot, entry, found, tile) {
    var lines = [h("div", { class: "gx-tip-title", text: m.name + " · " + entry.name })];
    var level = state.filter.level || "runs";
    function statusLine(badge, cell) {
      var s = status(badge, cell);
      return h("div", { class: "gx-tip-status" }, [h("span", { class: "gx-dot s-" + s.tone + (s.contract ? " is-contract" : "") }), h("span", { text: s.label })]);
    }
    if (!found.length) {
      lines.push(statusLine(tile.contract ? L.mark(data, m.name, [slot + "=" + entry.name]) : L.badge(null)));
      var reason = slot === "loss" ? "" : L.notProbed(m, slot, entry.name);
      if (reason) lines.push(h("div", { class: "gx-tip-line", text: "No grid cell: " + plain(reason) }));
      if (tile.contract) lines.push(h("div", { class: "gx-tip-line", text: "The answer of compose; no probe on record." }));
    }
    found.slice(0, 2).forEach(function (cell) {
      lines.push(statusLine(L.badge(cell), cell));
      if (slot === "loss") lines.push(h("div", { class: "gx-tip-mono", text: cell.sets.join(" ") }));
      var said = L.recordTile(data, cell, level);
      lines.push(h("div", { class: "gx-tip-row" }, [h("span", { text: capital(level) }), h("b", { text: said.label })]));
      if (cell.grasps && cell.grasps.length) lines.push(h("div", { class: "gx-tip-row" }, [h("span", { text: "Grasps per frame" }), h("b", { text: cell.grasps.map(num).join(" · ") })]));
      var why = cell.refusal
        ? plain(cell.refusal[0].split(". ")[0]) + "."
        : cell.failure
          ? (cell.failure.error || "") + " at " + (cell.failure.phase || "")
          : (cell.why || [])[0];
      if (why) lines.push(h("div", { class: "gx-tip-line", text: plain(why) }));
    });
    if (found.length > 2) lines.push(h("div", { class: "gx-tip-line", text: "and " + (found.length - 2) + " more records" }));
    lines.push(h("div", { class: "gx-tip-hint", text: "Click for actions and commands" }));
    return h("div", {}, lines);
  }

  function cellAnswer(m, slot, entry, cell) {
    var picks = {};
    if (cell) {
      if (cell.group === "loss") picks.loss = L.lossName(cell);
      else cell.sets.forEach(function (item) {
        picks[item.slice(0, item.indexOf("="))] = item.slice(item.indexOf("=") + 1);
      });
    } else if (slot !== "loss") {
      picks[slot] = entry.name;
    }
    return { method: m.name, picks: picks, answer: L.answer(data, m.name, picks) };
  }

  function gridTile(m, slot, entry, found, place, focusable) {
    var level = state.filter.level || "runs";
    var checked = !found.length && slot !== "loss" ? L.contract(data, m.name, [slot + "=" + entry.name]) : null;
    var tile = L.levelTile(data, found, level, checked);
    var key = "cell-" + slot + "-" + entry.name + "-" + m.name;
    var badge = found.length ? L.badgeOf(found) : checked ? L.contractBadge(checked) : L.badge(null);
    var label = m.name + ", " + entry.name + ", " + level + ": " + tile.label + (found.length ? " (" + status(badge, found[0]).label.toLowerCase() + ")" : "");
    var glyph = TILE_ICON[tile.state];
    return h("td", { class: "gx-m-cell" }, [
      h("button", {
        type: "button",
        class: "gx-tile t-" + tile.state + (tile.contract ? " is-contract" : ""),
        "data-key": key,
        "data-row": String(place[0]),
        "data-col": String(place[1]),
        tabindex: focusable ? "0" : "-1",
        "aria-label": label,
        "aria-haspopup": "menu",
        "aria-expanded": state.cellMenu === key ? "true" : "false",
        onmouseenter: function (event) {
          if (!state.cellMenu) showTip(event.currentTarget, tipFor(m, slot, entry, found, tile), true);
        },
        onmouseleave: hideTip,
        onfocus: function (event) {
          state.gridFocus = key;
          if (!state.cellMenu) showTip(event.currentTarget, tipFor(m, slot, entry, found, tile), true);
        },
        onblur: hideTip,
        onclick: function () {
          hideTip();
          var open = state.cellMenu !== key;
          closePopovers();
          state.focusKey = open ? "menu-0" : key;
          set({ cellMenu: open ? key : "" });
        },
      }, [glyph ? icon(glyph, 12) : tile.state === "none" ? h("span", { class: "gx-tile-dot" }) : null]),
    ]);
  }

  /* What a click on a tile offers: open it in Build, copy its commands; a loss probed in
   * several sets lists its records. */
  function cellMenu(m, slot, entry, found, key) {
    var items = [];
    function close(focus) {
      state.focusKey = focus || key;
      set({ cellMenu: "" });
    }
    if (found.length > 1) {
      found.forEach(function (cell, index) {
        items.push(h("button", { type: "button", role: "menuitem", class: "gx-menu-item", "data-key": "menu-" + index, onclick: function () {
          loadCell(cell);
        } }, [dot(L.badge(cell)), h("span", { class: "gx-menu-mono", text: cell.sets.join(" ") }), icon("arrowRight", 14)]));
      });
    } else {
      var target = cellAnswer(m, slot, entry, found[0]);
      items.push(h("button", { type: "button", role: "menuitem", class: "gx-menu-item", "data-key": "menu-0", onclick: function () {
        openBuilder(target.method, target.picks);
      } }, [icon("layers", 14), h("span", { text: "Open in Build" }), icon("arrowRight", 14)]));
      if (target.answer.compose) {
        items.push(h("button", { type: "button", role: "menuitem", class: "gx-menu-item", "data-key": "menu-copy-compose", "data-icon": "copy", onclick: function (event) {
          copyText(target.answer.compose, event.currentTarget);
        } }, [icon("copy", 14), h("span", { class: "gx-copy-label", text: "Copy compose command" })]));
      }
      if (target.answer.probe) {
        items.push(h("button", { type: "button", role: "menuitem", class: "gx-menu-item", "data-key": "menu-copy-probe", "data-icon": "copy", onclick: function (event) {
          copyText(target.answer.probe, event.currentTarget);
        } }, [icon("copy", 14), h("span", { class: "gx-copy-label", text: "Copy probe command" })]));
      }
    }
    return h("div", {
      class: "gx-pop gx-menu",
      role: "menu",
      "aria-label": entry.name + " under " + m.name,
      "data-anchor": key,
      hidden: true,
      onkeydown: function (event) {
        var step = { ArrowDown: 1, ArrowUp: -1 }[event.key];
        if (step === undefined) return;
        event.preventDefault();
        var all = Array.prototype.slice.call(event.currentTarget.querySelectorAll(".gx-menu-item"));
        var at = all.indexOf(document.activeElement);
        all[(at + step + all.length) % all.length].focus();
      },
    }, [
      h("div", { class: "gx-menu-head" }, [h("span", { class: "gx-menu-title", text: entry.name }), h("span", { class: "gx-muted", text: "under " + m.name })]),
      found.length > 1 ? h("div", { class: "gx-menu-note", text: "Part of " + found.length + " records. Open one:" }) : null,
      h("div", { class: "gx-menu-items" }, items),
      h("button", { type: "button", class: "gx-sr", "data-key": "menu-close", text: "Close", onclick: function () {
        close();
      } }),
    ]);
  }

  function moveInGrid(event) {
    var step = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowDown: [1, 0], ArrowUp: [-1, 0] }[event.key];
    var from = event.target && event.target.closest ? event.target.closest(".gx-tile") : null;
    if (!step || !from) return;
    var row = Number(from.getAttribute("data-row")) + step[0];
    var col = Number(from.getAttribute("data-col")) + step[1];
    var to = event.currentTarget.querySelector('.gx-tile[data-row="' + row + '"][data-col="' + col + '"]');
    if (!to) return;
    event.preventDefault();
    from.setAttribute("tabindex", "-1");
    to.setAttribute("tabindex", "0");
    to.focus();
  }

  /* The slot the grid opens on: the one with the most options. */
  function firstGridSlot() {
    var best = data.lanes[0].key;
    data.lanes.forEach(function (entry) {
      if ((data.options[entry.key] || []).length > (data.options[best] || []).length) best = entry.key;
    });
    return best;
  }

  /* The slots in groups: the stages of the seed-based methods, the training recipe, and
   * the slots of a family of its own (HGGD). */
  function slotGroups() {
    var groups = [["Model", []], ["Training recipe", []]];
    var family = {};
    data.lanes.forEach(function (entry) {
      var seed = data.methods.some(function (m) {
        return m.seed_based && m.slots.indexOf(entry.key) >= 0;
      });
      if (entry.recipe) groups[1][1].push(entry);
      else if (seed) groups[0][1].push(entry);
      else {
        var owners = data.methods.filter(function (m) {
          return m.slots.indexOf(entry.key) >= 0;
        }).map(function (m) {
          return m.name;
        }).join(", ");
        if (!family[owners]) {
          family[owners] = [owners, []];
          groups.push(family[owners]);
        }
        family[owners][1].push(entry);
      }
    });
    return groups.filter(function (group) {
      return group[1].length;
    });
  }

  /* A method's name over its column, in at most two lines: broken after its last hyphen
   * when it is long ("Scale-Balanced-" / "Grasp"). */
  function headName(name) {
    var at = name.lastIndexOf("-");
    if (name.length <= 13 || at < 0) return [h("span", { class: "gx-m-line", text: name })];
    return [h("span", { class: "gx-m-line", text: name.slice(0, at + 1) }), h("span", { class: "gx-m-line", text: name.slice(at + 1) })];
  }

  var CLASS_NAMES = { universal: "Universal", most: "Most", some: "Some", specific: "Specific", none: "None", relative: "N of M", undecided: "Undecided" };

  function classPill(option, level) {
    var found = L.classAt(data, option, level);
    if (!found) return h("span", { class: "gx-muted", text: "—" });
    var text = found[0];
    var column = found[1];
    var title = text + " (at " + level + "). Class: " + data.caveat + ".";
    if (column === "undecided") return h("span", { class: "gx-cls is-undecided", title: title, text: capital(text) });
    var shortText = column === "relative" ? (/^(\d+ of \d+)/.exec(text) || [text, text])[1] : CLASS_NAMES[column] || capital(text);
    return h("span", { class: "gx-cls c-" + column, title: title, text: shortText });
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
    var active = filter.classes.length + (filter.extra ? 1 : 0) + (filter.weights ? 1 : 0);
    function patch(change) {
      var next = { q: filter.q, level: level, classes: filter.classes.slice(), extra: filter.extra, weights: filter.weights };
      Object.keys(change).forEach(function (name) {
        next[name] = change[name];
      });
      set({ filter: next, cellMenu: "" });
    }
    var nav = h("nav", { class: "gx-slotnav", "aria-label": "Slots" }, slotGroups().map(function (group) {
      return h("div", { class: "gx-slotnav-group" }, [h("div", { class: "gx-label", text: group[0] })].concat(group[1].map(function (item) {
        var on = item.key === slot;
        return h("button", {
          type: "button",
          class: "gx-slotnav-item",
          "data-key": "slot-" + item.key,
          "aria-current": on ? "true" : null,
          onclick: function () {
            state.filter = { q: "", level: level, classes: [], extra: false, weights: false };
            closePopovers();
            state.focusKey = "slot-" + item.key;
            set({ slot: item.key, gridFocus: "" }, true);
          },
        }, [h("span", { text: item.title }), h("span", { class: "gx-slotnav-count", text: String((data.options[item.key] || []).length) })]);
      })));
    }));
    var withClass = (data.options[slot] || []).some(function (option) {
      return option.classes;
    });
    var levelItems = (data.levels || []).map(function (item) {
      return [item.key, capital(item.key), "level-" + item.key, plain(item.meaning)];
    });
    var levelNow = data.levels.filter(function (item) {
      return item.key === level;
    })[0];
    var docs = entry.pages.length > 1
      ? h("button", {
          type: "button",
          class: "gx-ghost",
          "data-key": "grid-docs",
          "aria-expanded": state.docsMenu ? "true" : "false",
          "aria-controls": "gx-docs",
          onclick: function () {
            var open = !state.docsMenu;
            closePopovers();
            state.focusKey = "grid-docs";
            set({ docsMenu: open });
          },
        }, [icon("book", 14), h("span", { text: "Docs" }), icon("chevronDown", 14)])
      : entry.pages.length
        ? h("a", { class: "gx-ghost", href: slotHref(entry.pages[0].stem), "data-key": "grid-docs" }, [icon("book", 14), h("span", { text: "Docs" })])
        : null;
    var wrappers = grid.rows.filter(function (row) {
      return /^(feature-propagation|furthest-point-subsample|image-painted(-subsampled)?)$/.test(row.option.name);
    }).length;
    var countText = (rows.length === grid.rows.length ? num(grid.rows.length) + " options" : num(rows.length) + " of " + num(grid.rows.length)) +
      (slot === "backbone" && wrappers ? " · " + wrappers + " are adapters" : "");
    var head = h("div", { class: "gx-cmp-head" }, [
      h("div", { class: "gx-cmp-titles" }, [
        h("div", { class: "gx-cmp-titlerow" }, [
          h("h2", { class: "gx-cmp-title", text: entry.title }),
          h("span", { class: "gx-cmp-count", "aria-live": "polite", text: countText }),
          loss ? infoButton("A loss sits on one term of a stage, and most were probed several terms at a time: a tile is the verdict of the sets this loss was part of under the method of its column, not of the loss alone.", "About losses") : null,
        ]),
        h("p", { class: "gx-cmp-task", text: task(entry) }),
      ]),
      docs,
    ]);
    var search = h("label", { class: "gx-search" }, [
      icon("search", 16),
      h("input", {
        type: "search",
        class: "gx-search-input",
        "data-key": "grid-search",
        placeholder: "Search options",
        "aria-label": "Search the options of " + entry.title,
        autocomplete: "off",
        spellcheck: "false",
        value: filter.q,
        oninput: function (event) {
          patch({ q: event.target.value });
        },
      }),
    ]);
    var filtersButton = h("button", {
      type: "button",
      class: "gx-btn" + (active ? " is-on" : ""),
      "data-key": "grid-filters",
      "aria-expanded": state.filters ? "true" : "false",
      "aria-controls": "gx-filters",
      onclick: function () {
        var open = !state.filters;
        closePopovers();
        state.focusKey = "grid-filters";
        set({ filters: open });
      },
    }, [icon("filter", 14), h("span", { text: "Filters" }), active ? h("span", { class: "gx-badge", text: String(active) }) : null]);
    var clear = active || filter.q.trim()
      ? h("button", {
          type: "button",
          class: "gx-ghost",
          "data-key": "grid-clear",
          onclick: function () {
            state.focusKey = "grid-search";
            set({ filter: { q: "", level: level, classes: [], extra: false, weights: false }, filters: false });
          },
        }, [h("span", { text: "Clear" })])
      : null;
    var toolbar = h("div", { class: "gx-toolbar" }, [
      h("div", { class: "gx-toolbar-left" }, [
        segmented(levelItems, level, function (value) {
          state.focusKey = "level-" + value;
          patch({ level: value, classes: [] });
        }, "Level"),
        levelNow ? infoButton(capital(levelNow.key) + ": " + levelNow.meaning + ".", "What " + levelNow.key + " means") : null,
      ]),
      h("div", { class: "gx-toolbar-right" }, [search, filtersButton, clear]),
    ]);
    var legendItems = [["yes", "Yes"], ["no", "No"], ["refused", "Refused"], ["unstable", "Unstable"], ["undecided", "Undecided"]];
    if (loss) legendItems.push(["mixed", "Mixed"]);
    else if (data.contracts) legendItems.push(["accepted", "Compose only"]);
    legendItems.push(["none", "No answer"]);
    var legend = h("ul", { class: "gx-legend", "aria-label": "Legend" }, legendItems.map(function (item) {
      var glyph = TILE_ICON[item[0]];
      return h("li", {}, [
        h("span", { class: "gx-tile is-mini t-" + item[0] + (item[0] === "accepted" ? " is-contract" : ""), "aria-hidden": "true" }, [glyph ? icon(glyph, 10) : item[0] === "none" ? h("span", { class: "gx-tile-dot" }) : null]),
        h("span", { text: item[1] }),
      ]);
    }));
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
        h("th", { scope: "row", class: "gx-m-opt" }, [
          h("div", { class: "gx-m-name", title: plain(option.summary) }, [h("code", { text: option.name })]),
          h("div", { class: "gx-m-sub" }, [
            h("span", { text: option.component }),
            option.extras.length ? h("span", { class: "gx-tag", title: 'pip install "grasppanda[' + option.extras.join(",") + ']"', text: "extra" }) : null,
            option.weights ? h("span", { class: "gx-tag", title: option.weights, text: "weights" }) : null,
          ]),
        ]),
      ].concat(
        grid.columns.map(function (m, c) {
          var key = "cell-" + slot + "-" + option.name + "-" + m.name;
          if (key === state.cellMenu) menu = cellMenu(m, slot, option, row.cells[m.name], key);
          return gridTile(m, slot, option, row.cells[m.name], [r, c], key === focusKey);
        }),
        withClass ? [h("td", { class: "gx-m-class" }, [classPill(option, level)])] : []
      ));
    });
    if (!rows.length) {
      body = [h("tr", {}, [h("td", { class: "gx-m-empty", colspan: String(grid.columns.length + 1 + (withClass ? 1 : 0)) }, [
        h("div", { class: "gx-empty" }, [
          h("span", { class: "gx-empty-icon" }, [icon("search", 20)]),
          h("p", { text: "No option matches" + (filter.q.trim() ? " “" + filter.q.trim() + "”" : " these filters") + "." }),
          h("button", { type: "button", class: "gx-btn is-small", "data-key": "grid-empty-clear", text: "Clear search and filters", onclick: function () {
            state.focusKey = "grid-search";
            set({ filter: { q: "", level: level, classes: [], extra: false, weights: false } });
          } }),
        ]),
      ])])];
    }
    var table = h("table", { class: "gx-matrix", onkeydown: moveInGrid }, [
      h("thead", {}, [
        h("tr", {}, [h("th", { scope: "col", class: "gx-m-opt", text: loss ? "Loss and term" : "Option" })].concat(
          grid.columns.map(function (m) {
            return h("th", { scope: "col", class: "gx-m-method", title: m.name + " · " + m.config }, headName(m.name));
          }),
          withClass ? [h("th", { scope: "col", class: "gx-m-class" }, [h("span", { text: "Class" }), infoButton("The class of an option counts, at the level picked above, the methods under which it reached the level. Class: " + data.caveat + ".", "What a class counts")])] : []
        )),
      ]),
      h("tbody", {}, body),
    ]);
    var filtersPop = null;
    if (state.filters) {
      var classes = L.classesOf(data, slot, level);
      var countOf = function (name) {
        return (data.options[slot] || []).filter(function (option) {
          return L.classColumn(data, option, level) === name;
        }).length;
      };
      var check = function (key, label, on, count, onchange) {
        return h("label", { class: "gx-check" }, [
          h("input", { type: "checkbox", "data-key": key, checked: on ? true : null, onchange: onchange }),
          h("span", { class: "gx-check-box", "aria-hidden": "true" }, [icon("check", 12)]),
          h("span", { class: "gx-check-label", text: label }),
          count !== null ? h("span", { class: "gx-check-count", text: String(count) }) : null,
        ]);
      };
      filtersPop = h("div", { class: "gx-pop gx-filters", id: "gx-filters", role: "dialog", "aria-label": "Filters", "data-anchor": "grid-filters", "data-align": "end", hidden: true }, [
        classes.length ? h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Class at " + level }), infoButton("Class: " + data.caveat + ".", "What a class counts")]) : null,
        classes.length ? h("div", { class: "gx-checks" }, classes.map(function (name) {
          var on = filter.classes.indexOf(name) >= 0;
          return check("class-" + name, CLASS_NAMES[name] || capital(name), on, countOf(name), function () {
            state.focusKey = "class-" + name;
            patch({ classes: on ? filter.classes.filter(function (other) { return other !== name; }) : filter.classes.concat([name]) });
          });
        })) : null,
        h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Option" })]),
        h("div", { class: "gx-checks" }, [
          check("f-extra", "Needs an extra", filter.extra, (data.options[slot] || []).filter(function (o) { return o.extras.length; }).length, function () {
            state.focusKey = "f-extra";
            patch({ extra: !filter.extra });
          }),
          check("f-weights", "Loads released weights", filter.weights, (data.options[slot] || []).filter(function (o) { return o.weights; }).length, function () {
            state.focusKey = "f-weights";
            patch({ weights: !filter.weights });
          }),
        ]),
      ]);
    }
    var docsPop = state.docsMenu && entry.pages.length > 1
      ? h("div", { class: "gx-pop gx-menu gx-docs", id: "gx-docs", role: "menu", "aria-label": "Docs", "data-anchor": "grid-docs", "data-align": "end", hidden: true }, [
          h("div", { class: "gx-menu-items" }, entry.pages.map(function (page, index) {
            return h("a", { role: "menuitem", class: "gx-menu-item", href: slotHref(page.stem), "data-key": "docs-" + index }, [h("span", { text: plain(page.title) }), icon("external", 14)]);
          })),
        ])
      : null;
    return h("div", { class: "gx-compare" }, [
      nav,
      h("div", { class: "gx-cmp-main" }, [
        head,
        toolbar,
        legend,
        h("div", { class: "gx-matrix-wrap", role: "region", "aria-label": entry.title + ": option by method", tabindex: "-1" }, [table]),
      ]),
      filtersPop,
      docsPop,
      menu,
    ]);
  }

  // ------------------------------------------------------------------------ run view

  /* The Run panel: a ported method on one frame, run by `grasppanda explore` with weights
   * its command line named. On the documentation site it is drawn disabled, with the
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
        runLive.error = "The server did not answer (" + error + "). Is `grasppanda explore` still running?";
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
    if (out && window.innerWidth < 960 && out.scrollIntoView) out.scrollIntoView({ block: "start", behavior: "smooth" });
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

  function runStep(n, label, children, id, aside) {
    return h("li", { class: "gx-step", role: "group", "aria-labelledby": id }, [
      h("span", { class: "gx-step-n", "aria-hidden": "true", text: String(n) }),
      h("div", { class: "gx-step-body" }, [h("div", { class: "gx-step-head" }, [h("span", { class: "gx-step-label", id: id, text: label }), aside || null])].concat(children)),
    ]);
  }

  function pickRunMethod(name) {
    if (name === state.method || !L.method(data, name)) return;
    runForm.typing = false;
    runLive.problem = "";
    state.focusKey = "run-method-" + name;
    set({ method: name, picks: {}, notice: [] }, true);
  }

  function methodStep(off) {
    var m = runMethod();
    var children = [
      methodTiles(runMethods(), m.name, function (item) {
        return "run-method-" + item.name;
      }, pickRunMethod, off),
    ];
    var selection = off ? null : runSelection();
    if (selection) {
      var count = selection.sets.length;
      var mine = runForm.target === "selection" && selection.accepted !== false;
      children.push(segmented([
        ["method", "As shipped", "run-target-method"],
        ["selection", "With your " + count + " change" + (count > 1 ? "s" : ""), "run-target-selection", null, selection.accepted === false],
      ], mine ? "selection" : "method", function (value) {
        runForm.target = value;
        state.focusKey = "run-target-" + value;
        render();
      }, "What to run", "is-small"));
      var said = selection.accepted === false
        ? "compose refuses your selection, so there is nothing to run."
        : selection.checking
          ? "compose is checking your selection…"
          : mine
            ? "The weights must fit the composed model: a new stage needs weights trained with it."
            : "";
      children.push(h("p", { class: "gx-step-fine" }, [h("span", { text: "Your changes: " })].concat(
        selection.sets.map(function (item, index) {
          return h("span", {}, [index ? ", " : "", h("code", { text: item })]);
        })
      )));
      if (said) children.push(h("p", { class: "gx-step-fine", text: said }));
    }
    children.push(h("p", { class: "gx-step-fine" }, [h("code", { text: m.config }), h("span", { text: m.needs_rgb ? " · reads depth and colour" : " · reads the point cloud of a depth image" })]));
    if (m.problem) children.push(alertBox("unstable", "alert", "This method cannot run here", [h("p", { text: m.problem })]));
    return runStep(1, "Method", children, "gx-run-l1");
  }

  function weightsStep(off) {
    var setup = runLive.setup;
    if (!off && setup && !setup.weights.length) {
      return runStep(2, "Weights", [
        alertBox("unstable", "alert", "This server reads no weights", [
          h("p", { text: "Stop it (Ctrl+C) and start it again with a folder of weights, or with one file:" }),
          terminal("grasppanda explore --weights-dir <dir>", "run-dir", "the command with a folder"),
          terminal("grasppanda explore --weights gsnet=<file>", "run-file", "the command with one file"),
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
      class: "gx-select-input",
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
    }, [h("option", { value: "", text: off ? "A weights file of your machine" : entries.length ? "Choose a weights file…" : "…" })]);
    if (named.length) select.appendChild(h("optgroup", { label: "Named on the command line" }, named.map(option)));
    if (listed.length) select.appendChild(h("optgroup", { label: "In " + setup.weights_dir }, listed.map(option)));
    if (setup && setup.weights_dir) select.appendChild(h("option", { value: TYPE_PATH, text: "Another file in this folder: type its path…" }));
    select.value = runForm.typing ? TYPE_PATH : value;
    if (select.value !== (runForm.typing ? TYPE_PATH : value)) select.value = "";
    var children = [h("span", { class: "gx-select" }, [select, icon("chevronDown", 16)])];
    if (runForm.typing) {
      children.push(h("input", {
        type: "text",
        class: "gx-input is-mono",
        "data-key": "run-typed",
        "aria-label": "Path of a weights file inside " + setup.weights_dir,
        placeholder: "run/checkpoints/epoch_0010.pth",
        spellcheck: "false",
        value: runForm.typed,
        oninput: function (event) {
          runForm.typed = event.target.value;
        },
      }));
    }
    var fine = [];
    if (!off && value && value === m.suggested && !runForm.typing) fine.push("Chosen by its name for " + m.name + "; the run says whether it fits.");
    if (setup && setup.truncated) fine.push("The folder holds more files than are listed; type the path of another one.");
    if (!off && setup) fine.push("The server reads only the files listed here.");
    if (fine.length) children.push(h("p", { class: "gx-step-fine", text: fine.join(" ") }));
    return runStep(2, "Weights", children, "gx-run-l2");
  }

  /* A file of the visitor's: a drop zone that opens the picker, then the name and size. */
  function dropZone(role, label, accept, hint, off) {
    var chosen = runForm[role];
    var id = "gx-run-file-" + role;
    function take(file) {
      if (!file) return;
      runForm[role] = { file: file, name: file.name, size: file.size };
      runLive.problem = "";
      state.focusKey = "run-file-" + role;
      render();
    }
    var input = h("input", {
      type: "file",
      class: "gx-file-input",
      id: id,
      accept: accept,
      "data-key": "run-file-" + role,
      disabled: off ? true : null,
      onchange: function (event) {
        take(event.target.files && event.target.files[0]);
      },
    });
    return h("div", { class: "gx-field" }, [
      h("span", { class: "gx-field-label", text: label }),
      h("label", {
        for: id,
        class: "gx-drop" + (chosen ? " is-chosen" : ""),
        ondragover: function (event) {
          if (off) return;
          event.preventDefault();
          event.currentTarget.classList.add("is-over");
        },
        ondragleave: function (event) {
          event.currentTarget.classList.remove("is-over");
        },
        ondrop: function (event) {
          if (off) return;
          event.preventDefault();
          take(event.dataTransfer && event.dataTransfer.files && event.dataTransfer.files[0]);
        },
      }, [
        input,
        icon(chosen ? "check" : "upload", 16),
        chosen
          ? h("span", { class: "gx-drop-text" }, [h("code", { text: chosen.name }), h("span", { class: "gx-muted", text: " " + megabytes(chosen.size) })])
          : h("span", { class: "gx-drop-text" }, [h("b", { text: "Drop a file" }), h("span", { class: "gx-muted", text: " or browse · " + hint })]),
        chosen ? h("span", { class: "gx-drop-action", text: "Change" }) : null,
      ]),
      chosen && role !== "depth"
        ? h("button", { type: "button", class: "gx-ghost is-small", "data-key": "run-clear-" + role, text: "Remove", onclick: function () {
            runForm[role] = null;
            render();
          } })
        : null,
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
          class: "gx-input",
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
    var children = [segmented([
      ["example", "Example scene", "run-input-example", null, off],
      ["depth", "Your depth image", "run-input-depth", null, off],
    ], runForm.input, function (value) {
      runForm.input = value;
      runLive.problem = "";
      state.focusKey = "run-input-" + value;
      render();
    }, "Input", "is-small")];
    if (runForm.input !== "depth") {
      children.push(h("p", { class: "gx-step-fine", text: "A synthetic table-top scene that ships with GraspPanda: six primitives, 1280 x 720 depth and colour, no sensor noise." }));
      return runStep(3, "Input", children, "gx-run-l3");
    }
    children.push(dropZone("depth", "Depth image", ".png,.npy,.tif,.tiff", "16-bit PNG in mm, or .npy in m", off));
    children.push(h("div", { class: "gx-field" }, [
      h("span", { class: "gx-field-label" }, ["Intrinsics, in pixels ", infoButton("Of the depth camera at the resolution of the image. Pasting 925,925,640,360 into one box fills all four.", "About the intrinsics")]),
      h("div", { class: "gx-nums" }, numberBoxes("run-k", runForm.intrinsics, ["fx", "fy", "cx", "cy"], off)),
    ]));
    children.push(dropZone("rgb", m.needs_rgb ? "Colour image (" + m.name + " needs it)" : "Colour image (optional)", ".png,.jpg,.jpeg", "registered to the depth image", off));
    var labels = ["x min", "y min", "z min", "x max", "y max", "z max"];
    children.push(h("div", { class: "gx-field" }, [
      h("span", { class: "gx-field-label" }, ["Workspace box (optional) ", infoButton("Metres, camera frame: x right, y down, z forward. Only points inside are kept. Empty: every pixel with depth is used, and the run says what that means for this method.", "About the workspace box")]),
      h("div", { class: "gx-nums is-six" }, numberBoxes("run-ws", runForm.workspace, labels, off)),
    ]));
    return runStep(3, "Input", children, "gx-run-l3");
  }

  function optionsStep(off) {
    return runStep(4, "Options", [
      h("label", { class: "gx-switch" }, [
        h("input", {
          type: "checkbox",
          role: "switch",
          "data-key": "run-collision",
          checked: runForm.collision ? true : null,
          disabled: off ? true : null,
          onchange: function (event) {
            runForm.collision = event.target.checked;
          },
        }),
        h("span", { class: "gx-switch-track", "aria-hidden": "true" }),
        h("span", { class: "gx-switch-text" }, [h("b", { text: "Collision filter" }), h("span", { class: "gx-muted", text: " drops grasps whose gripper hits observed points" })]),
      ]),
    ], "gx-run-l4");
  }

  function runLabel() {
    var m = runMethod();
    return "Run " + m.name + (runSets().length ? " with your changes" : "");
  }

  function goButton(off) {
    var busy = Boolean(runLive.sending || (runLive.job && runLive.job.state === "running"));
    return h("div", { class: "gx-go" }, [
      h("button", {
        type: "button",
        class: "gx-btn-primary",
        "data-key": "run-go",
        disabled: off || busy || (runLive.setup && !runLive.setup.weights.length) ? true : null,
        "aria-busy": busy ? "true" : null,
        onclick: startRun,
      }, [busy ? h("span", { class: "gx-spin is-light", "aria-hidden": "true" }) : icon("play", 16), h("span", { text: busy ? "Running…" : runLabel() })]),
      runLive.problem ? h("p", { class: "gx-run-problem", role: "alert" }, [icon("alertCircle", 14), h("span", { text: runLive.problem })]) : null,
    ]);
  }

  function stageList(stages) {
    return h("ol", { class: "gx-run-stages" }, stages.map(function (stage) {
      var time = stage.seconds === null || stage.seconds === undefined ? "" : seconds(stage.seconds);
      var mark = stage.state === "done" ? icon("check", 12) : stage.state === "failed" ? icon("x", 12) : stage.state === "running" ? h("span", { class: "gx-spin is-tiny" }) : null;
      return h("li", { class: "is-" + stage.state }, [
        h("span", { class: "gx-run-dot", "aria-hidden": "true" }, [mark]),
        h("span", { class: "gx-run-stage", text: capital(stage.name) + (stage.note ? " (" + stage.note + ")" : "") }),
        h("span", { class: "gx-run-time", text: time }),
      ]);
    }));
  }

  function signed(value) {
    return (value < 0 ? "−" : "+") + Math.abs(value).toFixed(3);
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
    return h("div", { class: "gx-run-body" }, [
      alertBox("refused", "x", failure.title, [
        h("pre", { class: "gx-run-message", text: shown }),
        long ? fold("run-message", "The whole message", [h("pre", { class: "gx-run-message is-long", text: failure.message })]) : null,
        hints[failure.kind] && (job || failure.kind !== "weights") ? h("p", { class: "gx-alert-fine", text: hints[failure.kind] }) : null,
      ], "is-role-alert"),
      job ? stageList(job.stages) : null,
    ]);
  }

  function resultView(result) {
    var steps = [{ name: "Predicted", count: result.predicted }].concat(result.counts.map(function (item) {
      return { name: "After " + item.step, count: item.count };
    }));
    var most = Math.max.apply(null, steps.map(function (step) { return step.count; }).concat([1]));
    var rows = result.grasps.map(function (grasp) {
      return h("tr", {}, [
        h("td", { text: String(grasp.rank) }),
        h("td", { text: grasp.score.toFixed(3) }),
        h("td", { text: grasp.width.toFixed(3) }),
        h("td", { text: grasp.depth.toFixed(3) }),
        h("td", { class: "is-vec", text: grasp.position.map(signed).join(" ") }),
        h("td", { class: "is-vec", text: grasp.approach.map(function (value) { return (value < 0 ? "−" : "+") + Math.abs(value).toFixed(2); }).join(" ") }),
      ]);
    });
    var lines = result.lines.map(function (line) {
      return [line[0], [line[1]]];
    });
    var caption = "The " + result.drawn + " best grasps, coloured by score rank from red to green." +
      (result.input === "the example scene" ? " A synthetic scene." : "") +
      (result.drawn_frame !== result.frame ? " Drawn before the step that changed their frame." : "");
    return h("div", { class: "gx-run-body" }, [
      h("div", { class: "gx-res-top" }, [
        h("span", { class: "gx-pill s-ok" }, [icon("check", 14), h("span", { text: "Done" })]),
        h("span", { class: "gx-res-meta", text: seconds(result.seconds.total) + (result.reused ? " · model reused" : "") }),
      ]),
      h("h2", { class: "gx-res-title", text: result.target }),
      h("p", { class: "gx-muted", text: "on " + result.input }),
      h("figure", { class: "gx-run-picture" }, [
        h("img", {
          src: runLive.urls ? runLive.urls.png : "",
          width: String(result.picture_size[0]),
          height: String(result.picture_size[1]),
          alt: "The " + result.drawn + " best grasps of " + result.target + " on " + result.input + ", drawn as grippers over the point cloud",
        }),
        h("figcaption", { text: caption }),
      ]),
      h("div", { class: "gx-block" }, [
        h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "Through post-processing" })]),
        h("ol", { class: "gx-funnel" }, steps.map(function (step) {
          return h("li", {}, [
            h("span", { class: "gx-funnel-name", text: step.name }),
            h("b", { text: num(step.count) }),
            h("span", { class: "gx-funnel-track", "aria-hidden": "true" }, [h("span", { style: "width:" + Math.max(2, Math.round((100 * step.count) / most)) + "%" })]),
          ]);
        })),
      ]),
      result.notes.length ? alertBox("unstable", "alert", "Note", result.notes.map(function (note) {
        return h("p", { text: note });
      })) : null,
      result.total
        ? h("div", { class: "gx-block" }, [
            h("div", { class: "gx-sec-head" }, [
              h("span", { class: "gx-label", text: "Best " + rows.length + " of " + num(result.total) }),
              infoButton("Camera frame, metres: x right, y down, z forward. The approach is the first column of the rotation. The .npy holds one row per grasp, best first: score, width, height, depth, rotation (9, row-major), translation (3), object id.", "About the table"),
            ]),
            h("div", { class: "gx-table-wrap", role: "region", "aria-label": "The best grasps", tabindex: "0" }, [
              h("table", { class: "gx-table" }, [
                h("thead", {}, [h("tr", {}, ["#", "Score", "Width", "Depth", "Position (m)", "Approach"].map(function (name) {
                  return h("th", { scope: "col", text: name });
                }))]),
                h("tbody", {}, rows),
              ]),
            ]),
          ])
        : h("p", { class: "gx-muted", text: "No grasp was returned: the detector found nothing it would grasp" + (result.counts.length ? ", or the filter removed everything." : ".") }),
      h("div", { class: "gx-downloads" }, [
        h("a", { class: "gx-btn", href: runLive.urls ? runLive.urls.npy : "", download: result.array_name, "data-key": "run-npy" }, [icon("download", 14), h("span", { text: "Grasps .npy" }), h("span", { class: "gx-muted", text: result.array_shape[0] + " × 17" })]),
        h("a", { class: "gx-btn", href: runLive.urls ? runLive.urls.png : "", download: result.array_name.replace(/\.npy$/, ".png"), "data-key": "run-png" }, [icon("download", 14), h("span", { text: "Picture .png" })]),
      ]),
      h("div", { class: "gx-block" }, [
        h("div", { class: "gx-sec-head" }, [h("span", { class: "gx-label", text: "The same in a terminal" })]),
        terminal(result.command, "run-command", "the command"),
      ]),
      h("div", { class: "gx-folds" }, [fold("run-ran", "What ran", [keyValues(lines)], "config, weights, input")]),
    ]);
  }

  function runOutput() {
    var job = runLive.job;
    var body;
    function empty(line) {
      return h("div", { class: "gx-empty is-run" }, [h("span", { class: "gx-empty-icon" }, [icon("image", 22)]), h("p", { text: line })]);
    }
    if (!liveBase) {
      body = empty("The picture of the predicted grasps, the best ten and the counts appear here.");
    } else if (runLive.failure) {
      body = failureCard(runLive.failure, null);
    } else if (runLive.sending) {
      body = h("div", { class: "gx-run-wait" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("span", { text: "Sending the request…" })]);
    } else if (job && job.state === "running") {
      body = h("div", { class: "gx-run-body" }, [
        h("div", { class: "gx-run-wait" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("span", { text: "Running " + job.target + " on " + job.input })]),
        h("p", { class: "gx-muted gx-small", text: seconds(job.elapsed) + " so far. The first run builds the model and loads its weights; the next ones reuse them." }),
        stageList(job.stages),
      ]);
    } else if (job && job.state === "failed") {
      body = failureCard(job.error, job);
    } else if (job && job.result) {
      body = resultView(job.result);
    } else {
      body = empty("Press Run: the picture of the predicted grasps appears here.");
    }
    return h("section", { class: "gx-card gx-run-out", id: "gx-run-out", "aria-live": "polite", "aria-label": "Run result" }, [body]);
  }

  function runOff() {
    var command = "grasppanda explore --weights-dir <dir>";
    return h("div", { class: "gx-banner" }, [
      h("span", { class: "gx-banner-icon" }, [icon("terminal", 18)]),
      h("div", { class: "gx-banner-text" }, [
        h("p", {}, [h("b", { text: "Runs on your machine." }), h("span", { class: "gx-muted", text: " Start the explorer with a folder of weights to run a method here." })]),
        h("div", { class: "gx-banner-actions" }, [
          h("span", { class: "gx-cmdpill" }, [h("code", { text: command }), copyButton(command, "copy-run-explore", command, { cls: "is-pill" })]),
          external(docHref("user_guides/inference") + "#run-in-the-browser", "How it works"),
        ]),
      ]),
    ]);
  }

  function runView() {
    var off = !liveBase;
    if (!off) loadSetup();
    var top = null;
    if (off) top = runOff();
    else if (runLive.status === "loading") top = h("div", { class: "gx-banner is-plain" }, [h("span", { class: "gx-spin", "aria-hidden": "true" }), h("p", { text: "Asking the server which weights it reads…" })]);
    else if (runLive.status === "error") top = alertBox("fails", "alertCircle", "No answer from the server", [rich(runLive.error, "p")]);
    var form = h("section", { class: "gx-card gx-run-form" + (off ? " is-off" : ""), "aria-disabled": off ? "true" : null, "aria-label": "Run a method" }, [
      h("div", { class: "gx-card-head" }, [h("h2", { class: "gx-card-title", text: "Run a method" }), h("p", { class: "gx-muted", text: "One frame, the weights you name. Nothing is trained." })]),
      h("ol", { class: "gx-steps" }, [methodStep(off), weightsStep(off), inputStep(off), optionsStep(off)]),
      goButton(off),
    ]);
    return h("div", { class: "gx-run" }, [top, h("div", { class: "gx-run-grid" }, [form, runOutput()])]);
  }

  // -------------------------------------------------------------------------- render

  /* The result column follows the page; when it is taller than the window it scrolls with
   * the page until its end is in view, then stays. */
  function fitSide() {
    var inner = root.querySelector(".gx-side-inner");
    if (!inner) return;
    var bar = document.querySelector(".md-header");
    var base = (bar ? bar.getBoundingClientRect().height : 0) + 16;
    inner.style.top = Math.min(base, window.innerHeight - inner.offsetHeight - 16) + "px";
  }

  function watch() {
    watchers.forEach(function (stop) {
      stop();
    });
    watchers = [];
    var inner = root.querySelector(".gx-side-inner");
    if (inner && window.ResizeObserver) {
      var sizes = new window.ResizeObserver(fitSide);
      sizes.observe(inner);
      watchers.push(function () {
        sizes.disconnect();
      });
    }
    /* The floating verdict of a narrow screen stands in for the answer while the answer
     * is out of sight. */
    var bar = root.querySelector(".gx-bar");
    var panel = document.getElementById("gx-result");
    if (bar && panel && window.IntersectionObserver) {
      var seen = new window.IntersectionObserver(function (entries) {
        bar.classList.toggle("is-away", entries[entries.length - 1].isIntersecting);
      });
      seen.observe(panel);
      watchers.push(function () {
        seen.disconnect();
      });
    }
  }

  function render() {
    var active = document.activeElement;
    var key = state.focusKey || (active && root.contains(active) && active.getAttribute ? active.getAttribute("data-key") : null);
    var caret = !state.focusKey && key && active.selectionStart !== undefined && active.selectionStart !== null ? active.selectionStart : null;
    state.focusKey = "";
    hideTip();
    var view = state.view === "grid" ? gridView() : state.view === "run" ? runView() : buildView();
    var body = h("div", { class: "gx-body gx-view-" + state.view }, [
      header(),
      h("div", { class: "gx-tabrow" }, [tabs()]),
      notice(),
      h("div", { class: "gx-panel", id: "gx-panel", role: "tabpanel", "aria-labelledby": "gx-tab-" + state.view }, [view]),
    ]);
    var old = root.querySelector(".gx-body");
    if (old) root.replaceChild(body, old);
    else {
      root.textContent = "";
      root.appendChild(body);
    }
    placeAll();
    fitSide();
    watch();
    state.flash = "";
    if (key) {
      var again = root.querySelector('[data-key="' + key.replace(/"/g, '\\"') + '"]');
      if (again) {
        again.focus({ preventScroll: true });
        if (caret !== null && again.setSelectionRange) {
          try {
            again.setSelectionRange(caret, caret);
          } catch (error) {
            /* a field that does not support it */
          }
        }
      }
    }
    var current = root.querySelector(".gx-opt.is-active");
    if (current && current.scrollIntoView) current.scrollIntoView({ block: "nearest" });
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
    closePopovers();
  }

  /* The open popover, its anchor and how to close it. */
  function openPopover() {
    if (state.open) return { anchor: "lane-" + state.open, close: { open: "", laneQuery: "" } };
    if (state.cellMenu) return { anchor: state.cellMenu, close: { cellMenu: "" } };
    if (state.filters) return { anchor: "grid-filters", close: { filters: false } };
    if (state.docsMenu) return { anchor: "grid-docs", close: { docsMenu: false } };
    if (state.about) return { anchor: "about", close: { about: false } };
    return null;
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
    root.classList.add("is-ready");
    document.addEventListener("keydown", function (event) {
      var open = openPopover();
      if (event.key === "Escape" && open) {
        event.preventDefault();
        state.focusKey = open.anchor;
        set(open.close);
        return;
      }
      /* The sheet of a phone is modal: Tab stays inside it. */
      var sheet = root.querySelector(".gx-picker.is-sheet");
      if (event.key === "Tab" && sheet) {
        var items = Array.prototype.slice.call(sheet.querySelectorAll("a[href], button, input, [tabindex='0']"));
        if (!items.length) return;
        var at = items.indexOf(document.activeElement);
        if (event.shiftKey && at <= 0) {
          event.preventDefault();
          items[items.length - 1].focus();
        } else if (!event.shiftKey && at === items.length - 1) {
          event.preventDefault();
          items[0].focus();
        }
      }
    });
    /* A click outside the open popover closes it. It runs after the click of the target,
     * so a click on another anchor opens that one instead of being swallowed. */
    document.addEventListener("click", function (event) {
      var open = openPopover();
      if (!open) return;
      var target = event.target;
      if (target.closest && (target.closest(".gx-pop") || target.closest('[data-key="' + open.anchor + '"]'))) return;
      set(open.close);
    });
    window.addEventListener("resize", function () {
      placeAll();
      fitSide();
      hideTip();
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
