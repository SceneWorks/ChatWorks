// The decode-path status view (sc-24139): which path the runtime reports for the served model.
// Written with `createElement` rather than JSX so `node --test` can render it without a bundler.
import React from "react";
import { decodePathRows, speculativeNotice } from "../state/decodePath.js";

const h = React.createElement;

/// The one-time "speculative decoding is available" notice with its one-click action. Renders
/// nothing unless `speculativeNotice` says to show it.
export function SpeculativeNotice({ appSettings, executionBackend, onEnableAuto, onDismiss, busy = false }) {
  const notice = speculativeNotice(appSettings, executionBackend);
  if (!notice.show) return null;
  return h(
    "div",
    { className: "speculative-notice", role: "status", "data-notice": "speculative" },
    h("p", null, h("strong", null, notice.message), h("small", null, notice.detail)),
    h(
      "div",
      { className: "speculative-notice-actions" },
      h("button", { className: "primary-btn", disabled: busy, onClick: onEnableAuto, type: "button" }, notice.actionLabel),
      h("button", { className: "ghost-btn", disabled: busy, onClick: onDismiss, type: "button" }, notice.dismissLabel),
    ),
  );
}

/// Rows grouped by section, in order: the host and load rows, then the "Last generation" rows.
function sections(rows) {
  const groups = [];
  for (const row of rows) {
    const group = groups.at(-1);
    if (group && group.section === row.section) group.rows.push(row);
    else groups.push({ section: row.section, rows: [row] });
  }
  return groups;
}

export function DecodePathStatus({ engineStatus, title = "Decode path", notice = null }) {
  const groups = sections(decodePathRows(engineStatus));
  return h(
    "section",
    { className: "decode-path", "aria-label": title },
    h("p", { className: "eyebrow" }, title),
    notice ? h(SpeculativeNotice, notice) : null,
    groups.map((group) =>
      h(
        React.Fragment,
        { key: group.section ?? "model" },
        group.section ? h("p", { className: "decode-path-section" }, group.section) : null,
        h(
          "dl",
          { className: "decode-path-rows", "data-section": group.section ?? "model" },
          group.rows.map((row) =>
            h(
              "div",
              { className: "decode-path-row", key: row.key, "data-row": row.key },
              h("dt", null, row.label),
              h("dd", null, row.value, row.detail ? h("small", null, row.detail) : null),
            ),
          ),
        ),
      ),
    ),
  );
}
