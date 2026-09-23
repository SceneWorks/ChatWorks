// The decode-path status view (sc-24139): which path the runtime reports for the served model.
// Written with `createElement` rather than JSX so `node --test` can render it without a bundler.
import React from "react";
import { decodePathRows } from "../state/decodePath.js";

const h = React.createElement;

export function DecodePathStatus({ engineStatus, title = "Decode path" }) {
  const rows = decodePathRows(engineStatus);
  return h(
    "section",
    { className: "decode-path", "aria-label": title },
    h("p", { className: "eyebrow" }, title),
    h(
      "dl",
      { className: "decode-path-rows" },
      rows.map((row) =>
        h(
          "div",
          { className: "decode-path-row", key: row.key, "data-row": row.key },
          h("dt", null, row.label),
          h("dd", null, row.value, row.detail ? h("small", null, row.detail) : null),
        ),
      ),
    ),
  );
}
