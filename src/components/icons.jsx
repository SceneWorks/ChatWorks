/// `Copy` and `Rewind` are not part of `@sceneworks/ui` (sc-8147). These inline SVGs mirror the
/// package's icon base (24×24 viewBox, `currentColor` stroke, round joins, strokeWidth 1.7) so they
/// render identically alongside `Icon.*` glyphs. A `Check` glyph backs the Copy "Copied" state.
export function CopyIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M9 4h6v2H9z M8 6h8a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2" />
    </svg>
  );
}

export function RewindIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M9 6L4 12l5 6 M4 12h9a6 6 0 0 1 6 6" />
    </svg>
  );
}

export function CheckIcon({ size = 18, ...rest }) {
  return (
    <svg
      aria-hidden="true"
      fill="none"
      height={size}
      stroke="currentColor"
      strokeLinecap="round"
      strokeLinejoin="round"
      strokeWidth="1.7"
      viewBox="0 0 24 24"
      width={size}
      {...rest}
    >
      <path d="M5 12l5 5L19 7" />
    </svg>
  );
}
