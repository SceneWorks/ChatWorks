import { useState } from "react";
import { messageTextContent } from "../api/sse";
import { CheckIcon, CopyIcon, PinIcon, RewindIcon } from "./icons";

/// Per-message Copy + Rewind + optional Pin-document actions rendered at the foot of every bubble
/// (sc-8147, SW-F356). Copy writes the message's text content to the clipboard (disabled on
/// image/video-only turns); Rewind asks the parent to drop this message and everything after it and
/// load its text into the composer (disabled mid-stream and on text-less turns); Pin document marks
/// `0..=index` as reusable document text protected across follow-up questions.
export function MessageActions({
  message,
  index,
  onRewind,
  busy,
  showPinDocument = false,
  pinnedDocument = false,
  onTogglePinDocument = null,
}) {
  const [copied, setCopied] = useState(false);
  const text = messageTextContent(message);
  const hasText = text.trim().length > 0;
  const canRewind = hasText && !busy;

  async function handleCopy() {
    if (!hasText) return;
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="message-actions">
      <button
        type="button"
        className="message-action-btn"
        onClick={handleCopy}
        disabled={!hasText}
        aria-label="Copy message text"
        title={hasText ? "Copy message text" : "No text to copy"}
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
      </button>
      <button
        type="button"
        className="message-action-btn"
        onClick={() => onRewind(index)}
        disabled={!canRewind}
        aria-label="Rewind to this message"
        title={
          busy ? "Wait for the response to finish" : hasText ? "Rewind to this message" : "No text to load"
        }
      >
        <RewindIcon />
      </button>
      {showPinDocument && onTogglePinDocument ? (
        <button
          type="button"
          className={pinnedDocument ? "message-action-pill active" : "message-action-pill"}
          onClick={() => onTogglePinDocument(index)}
          disabled={busy}
          aria-pressed={pinnedDocument}
          aria-label={pinnedDocument ? "Document pinned" : "Pin document"}
          title={
            busy
              ? "Wait for the response to finish"
              : pinnedDocument
                ? "Unpin reusable document boundary"
                : "Pin messages through this turn as a reusable document"
          }
        >
          <PinIcon size={14} />
          <span>{pinnedDocument ? "Document pinned" : "Pin document"}</span>
        </button>
      ) : null}
    </div>
  );
}
