import { parseNumber } from "../media/image";

/// Cap for frontend-derived conversation titles. Matches the backend preview cap
/// (conversations.rs `PREVIEW_MAX_CHARS`) so a frontend-derived title and the server-derived
/// preview/title stay byte-consistent.
export const CONVERSATION_TITLE_MAX_CHARS = 80;

/// Convert the frontend's per-session `params` — stringified numbers bound to the Sampling inputs —
/// into the typed `ConversationParams` shape the Tauri `save_conversation` command expects
/// (systemPrompt: String, temperature/topP: f32, maxTokens: u32, disableThinking: bool). The Rust
/// serde types reject JSON strings, so the conversion is mandatory before persisting.
export function paramsToConversation(params) {
  return {
    systemPrompt: params.systemPrompt ?? "",
    temperature: parseNumber(params.temperature) ?? 0,
    topP: parseNumber(params.topP) ?? 0,
    maxTokens: parseNumber(params.maxTokens) ?? 0,
    disableThinking: Boolean(params.disableThinking),
  };
}

/// Inverse of `paramsToConversation`: restore typed conversation params back into the string-input
/// shape the Sampling panel binds to. Mirrors `paramsFromSettings` so a loaded conversation drops
/// into the panel exactly like a fresh chat seeded from the app defaults.
export function paramsFromConversation(params) {
  const p = params ?? {};
  return {
    systemPrompt: p.systemPrompt ?? "",
    temperature: String(p.temperature ?? ""),
    topP: String(p.topP ?? ""),
    maxTokens: String(p.maxTokens ?? ""),
    disableThinking: Boolean(p.disableThinking),
  };
}

/// Derive a conversation title from the first user message: collapse whitespace and cap at
/// `CONVERSATION_TITLE_MAX_CHARS` code points with an ellipsis. This is the lazy-save title used on
/// the first send of a new chat; once set it is preserved across upserts.
export function deriveConversationTitle(messages) {
  for (const message of messages ?? []) {
    if (message?.role !== "user") continue;
    const text = String(message.content ?? "").replace(/\s+/g, " ").trim();
    if (!text) continue;
    return truncateForTitle(text, CONVERSATION_TITLE_MAX_CHARS);
  }
  return "New conversation";
}

export function truncateForTitle(text, maxChars) {
  const chars = Array.from(text);
  if (chars.length <= maxChars) return text;
  return `${chars.slice(0, maxChars).join("")}\u{2026}`;
}
