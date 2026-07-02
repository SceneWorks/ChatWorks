import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Icon } from "@sceneworks/ui";
import { useApp, readStoredValue } from "../state/AppContext";
import { useConversations } from "../state/ConversationsContext";

/// Page size for the history list ("Show more…" reveals the next batch) and the localStorage key
/// that remembers the Chat group's expand/collapse state across sessions.
export const CHAT_NAV_PAGE_SIZE = 10;
export const CHAT_NAV_EXPANDED_KEY = "chatworks-chat-nav-expanded";

/// A single conversation row in the Chat history list: title (click to select), active highlight,
/// inline rename, and delete. Inline rename turns the title into a text input that commits on
/// Enter/blur and cancels on Escape.
export function ConversationRow({ conversation, active, busy, onSelect, onRename, onDelete }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(conversation.title);
  const inputRef = useRef(null);

  useEffect(() => {
    if (!editing) return;
    setDraft(conversation.title);
    const el = inputRef.current;
    if (el) {
      el.focus();
      el.select();
    }
  }, [editing, conversation.title]);

  const commitRename = () => {
    setEditing(false);
    const next = draft.trim();
    if (next && next !== conversation.title) onRename(conversation.id, next);
  };
  const cancelRename = () => setEditing(false);

  return (
    <div className={"conv-row" + (active ? " active" : "") + (editing ? " editing" : "")}>
      {editing ? (
        <input
          className="conv-title-input"
          onBlur={commitRename}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              event.preventDefault();
              commitRename();
            } else if (event.key === "Escape") {
              event.preventDefault();
              cancelRename();
            }
          }}
          ref={inputRef}
          type="text"
          value={draft}
        />
      ) : (
        <button
          className="conv-title"
          disabled={busy}
          onClick={() => onSelect(conversation.id)}
          title={conversation.title}
          type="button"
        >
          {conversation.title || "New chat"}
        </button>
      )}
      <div className="conv-actions">
        <button
          className="conv-action"
          onClick={() => setEditing(true)}
          title="Rename"
          type="button"
        >
          <Icon.Editor />
        </button>
        <button
          className="conv-action delete"
          disabled={busy && active}
          onClick={() => onDelete(conversation)}
          title="Delete"
          type="button"
        >
          <span aria-hidden="true">×</span>
        </button>
      </div>
    </div>
  );
}

/// Collapsible "Chat" parent group in the sidebar (story C). The header row toggles expand/collapse
/// and navigates to the Chat view; a "+ New chat" button starts a fresh conversation. Expanded, it
/// shows the most recent conversations from the metadata cache with client-side paging, inline
/// rename, delete, and active-conversation highlight. Selection is hard-blocked while a response is
/// streaming (`busy`), matching the "Clear conversation" guard.
export function ChatNavGroup() {
  const { activeView, setActiveView } = useApp();
  const {
    activeConversationId,
    conversations,
    busy,
    selectConversation,
    startNewChat,
    renameConversation,
    deleteConversation,
  } = useConversations();

  const [expanded, setExpanded] = useState(
    () => readStoredValue(CHAT_NAV_EXPANDED_KEY, "true") !== "false",
  );
  const [visibleCount, setVisibleCount] = useState(CHAT_NAV_PAGE_SIZE);

  useEffect(() => {
    try {
      window.localStorage.setItem(CHAT_NAV_EXPANDED_KEY, expanded ? "true" : "false");
    } catch {
      /* localStorage unavailable — keep state in-memory only */
    }
  }, [expanded]);

  // The backend already returns the cache sorted by `updatedAt` desc; sort defensively so the nav
  // order stays stable regardless of any future refresh ordering.
  const sorted = useMemo(
    () =>
      [...conversations].sort(
        (a, b) => (b.updatedAt ?? 0) - (a.updatedAt ?? 0) || String(a.id).localeCompare(String(b.id)),
      ),
    [conversations],
  );
  const visible = sorted.slice(0, visibleCount);
  const hasMore = visibleCount < sorted.length;

  const goChat = useCallback(() => setActiveView("Chat"), [setActiveView]);

  const handleHeaderClick = useCallback(() => {
    setExpanded((prev) => !prev);
    goChat();
  }, [goChat]);

  const handleNewChat = useCallback(() => {
    startNewChat();
    goChat();
  }, [startNewChat, goChat]);

  const handleSelect = useCallback(
    (id) => {
      selectConversation(id);
      goChat();
    },
    [selectConversation, goChat],
  );

  const handleRename = useCallback(
    (id, title) => {
      renameConversation(id, title).catch(() => {
        /* refreshConversations already ran inside the action; surface nothing in the nav */
      });
    },
    [renameConversation],
  );

  const handleDelete = useCallback(
    (conversation) => {
      const label = conversation.title || "this conversation";
      if (!window.confirm(`Delete "${label}"? This cannot be undone.`)) return;
      deleteConversation(conversation.id).catch(() => {
        /* ignore — the cache refresh inside the action keeps the list consistent */
      });
    },
    [deleteConversation],
  );

  return (
    <div className={"chat-nav-group" + (expanded ? " expanded" : "")}>
      <div className="chat-nav-header">
        <button
          aria-expanded={expanded}
          className={"chat-nav-toggle" + (activeView === "Chat" ? " is-active" : "")}
          onClick={handleHeaderClick}
          title="Chat"
          type="button"
        >
          <Icon.ChevDown className="chat-nav-chevron" />
          <span className="nav-label">Chat</span>
        </button>
        <button
          className="chat-nav-new icon-btn"
          disabled={busy}
          onClick={handleNewChat}
          title="New chat"
          type="button"
        >
          <Icon.Plus />
        </button>
      </div>
      {expanded ? (
        <div className="chat-nav-list">
          {sorted.length === 0 ? (
            <p className="chat-nav-empty">No conversations yet.</p>
          ) : (
            <>
              {visible.map((conversation) => (
                <ConversationRow
                  active={conversation.id === activeConversationId}
                  busy={busy}
                  conversation={conversation}
                  key={conversation.id}
                  onDelete={handleDelete}
                  onRename={handleRename}
                  onSelect={handleSelect}
                />
              ))}
              {hasMore ? (
                <button
                  className="chat-nav-more"
                  onClick={() => setVisibleCount((count) => count + CHAT_NAV_PAGE_SIZE)}
                  type="button"
                >
                  Show more…
                </button>
              ) : null}
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}
