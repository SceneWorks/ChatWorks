import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { useApp, paramsFromSettings } from "./AppContext";
import {
  paramsToConversation,
  paramsFromConversation,
  deriveConversationTitle,
} from "./conversations";

export const ConversationsContext = createContext(null);
export const ChatStateContext = createContext(null);

/// Owns the full conversation lifecycle and the per-chat ephemeral state, calling story A's five
/// Tauri commands (`list_conversations`, `get_conversation`, `save_conversation`,
/// `rename_conversation`, `delete_conversation`) via `invoke`.
///
/// The state is split across two contexts on purpose so the high-frequency transcript updates
/// (streaming tokens) do not re-render subscribers that only care about the metadata cache / active
/// id (e.g. the history nav in story C):
///   - `ConversationsContext` (rarely changes): active id, metadata cache, and the lifecycle
///     actions (`selectConversation`, `startNewChat`, `persistConversation`, `renameConversation`,
///     `deleteConversation`, `refreshConversations`).
///   - `ChatStateContext` (changes every token): `messages`, `draft`, `params`, `attachments`,
///     `videoAttachments` and their setters.
///
/// App start opens a fresh, unsaved new chat (activeConversationId === null) and loads the history
/// metadata cache independently — there is no auto-resume.
export function ConversationsProvider({ children }) {
  const { appSettings } = useApp();
  const defaultParams = useMemo(() => paramsFromSettings(appSettings.sampling), [appSettings]);

  const [activeConversationId, setActiveConversationId] = useState(null);
  const [conversations, setConversations] = useState([]);
  const [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState("");
  const [params, setParams] = useState(defaultParams);
  const [attachments, setAttachments] = useState([]);
  const [videoAttachments, setVideoAttachments] = useState([]);
  // `busy` is the active-stream flag. It lives here (not in ChatScreen) so the history nav — a
  // sibling of ChatScreen in the shell — can read it and hard-block conversation switching while a
  // response is streaming (story C). It only flips at stream boundaries, so exposing it through
  // ConversationsContext does not add per-token re-renders to nav subscribers.
  const [busy, setBusy] = useState(false);

  // Refs let the action callbacks read the latest state without depending on it, which keeps the
  // ConversationsContext value referentially stable across streaming-driven `messages` updates.
  const activeIdRef = useRef(activeConversationId);
  const conversationsRef = useRef(conversations);
  const messagesRef = useRef(messages);
  const paramsRef = useRef(params);
  const busyRef = useRef(busy);
  useEffect(() => {
    activeIdRef.current = activeConversationId;
  }, [activeConversationId]);
  useEffect(() => {
    conversationsRef.current = conversations;
  }, [conversations]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  useEffect(() => {
    paramsRef.current = params;
  }, [params]);
  useEffect(() => {
    busyRef.current = busy;
  }, [busy]);

  // Metadata cache (from `list_conversations`) so the nav renders without a refetch; refreshed
  // after every save/rename/delete.
  const refreshConversations = useCallback(() => {
    return invoke("list_conversations")
      .then((list) => {
        setConversations(Array.isArray(list) ? list : []);
        return list;
      })
      .catch(() => {
        setConversations([]);
        return [];
      });
  }, []);

  // A single send can persist several times (user message, each assistant turn, each tool-result
  // batch — up to MAX_TOOL_STEPS). Each persist used to await a full re-list, making the steady
  // state O(turns × conversations) per send (F-013, code-review 2026-07-02). This coalesces rapid
  // refreshes into one trailing re-list at the end of the burst; rename/delete still refresh
  // immediately because the user expects instant feedback there.
  const refreshDebounceRef = useRef(null);
  const scheduleConversationRefresh = useCallback(() => {
    if (refreshDebounceRef.current) {
      clearTimeout(refreshDebounceRef.current);
    }
    refreshDebounceRef.current = setTimeout(() => {
      refreshDebounceRef.current = null;
      refreshConversations();
    }, 150);
  }, [refreshConversations]);

  useEffect(() => {
    refreshConversations();
  }, [refreshConversations]);

  // Clear any pending debounced refresh when the provider unmounts, so a timer doesn't fire its
  // setState after the component is gone (PR #30 review).
  useEffect(() => {
    return () => {
      if (refreshDebounceRef.current) {
        clearTimeout(refreshDebounceRef.current);
        refreshDebounceRef.current = null;
      }
    };
  }, []);

  // A fresh new chat tracks the app default sampling profile; a loaded conversation owns the params
  // it was run with, so defaults are only re-applied while there is no active conversation.
  useEffect(() => {
    if (activeConversationId === null) {
      setParams(defaultParams);
    }
  }, [defaultParams, activeConversationId]);

  /// Reset to a fresh, unsaved new chat: clears messages/draft/attachments, clears the active id,
  /// and resets params to the app defaults. The saved history is untouched.
  const startNewChat = useCallback(() => {
    setActiveConversationId(null);
    setMessages([]);
    setDraft("");
    setAttachments([]);
    setVideoAttachments([]);
    setParams(paramsFromSettings(appSettings.sampling));
  }, [appSettings]);

  /// Load a conversation: `get_conversation(id)` → messages into the transcript + params restored
  /// into the Sampling panel, and set as active. Throws on failure so the caller (story C) can
  /// surface the error. Hard-blocks while a response is streaming — switching the transcript
  /// mid-stream would discard the in-flight assistant turn; the nav also disables row selection
  /// while busy, so this is a defensive backstop.
  const selectConversation = useCallback(async (id) => {
    if (busyRef.current) return;
    const conversation = await invoke("get_conversation", { id });
    setActiveConversationId(conversation.id);
    setMessages(Array.isArray(conversation.messages) ? conversation.messages : []);
    setDraft("");
    setAttachments([]);
    setVideoAttachments([]);
    setParams(paramsFromConversation(conversation.params));
    return conversation;
  }, []);

  /// Persist the active conversation. On the first send of a new chat this lazily creates it with a
  /// `crypto.randomUUID()` id, a title derived from the first user message, and the active params;
  /// on subsequent turns / rewind it upserts the same id and the backend bumps `updatedAt`.
  /// Callers (the send loop, story D's rewind) should pass `messages`/`params` explicitly so the
  /// committed transcript is captured even when React state has not flushed yet; otherwise the
  /// latest state (via refs) is used. Always refreshes the metadata cache on success.
  const persistConversation = useCallback(async ({ id, messages: msgs, params: p, title } = {}) => {
    const resolvedId = id ?? activeIdRef.current ?? crypto.randomUUID();
    const finalMessages = msgs ?? messagesRef.current;
    const finalParams = p ?? paramsRef.current;
    const existing = conversationsRef.current.find((entry) => entry.id === resolvedId);
    const finalTitle = title ?? existing?.title ?? deriveConversationTitle(finalMessages);
    const record = {
      id: resolvedId,
      title: finalTitle,
      createdAt: 0,
      updatedAt: 0,
      params: paramsToConversation(finalParams),
      messages: finalMessages,
    };
    const saved = await invoke("save_conversation", { conversation: record });
    setActiveConversationId(saved.id);
    // Coalesce the metadata-cache refresh: a single send persists many times, so a debounced trailing
    // re-list replaces a full read_dir per persist (F-013). The id is set synchronously above, so the
    // active row stays correct while the sidebar sorts/preview catches up.
    scheduleConversationRefresh();
    return saved;
  }, [scheduleConversationRefresh]);

  const renameConversation = useCallback(
    async (id, title) => {
      const meta = await invoke("rename_conversation", { id, title });
      await refreshConversations();
      return meta;
    },
    [refreshConversations],
  );

  const deleteConversation = useCallback(
    async (id) => {
      await invoke("delete_conversation", { id });
      if (id === activeIdRef.current) {
        startNewChat();
      }
      await refreshConversations();
    },
    [refreshConversations, startNewChat],
  );

  const conversationsValue = useMemo(
    () => ({
      activeConversationId,
      conversations,
      busy,
      setBusy,
      selectConversation,
      startNewChat,
      persistConversation,
      renameConversation,
      deleteConversation,
      refreshConversations,
    }),
    [
      activeConversationId,
      conversations,
      busy,
      setBusy,
      selectConversation,
      startNewChat,
      persistConversation,
      renameConversation,
      deleteConversation,
      refreshConversations,
    ],
  );

  const chatStateValue = useMemo(
    () => ({
      messages,
      setMessages,
      draft,
      setDraft,
      params,
      setParams,
      attachments,
      setAttachments,
      videoAttachments,
      setVideoAttachments,
    }),
    [messages, draft, params, attachments, videoAttachments],
  );

  return (
    <ConversationsContext.Provider value={conversationsValue}>
      <ChatStateContext.Provider value={chatStateValue}>{children}</ChatStateContext.Provider>
    </ConversationsContext.Provider>
  );
}

export function useConversations() {
  const context = useContext(ConversationsContext);
  if (!context) throw new Error("useConversations must be used inside ConversationsProvider");
  return context;
}

export function useChatState() {
  const context = useContext(ChatStateContext);
  if (!context) throw new Error("useChatState must be used inside ConversationsProvider");
  return context;
}
