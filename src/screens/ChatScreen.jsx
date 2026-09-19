import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { StatusDot } from "@sceneworks/ui";
import { useApp } from "../state/AppContext";
import { useChatState, useConversations } from "../state/ConversationsContext";
import {
  buildLocalApiBase,
  chatRequestBody,
  messageTextContent,
  parseToolArguments,
  streamChatCompletion,
  supportsThinking,
  supportsTools,
  supportsVideo,
  supportsVision,
} from "../api/sse";
import { normalizeImageAttachment } from "../media/image";
import { sampleVideoAttachment } from "../media/video";
import { MessageActions } from "../components/MessageActions";
import { MessageContent } from "../components/MessageContent";
import { GenerationControls } from "../components/GenerationControls";
import { formatToolArguments, ToolCallList, ToolResult } from "../components/ToolCallList";

/// The maximum number of model→tool→model round-trips in a single send, to bound runaway loops.
const MAX_TOOL_STEPS = 8;

function messageMedia(message) {
  if (Array.isArray(message?.media)) return message.media;
  return [
    ...(message?.images ?? []).map((url) => ({ type: "image", url })),
    ...(message?.videos ?? []).map((video) => ({ type: "video", ...video })),
  ];
}

function matchesHttpUrl(url) {
  return url.protocol === "https:" || url.protocol === "http:";
}

export function ChatScreen() {
  const { engineStatus, refreshEngineStatus, appSettings, apiAuthToken } = useApp();
  const { activeConversationId, persistConversation, startNewChat, busy, setBusy } = useConversations();
  const {
    messages,
    setMessages,
    draft,
    setDraft,
    params,
    setParams,
    mediaAttachments,
    setMediaAttachments,
  } = useChatState();
  const [serverStatus, setServerStatus] = useState(null);
  const [error, setError] = useState(null);
  const [pendingAttachments, setPendingAttachments] = useState(0);
  const [mediaUrl, setMediaUrl] = useState("");
  const [mediaUrlKind, setMediaUrlKind] = useState("image");
  const [toolSpecs, setToolSpecs] = useState([]); // OpenAI function-tool defs from the backend
  const [toolsEnabled, setToolsEnabled] = useState(true); // offer tools when the model supports them
  const [pendingApproval, setPendingApproval] = useState(null); // {calls, decisions, resolve}
  // AbortController for the in-flight stream, so a Stop click (or unmount) cancels the fetch + the
  // backend generation (F-004). `null` when no stream is in flight.
  const abortRef = useRef(null);
  const samplingPresetRef = useRef(null);
  const thinkingCapable = supportsThinking(engineStatus);
  const visionCapable = supportsVision(engineStatus);
  const videoCapable = supportsVideo(engineStatus);
  const toolsCapable = supportsTools(engineStatus);
  const apiBase = buildLocalApiBase(serverStatus);
  const canSend =
    Boolean(engineStatus?.loaded) &&
    !busy &&
    pendingAttachments === 0 &&
    (Boolean(draft.trim()) || mediaAttachments.length > 0);

  // Load the built-in tool definitions once; the chat loop offers them when tools are enabled.
  useEffect(() => {
    invoke("list_builtin_tools")
      .then((specs) => setToolSpecs(Array.isArray(specs) ? specs : []))
      .catch(() => setToolSpecs([]));
  }, []);

  // Resolve the pending approval promise once every proposed call has an Approve/Deny decision.
  useEffect(() => {
    if (pendingApproval && pendingApproval.decisions.every((decision) => decision !== null)) {
      pendingApproval.resolve(pendingApproval.decisions);
      setPendingApproval(null);
    }
  }, [pendingApproval]);

  // Open the approval panel for a turn's tool calls; resolves with a per-call approve/deny array.
  const requestApproval = useCallback((calls) => {
    return new Promise((resolve) => {
      setPendingApproval({ calls, decisions: calls.map(() => null), resolve });
    });
  }, []);

  const decideApproval = useCallback((index, approved) => {
    setPendingApproval((current) => {
      if (!current) return current;
      const decisions = current.decisions.slice();
      decisions[index] = approved;
      return { ...current, decisions };
    });
  }, []);

  const refreshServerStatus = useCallback(() => {
    return invoke("openai_server_status")
      .then((status) => {
        setServerStatus(status);
        return status;
      })
      .catch(() => {
        setServerStatus(null);
        return null;
      });
  }, []);

  useEffect(() => {
    refreshServerStatus();
  }, [refreshServerStatus]);

  // A provider may expose only one visual modality. Keep the URL picker on an offered choice when
  // the served model changes, rather than producing a select value with no matching option.
  useEffect(() => {
    if (videoCapable && !visionCapable) setMediaUrlKind("video");
    if (visionCapable && !videoCapable) setMediaUrlKind("image");
  }, [visionCapable, videoCapable]);

  // Providers can publish separate thinking/non-thinking model-card presets. Apply the relevant
  // one once per served model/mode as an editable initial value; the request contains the concrete
  // values afterwards, and later user edits are never replaced by the runtime.
  useEffect(() => {
    const loaded = engineStatus?.loaded;
    const presets = loaded?.provider?.capabilities?.model_sampling_defaults;
    const preset = params.disableThinking ? presets?.non_thinking : presets?.thinking;
    const key = loaded && preset ? `${loaded.source}:${params.disableThinking ? "off" : "on"}` : null;
    if (!preset || samplingPresetRef.current === key) return;
    samplingPresetRef.current = key;
    setParams((current) => ({
      ...current,
      temperature: String(preset.temperature),
      topP: String(preset.top_p),
      topK: String(preset.top_k),
      presencePenalty: String(preset.presence_penalty),
      repetitionPenalty: String(preset.repetition_penalty),
      repetitionContext: String(preset.repetition_context),
    }));
  }, [engineStatus?.loaded, params.disableThinking, setParams]);

  /// Rewind (sc-8147, decision 3): drop message `index` and every message after it, load that
  /// message's text into the composer, and persist the trimmed transcript so the trim survives a
  /// relaunch. Blocked mid-stream and on text-less turns (the buttons are disabled then; this is a
  /// defensive backstop). The in-memory trim always runs even when there is no saved conversation
  /// yet (unsaved new chat), matching the existing lazy-save semantics.
  const handleRewind = useCallback(
    (index) => {
      if (busy) return;
      const target = messages[index];
      const text = target ? messageTextContent(target).trim() : "";
      if (!text) return;
      const trimmed = messages.slice(0, index);
      setMessages(trimmed);
      setDraft(text);
      if (activeConversationId !== null) {
        persistConversation({ messages: trimmed }).catch((err) => {
          setError(err instanceof Error ? err.message : String(err));
        });
      }
    },
    [busy, messages, setMessages, setDraft, activeConversationId, persistConversation],
  );

  async function handleSubmit(eventArg) {
    eventArg.preventDefault();
    if (!canSend) return;
    setBusy(true);
    setError(null);
    const userMessage = {
      role: "user",
      content: draft.trim(),
      media: mediaAttachments,
    };
    // `conversation` is the committed transcript; the in-flight assistant turn is appended for
    // rendering and only committed once it finishes streaming.
    let conversation = [...messages, userMessage];
    setMessages(conversation);
    setDraft("");
    setMediaAttachments([]);

    // Lazy save / upsert: on the first send of a new chat this creates the conversation
    // (`save_conversation` with a `crypto.randomUUID()` id, a title derived from the first user
    // message, and the active per-session `params`); on subsequent commits it upserts the same id
    // and the backend bumps `updatedAt`. Capturing the user message here means the conversation
    // survives an interrupted stream. `conversationId` is tracked locally because the context's
    // `activeConversationId` does not flush within this handler.
    let conversationId = activeConversationId ?? crypto.randomUUID();
    const persist = async (transcript) => {
      try {
        const saved = await persistConversation({
          id: conversationId,
          messages: transcript,
          params,
        });
        conversationId = saved.id;
      } catch (cause) {
        // Persistence failures must not abort the in-flight chat; surface a soft error.
        setError(`Could not save conversation: ${String(cause?.message ?? cause)}`);
      }
    };
    await persist(conversation);

    // The AbortController for this send: a Stop click (or unmount) aborts the in-flight fetch, the
    // local server observes the dropped SSE receiver and cancels its generation, and the loop below
    // commits the partial turn (F-004). Cleared in `finally`.
    const abort = new AbortController();
    abortRef.current = abort;

    try {
      const status = await refreshServerStatus();
      const nextEngineStatus = await refreshEngineStatus();
      const activeEngineStatus = nextEngineStatus ?? engineStatus;
      const url = `${buildLocalApiBase(status)}/v1/chat/completions`;
      const headers = { "Content-Type": "application/json" };
      if (appSettings.server.authEnabled && apiAuthToken) {
        headers.Authorization = `Bearer ${apiAuthToken}`;
      }
      const offerTools = toolsEnabled && supportsTools(activeEngineStatus) && toolSpecs.length > 0;

      let hitStepLimit = true;
      for (let step = 0; step < MAX_TOOL_STEPS; step += 1) {
        const committed = conversation;
        const assistantMessage = { role: "assistant", content: "", thinking: "" };
        setMessages([...committed, assistantMessage]);

        const result = await streamChatCompletion({
          url,
          headers,
          body: chatRequestBody({
            engineStatus: activeEngineStatus,
            messages: committed,
            params,
            thinkingCapable,
            tools: offerTools ? toolSpecs : null,
          }),
          onUpdate: ({ content, thinking }) =>
            setMessages([...committed, { ...assistantMessage, content, thinking }]),
          signal: abort.signal,
        });

        // A user-initiated Stop aborts the stream; the server cancels and we get the partial turn
        // back with finishReason "stopped". Commit it and end the send (do not loop into tools).
        if (result.finishReason === "stopped") {
          const partial = { role: "assistant", content: result.content, thinking: result.thinking };
          if (result.content || result.thinking) {
            conversation = [...committed, partial];
          }
          setMessages(conversation);
          await persist(conversation);
          hitStepLimit = false;
          break;
        }

        // Commit the assistant turn (with any tool calls it requested).
        const finalAssistant = { role: "assistant", content: result.content, thinking: result.thinking };
        if (result.toolCalls.length) finalAssistant.tool_calls = result.toolCalls;
        conversation = [...committed, finalAssistant];
        setMessages(conversation);
        await persist(conversation);

        if (!result.toolCalls.length) {
          hitStepLimit = false;
          break;
        }

        // Human-in-the-loop: approve/deny each call, then execute the approved ones in the backend.
        const decisions = await requestApproval(result.toolCalls);
        const toolMessages = [];
        for (let i = 0; i < result.toolCalls.length; i += 1) {
          const call = result.toolCalls[i];
          if (!decisions[i]) {
            toolMessages.push({
              role: "tool",
              name: call.name,
              content: "Tool call denied by the user.",
              denied: true,
            });
            continue;
          }
          try {
            const output = await invoke("execute_tool", {
              name: call.name,
              arguments: parseToolArguments(call.arguments),
            });
            toolMessages.push({ role: "tool", name: call.name, content: String(output) });
          } catch (cause) {
            toolMessages.push({
              role: "tool",
              name: call.name,
              content: `Error: ${String(cause?.message ?? cause)}`,
              isError: true,
            });
          }
        }
        conversation = [...conversation, ...toolMessages];
        setMessages(conversation);
        await persist(conversation);
        // Loop: re-send the transcript (now with the tool results) for the model's next turn.
      }

      if (hitStepLimit) {
        setError(`Stopped after the tool-call step limit (${MAX_TOOL_STEPS}).`);
      }
    } catch (cause) {
      setError(String(cause?.message ?? cause));
      setMessages(conversation); // drop the in-flight assistant placeholder, keep committed turns
      await persist(conversation);
    } finally {
      abortRef.current = null;
      setBusy(false);
    }
  }

  /// Stop the in-flight generation (F-004): abort the SSE fetch (the server then cancels its
  /// generation on the dropped receiver) and also call the engine's stop_generation command as a
  /// belt-and-suspenders for the IPC path. The send loop commits the partial turn and ends.
  const handleStop = useCallback(() => {
    if (abortRef.current) {
      abortRef.current.abort();
    }
    invoke("stop_generation").catch(() => {
      /* best-effort; the fetch abort is the primary cancel path */
    });
  }, []);

  function updateParam(key, value) {
    setParams((current) => ({ ...current, [key]: value }));
  }

  async function prepareMedia(files, prepare, toAttachment) {
    if (!files.length) return;
    setError(null);
    setPendingAttachments((current) => current + files.length);
    const settled = await Promise.allSettled(files.map(prepare));
    const ready = [];
    const errors = [];
    settled.forEach((result, index) => {
      if (result.status === "fulfilled") {
        ready.push(toAttachment(result.value, files[index]));
      } else {
        errors.push(String(result.reason?.message ?? result.reason));
      }
    });
    if (ready.length) setMediaAttachments((current) => [...current, ...ready]);
    if (errors.length) setError(errors[0]);
    setPendingAttachments((current) => Math.max(0, current - files.length));
  }

  function addImageFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file && file.type.startsWith("image/"));
    prepareMedia(files, normalizeImageAttachment, (url, file) => ({
      type: "image",
      url,
      name: file.name || "image",
    }));
  }

  function addVideoFiles(fileList) {
    const files = Array.from(fileList || []).filter((file) => file && file.type.startsWith("video/"));
    prepareMedia(files, sampleVideoAttachment, (sampled, file) => ({
      type: "video",
      name: file.name || "video",
      ...sampled,
    }));
  }

  function addMediaUrl() {
    const source = mediaUrl.trim();
    if (!source) return;
    try {
      const parsed = new URL(source);
      if (!matchesHttpUrl(parsed)) throw new Error("Media URL must use http or https.");
    } catch (cause) {
      setError(String(cause?.message ?? "Enter a valid media URL."));
      return;
    }
    const type = mediaUrlKind;
    const capable = type === "image" ? visionCapable : videoCapable;
    if (!capable) {
      setError(`${type === "image" ? "Image" : "Video"} input is not supported by the loaded model.`);
      return;
    }
    setMediaUrl("");
    const sourceName = source.split("/").pop() || `${type} URL`;
    prepareMedia(
      [source],
      type === "image" ? normalizeImageAttachment : sampleVideoAttachment,
      (prepared) =>
        type === "image"
          ? { type: "image", url: prepared, name: sourceName }
          : { type: "video", name: sourceName, ...prepared },
    );
  }

  return (
    <section className="chat-layout">
      <div className="panel chat-panel">
        <div className="panel-head chat-head">
          <div>
            <p className="eyebrow">Streaming chat</p>
            <h2>{engineStatus?.loaded ? engineStatus.loaded.name : "Load a model to chat"}</h2>
            <p className="view-copy">Dogfoods {apiBase}/v1/chat/completions over SSE.</p>
          </div>
          <span className={serverStatus?.running ? "status-pill" : "status-pill warning"}>
            <StatusDot ok={Boolean(serverStatus?.running)} />
            {serverStatus?.running ? "API online" : "API offline"}
          </span>
        </div>

        <div className="message-list" aria-live="polite">
          {messages.length ? (
            messages.map((message, index) => {
              const hasToolCalls = Boolean(message.tool_calls && message.tool_calls.length);
              const media = messageMedia(message);
              return (
                <article className={`message-bubble ${message.role}`} key={`${message.role}-${index}`}>
                  <div className="message-role">{message.role}</div>
                  {media.length ? (
                    <div className="message-images">
                      {media.map((item, mediaIndex) =>
                        item.type === "video" ? (
                          <img
                            key={mediaIndex}
                            className="message-image"
                            src={item.frames?.[0]}
                            alt={`video ${mediaIndex + 1} (${item.frames?.length ?? 0} frames)`}
                            title={`${item.frames?.length ?? 0} sampled frames`}
                          />
                        ) : (
                          <img
                            key={mediaIndex}
                            className="message-image"
                            src={item.url}
                            alt={`attachment ${mediaIndex + 1}`}
                          />
                        ),
                      )}
                    </div>
                  ) : null}
                  {message.role === "tool" ? (
                    <ToolResult message={message} />
                  ) : (
                    <>
                      {message.content || !hasToolCalls ? (
                        <MessageContent
                          content={message.content}
                          thinking={message.thinking}
                          stripThinking={thinkingCapable && params.disableThinking}
                        />
                      ) : null}
                      {hasToolCalls ? <ToolCallList calls={message.tool_calls} /> : null}
                    </>
                  )}
                  <MessageActions message={message} index={index} onRewind={handleRewind} busy={busy} />
                </article>
              );
            })
          ) : (
            <div className="empty-panel">Ask a question to start a multi-turn chat with the served model.</div>
          )}
        </div>

        {pendingApproval ? (
          <div className="tool-approval" role="alertdialog" aria-label="Approve tool calls">
            <p className="tool-approval-title">The model wants to run a tool. Approve to execute it locally.</p>
            {pendingApproval.calls.map((call, index) => (
              <div className="tool-approval-row" key={index}>
                <code className="tool-call-sig">
                  {call.name}({formatToolArguments(call.arguments)})
                </code>
                {pendingApproval.decisions[index] === null ? (
                  <span className="tool-approval-actions">
                    <button className="primary-btn" type="button" onClick={() => decideApproval(index, true)}>
                      Approve
                    </button>
                    <button className="ghost-btn" type="button" onClick={() => decideApproval(index, false)}>
                      Deny
                    </button>
                  </span>
                ) : (
                  <span className="tool-approval-decided">
                    {pendingApproval.decisions[index] ? "Approved" : "Denied"}
                  </span>
                )}
              </div>
            ))}
          </div>
        ) : null}

        {error ? <p className="form-error" role="alert">{error}</p> : null}

        <form className="composer" onSubmit={handleSubmit}>
          {mediaAttachments.length ? (
            <div className="composer-attachments">
              {mediaAttachments.map((item, index) => (
                <div className={`composer-thumb ${item.type === "video" ? "composer-thumb-video" : ""}`} key={index}>
                  <img src={item.type === "video" ? item.frames?.[0] : item.url} alt={`${item.type} ${index + 1}`} />
                  {item.type === "video" ? <span className="composer-thumb-badge">{item.frames?.length ?? 0}f</span> : null}
                  <button
                    type="button"
                    aria-label={`Remove ${item.type}`}
                    onClick={() => setMediaAttachments((current) => current.filter((_, i) => i !== index))}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          ) : null}
          <textarea
            disabled={!engineStatus?.loaded || busy}
            onChange={(event) => setDraft(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            onPaste={
              visionCapable
                ? (event) => {
                    const files = Array.from(event.clipboardData?.items ?? [])
                      .filter((item) => item.type.startsWith("image/"))
                      .map((item) => item.getAsFile());
                    if (files.length) {
                      event.preventDefault();
                      addImageFiles(files);
                    }
                  }
                : undefined
            }
            placeholder={engineStatus?.loaded ? "Message the local model…" : "Load a model from Models first"}
            rows={3}
            value={draft}
          />
          <div className="composer-actions">
            {visionCapable ? (
              <label className="ghost-btn" title="Attach image">
                <input
                  type="file"
                  accept="image/*"
                  multiple
                  style={{ display: "none" }}
                  disabled={!engineStatus?.loaded || busy}
                  onChange={(event) => {
                    addImageFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                {pendingAttachments ? "Preparing..." : "Image"}
              </label>
            ) : null}
            {videoCapable ? (
              <label className="ghost-btn" title="Attach video (sampled into frames)">
                <input
                  type="file"
                  accept="video/*"
                  multiple
                  style={{ display: "none" }}
                  disabled={!engineStatus?.loaded || busy}
                  onChange={(event) => {
                    addVideoFiles(event.target.files);
                    event.target.value = "";
                  }}
                />
                {pendingAttachments ? "Preparing..." : "Video"}
              </label>
            ) : null}
            {visionCapable || videoCapable ? (
              <span className="composer-url-input">
                <select
                  aria-label="Media URL type"
                  disabled={busy}
                  onChange={(event) => setMediaUrlKind(event.target.value)}
                  value={mediaUrlKind}
                >
                  {visionCapable ? <option value="image">Image URL</option> : null}
                  {videoCapable ? <option value="video">Video URL</option> : null}
                </select>
                <input
                  aria-label="Media URL"
                  disabled={busy}
                  onChange={(event) => setMediaUrl(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      addMediaUrl();
                    }
                  }}
                  placeholder="https://…"
                  type="url"
                  value={mediaUrl}
                />
                <button className="ghost-btn" disabled={!mediaUrl.trim() || busy} onClick={addMediaUrl} type="button">
                  Add URL
                </button>
              </span>
            ) : null}
            <button className="primary-btn" disabled={!canSend} type="submit">
              {busy ? "Streaming…" : "Send"}
            </button>
            {busy ? (
              <button className="ghost-btn" type="button" onClick={handleStop}>
                Stop
              </button>
            ) : null}
          </div>
        </form>
      </div>

      <aside className="panel chat-settings">
        <div className="panel-head">
          <p className="eyebrow">Conversation overrides</p>
          <h2>Sampling</h2>
          <p className="view-copy">Applies only to this chat session.</p>
        </div>
        <div className="field">
          <label htmlFor="system-prompt">System prompt</label>
          <textarea
            id="system-prompt"
            onChange={(event) => updateParam("systemPrompt", event.target.value)}
            rows={5}
            value={params.systemPrompt}
          />
        </div>
        <div className="field-grid">
          <div className="field">
            <label htmlFor="temperature">Temperature</label>
            <input
              id="temperature"
              inputMode="decimal"
              onChange={(event) => updateParam("temperature", event.target.value)}
              type="number"
              step="0.1"
              min="0"
              max="2"
              value={params.temperature}
            />
          </div>
          <div className="field">
            <label htmlFor="top-p">Top P</label>
            <input
              id="top-p"
              inputMode="decimal"
              onChange={(event) => updateParam("topP", event.target.value)}
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={params.topP}
            />
          </div>
          <div className="field">
            <label htmlFor="max-tokens">Max tokens</label>
            <input
              id="max-tokens"
              inputMode="numeric"
              onChange={(event) => updateParam("maxTokens", event.target.value)}
              type="number"
              step="1"
              min="1"
              value={params.maxTokens}
            />
          </div>
        </div>
        {thinkingCapable ? (
          <label className="toggle-row">
            <input
              checked={params.disableThinking}
              onChange={(event) => updateParam("disableThinking", event.target.checked)}
              type="checkbox"
            />
            <span>
              Disable thinking
              <small>No-think request flag and hidden &lt;think&gt; output.</small>
            </span>
          </label>
        ) : null}
        <GenerationControls params={params} onChange={updateParam}
          capabilities={engineStatus?.loaded?.provider?.capabilities ?? {}} />
        {toolsCapable ? (
          <label className="toggle-row">
            <input
              checked={toolsEnabled}
              onChange={(event) => setToolsEnabled(event.target.checked)}
              type="checkbox"
            />
            <span>
              Enable tools
              <small>
                Offer {toolSpecs.length} built-in tool{toolSpecs.length === 1 ? "" : "s"}; each call needs your approval.
              </small>
            </span>
          </label>
        ) : null}
        <button className="ghost-btn" disabled={busy || !messages.length} onClick={startNewChat} type="button">
          Clear conversation
        </button>
      </aside>
    </section>
  );
}
