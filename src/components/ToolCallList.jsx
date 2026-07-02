import { parseToolArguments } from "../api/sse";

/// Pretty-print a tool call's arguments (a JSON string or object) for display.
export function formatToolArguments(args) {
  const value = parseToolArguments(args);
  const text = JSON.stringify(value);
  return text === "{}" ? "" : JSON.stringify(value, null, 0);
}

/// Render the tool calls an assistant turn requested.
export function ToolCallList({ calls }) {
  return (
    <div className="tool-calls">
      {calls.map((call, index) => (
        <div className="tool-call" key={index}>
          <span className="tool-call-icon" aria-hidden="true">🛠</span>
          <code className="tool-call-sig">
            {call.name}({formatToolArguments(call.arguments)})
          </code>
        </div>
      ))}
    </div>
  );
}

/// Render a tool-result turn (the executed output, an error, or a denial).
export function ToolResult({ message }) {
  const status = message.denied ? "denied" : message.isError ? "error" : "ok";
  return (
    <div className={`tool-result tool-result-${status}`}>
      <div className="tool-result-head">
        {message.name ? <code>{message.name}</code> : null}
        <span className="tool-result-tag">{status}</span>
      </div>
      <pre className="tool-result-body">{message.content}</pre>
    </div>
  );
}
