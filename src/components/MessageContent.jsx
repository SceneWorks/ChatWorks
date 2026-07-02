import { Markdown } from "@sceneworks/ui";
import { stripThinkBlocks } from "../api/sse";

export function MessageContent({ content, thinking, stripThinking }) {
  const visibleContent = stripThinking ? stripThinkBlocks(content) : content;
  if (!visibleContent.trim()) return <p className="thinking-hidden">Thinking hidden.</p>;
  return (
    <>
      {!stripThinking && thinking ? (
        <details className="thinking-block">
          <summary>Reasoning</summary>
          <Markdown content={thinking} />
        </details>
      ) : null}
      <Markdown content={visibleContent} />
    </>
  );
}
