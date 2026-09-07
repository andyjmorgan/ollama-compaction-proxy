package anthropicproxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/compactstate"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
)

// errStateInvalid marks a compaction block that carries neither a valid blob
// nor usable plaintext content — treated as Anthropic's failed-compaction
// no-op (block dropped, history kept).
var errStateInvalid = fmt.Errorf("compaction state unusable")

// resolveSummary applies the per-dialect trust policy to an incoming block:
// prefer the verified blob, fall back to the client-visible plaintext content,
// and report errStateInvalid when neither is usable.
func resolveSummary(ib *incomingBlock, key []byte) (summary string, blobInvalid bool, err error) {
	if ib.encrypted != "" {
		blob, derr := compactstate.Decode(ib.encrypted, key)
		if derr == nil {
			return blob.Summary, false, nil
		}
		blobInvalid = true
	}
	if ib.content != nil && strings.TrimSpace(*ib.content) != "" {
		// The plaintext field is client-visible by contract; honoring it
		// cannot escalate anything — a summary is just conversation text.
		return *ib.content, blobInvalid, nil
	}
	return "", blobInvalid, errStateInvalid
}

// substitute rewrites req.Messages so the compacted history is replaced by
// the summary: everything before the last compaction block is dropped, a
// synthetic user turn carrying the summary leads, and everything after the
// block survives verbatim. Returns whether a substitution actually happened.
func substitute(req *messages.MessagesRequest, ib *incomingBlock, summary string) error {
	var rebuilt []messages.Message

	rebuilt = append(rebuilt, summaryTurn(summary))

	// Remainder of the message containing the block: blocks after it.
	holder := req.Messages[ib.msgIdx]
	if blocks, ok := holder.ContentAsBlocks(); ok && ib.blockIdx+1 < len(blocks) {
		rest := messages.Message{Role: holder.Role}
		if err := rest.SetContentBlocks(blocks[ib.blockIdx+1:]); err != nil {
			return fmt.Errorf("rebuild post-compaction blocks: %w", err)
		}
		rebuilt = append(rebuilt, rest)
	}

	rebuilt = append(rebuilt, req.Messages[ib.msgIdx+1:]...)
	req.Messages = rebuilt
	return nil
}

// dropBlock removes just the compaction block (failed-compaction no-op path),
// keeping all history.
func dropBlock(req *messages.MessagesRequest, ib *incomingBlock) error {
	holder := &req.Messages[ib.msgIdx]
	blocks, ok := holder.ContentAsBlocks()
	if !ok {
		return nil
	}
	remaining := append(append([]messages.ContentBlock{}, blocks[:ib.blockIdx]...), blocks[ib.blockIdx+1:]...)
	if len(remaining) == 0 {
		req.Messages = append(req.Messages[:ib.msgIdx], req.Messages[ib.msgIdx+1:]...)
		return nil
	}
	return holder.SetContentBlocks(remaining)
}

// summaryTurn builds the synthetic user turn carrying the summary.
func summaryTurn(summary string) messages.Message {
	m := messages.Message{Role: "user"}
	_ = m.SetContentBlocks([]messages.ContentBlock{
		&messages.TextBlock{Type: "text", Text: summarize.WrapSummary(summary)},
	})
	return m
}

// normalize fixes request shapes Ollama rejects: absent/null message content
// becomes an empty string, and the context_management config is stripped.
func normalize(req *messages.MessagesRequest) {
	for i := range req.Messages {
		raw := req.Messages[i].Content
		if len(raw) == 0 || string(raw) == "null" {
			_ = req.Messages[i].SetContentString("")
		}
	}
	req.ContextManagement = nil
}

// splitForCompaction divides messages into the head to summarize and the tail
// to keep verbatim. The tail starts at the last real user turn — a user
// message that is not purely tool_results — so a live tool loop is never
// severed: summarizing away a tool_use while keeping its tool_result would
// orphan the result and the model would lose (and re-fetch) it. A request too
// short to compact returns ok=false.
func splitForCompaction(msgs []messages.Message) (head, tail []messages.Message, ok bool) {
	tailStart := len(msgs) - 1
	for i := len(msgs) - 1; i >= 0; i-- {
		if msgs[i].Role == "user" && !isToolResultOnly(msgs[i]) {
			tailStart = i
			break
		}
	}
	if tailStart <= 0 {
		return nil, nil, false
	}
	return msgs[:tailStart], msgs[tailStart:], true
}

// isToolResultOnly reports whether a message consists solely of tool_result
// blocks (the reply half of a tool loop).
func isToolResultOnly(m messages.Message) bool {
	blocks, ok := m.ContentAsBlocks()
	if !ok || len(blocks) == 0 {
		return false
	}
	for _, b := range blocks {
		if b.BlockType() != "tool_result" {
			return false
		}
	}
	return true
}

// transcript flattens messages into plain text for the summarizer.
func transcript(msgs []messages.Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(strings.ToUpper(m.Role))
		b.WriteString(": ")
		if s, ok := m.ContentAsString(); ok {
			b.WriteString(s)
			b.WriteString("\n\n")
			continue
		}
		blocks, ok := m.ContentAsBlocks()
		if !ok {
			b.WriteString("\n\n")
			continue
		}
		for _, block := range blocks {
			switch bl := block.(type) {
			case *messages.TextBlock:
				b.WriteString(bl.Text)
				b.WriteString("\n")
			case *messages.ToolUseBlock:
				fmt.Fprintf(&b, "[tool call %s(%s)]\n", bl.Name, bl.Input)
			case *messages.ToolResultBlock:
				fmt.Fprintf(&b, "[tool result: %s]\n", bl.Content)
			case *messages.ThinkingBlock:
				// Reasoning is not part of the durable conversation.
			default:
				// Unknown blocks: include their raw JSON so nothing the
				// client considers content silently vanishes.
				raw, err := json.Marshal(block)
				if err == nil {
					fmt.Fprintf(&b, "[%s block: %s]\n", block.BlockType(), raw)
				}
			}
		}
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}
