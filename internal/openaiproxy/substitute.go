package openaiproxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
)

// summaryItem builds the synthetic user message item carrying a summary.
func summaryItem(summary string) item {
	raw, _ := json.Marshal(map[string]any{
		"type": "message",
		"role": "user",
		"content": []map[string]string{
			{"type": "input_text", "text": summarize.WrapSummary(summary)},
		},
	})
	return item{raw: raw, typ: "message", role: "user"}
}

// substituteIncoming applies the round-tripped compaction item at index idx:
// before it only user messages survive (verbatim raw bytes), the summary item
// takes its position, and everything after survives verbatim. compaction /
// compaction_trigger items in the tail are NOT handled here — strip() runs
// after.
func substituteIncoming(items []item, idx int, summary string) []item {
	var out []item
	for _, it := range items[:idx] {
		if it.isUserMessage() {
			out = append(out, it)
		}
	}
	out = append(out, summaryItem(summary))
	out = append(out, items[idx+1:]...)
	return out
}

// strip removes items Ollama's /v1/responses rejects as unknown types.
func strip(items []item) []item {
	out := items[:0:0]
	for _, it := range items {
		switch it.typ {
		case "compaction", "compaction_trigger":
			// Consumed by the proxy; Ollama 400s on them.
		default:
			out = append(out, it)
		}
	}
	return out
}

// splitForCompaction partitions items for a fresh compaction: earlier user
// messages are kept verbatim, the contiguous tail from the last user message
// onward is kept verbatim (so a live tool loop is never severed), and
// everything is summarized for context. ok=false when there is nothing worth
// compacting.
func splitForCompaction(items []item) (keepUsers, tail []item, ok bool) {
	lastUser := -1
	for i := range items {
		if items[i].isUserMessage() {
			lastUser = i
		}
	}

	tailStart := len(items) - 1 // no user item: keep just the final item live
	if lastUser >= 0 {
		tailStart = lastUser
	}
	if tailStart <= 0 {
		return nil, nil, false // nothing precedes the live tail
	}

	for _, it := range items[:tailStart] {
		if it.isUserMessage() {
			keepUsers = append(keepUsers, it)
		}
	}
	return keepUsers, items[tailStart:], true
}

// rebuildInput assembles the forwarded input array after a fresh compaction.
func rebuildInput(keepUsers []item, summary string, tail []item) []item {
	out := append([]item{}, keepUsers...)
	out = append(out, summaryItem(summary))
	out = append(out, tail...)
	return out
}

// marshalItems re-serializes items as the input array.
func marshalItems(items []item) json.RawMessage {
	raws := make([]json.RawMessage, len(items))
	for i, it := range items {
		raws[i] = it.raw
	}
	out, _ := json.Marshal(raws)
	return out
}

// transcript flattens items into plain text for the summarizer.
func transcript(items []item) string {
	var b strings.Builder
	for _, it := range items {
		switch it.typ {
		case "message", "":
			var m struct {
				Role    string          `json:"role"`
				Content json.RawMessage `json:"content"`
			}
			if err := json.Unmarshal(it.raw, &m); err != nil {
				continue
			}
			fmt.Fprintf(&b, "%s: %s\n\n", strings.ToUpper(m.Role), contentText(m.Content))
		case "function_call":
			var fc struct {
				Name      string `json:"name"`
				Arguments string `json:"arguments"`
			}
			_ = json.Unmarshal(it.raw, &fc)
			fmt.Fprintf(&b, "[tool call %s(%s)]\n\n", fc.Name, fc.Arguments)
		case "function_call_output":
			var fo struct {
				Output json.RawMessage `json:"output"`
			}
			_ = json.Unmarshal(it.raw, &fo)
			fmt.Fprintf(&b, "[tool result: %s]\n\n", contentText(fo.Output))
		case "reasoning":
			// Reasoning is not part of the durable conversation.
		default:
			fmt.Fprintf(&b, "[%s item: %s]\n\n", it.typ, it.raw)
		}
	}
	return strings.TrimSpace(b.String())
}

// contentText reduces a message content value (string or part array) to text.
func contentText(raw json.RawMessage) string {
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	}
	if err := json.Unmarshal(raw, &parts); err == nil {
		var out []string
		for _, p := range parts {
			if p.Text != "" {
				out = append(out, p.Text)
			}
		}
		return strings.Join(out, "\n")
	}
	return string(raw)
}
