package anthropicproxy

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"
)

// Models exposes configured serving limits, not the model family's theoretical
// maximum. Claude's generic gateway discovery ignores these fields; SDK clients
// can use this catalog to configure their runtime before starting a session.
func (h *Handler) Models(w http.ResponseWriter, r *http.Request) {
	type row struct {
		ID              string `json:"id"`
		Type            string `json:"type"`
		DisplayName     string `json:"display_name"`
		ContextWindow   int    `json:"context_window"`
		MaxInputTokens  int    `json:"max_input_tokens"`
		MaxTokens       int    `json:"max_tokens"`
		CompactAt       int    `json:"compact_at_input_tokens"`
		SummaryThinking string `json:"summary_thinking,omitempty"`
	}
	rows := []row{}
	for model, p := range h.cfg.ClaudeModels {
		rows = append(rows, row{h.cfg.ModelPrefix + model, "model", model, p.ContextWindow, p.ContextWindow - p.MaxOutputTokens, p.MaxOutputTokens, p.CompactAt, p.SummaryThinking})
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].ID < rows[j].ID })
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]any{"data": rows, "has_more": false, "compaction_owner": "claude", "policy_header": "X-Claude-Compaction: native"})
}

// claudeSummary recognizes the terminal user instruction emitted by the pinned
// Claude runtime (2.1.269). This is protocol compatibility, not authentication:
// summary requests still obey the physical context and output reserve limits.
func claudeSummary(body []byte) bool {
	var req struct {
		Messages []struct {
			Role    string          `json:"role"`
			Content json.RawMessage `json:"content"`
		} `json:"messages"`
	}
	if json.Unmarshal(body, &req) != nil {
		return false
	}
	for i := len(req.Messages) - 1; i >= 0; i-- {
		m := req.Messages[i]
		if m.Role != "user" {
			continue
		}
		var text string
		if json.Unmarshal(m.Content, &text) != nil {
			var blocks []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}
			if json.Unmarshal(m.Content, &blocks) != nil {
				return false
			}
			for _, b := range blocks {
				if b.Type == "text" {
					text = b.Text
				}
			}
		}
		return strings.HasPrefix(text, "CRITICAL: Respond with TEXT ONLY. Do NOT call any tools.") && strings.Contains(text, "Your task is to create a detailed summary of the conversation so far")
	}
	return false
}

// enforceClaudeBudget returns true when the request has been handled locally.
// The standard Anthropic overflow error invokes Claude's own reactive compact
// path, including for native subagents that inherit their parent's window.
func (h *Handler) enforceClaudeBudget(w http.ResponseWriter, r *http.Request, req *messages.MessagesRequest, body []byte) bool {
	if r.Header.Get("X-Claude-Compaction") != "native" {
		return false
	}
	p, ok := h.cfg.ClaudeModels[req.Model]
	if !ok {
		httpapi.WriteError(w, httpapi.DialectAnthropic, 400, "no Claude compaction policy for model")
		return true
	}
	// Claude may omit thinking configuration for unrecognized model names.
	// Preserve the application's explicit choice across that compatibility gap.
	if r.Header.Get("X-Ollama-Thinking") == "disabled" || (r.Header.Get("X-Ollama-Summary-Thinking") == "disabled" && claudeSummary(body)) {
		req.Thinking = &messages.ThinkingConfig{Type: "disabled"}
	}
	summary := claudeSummary(body)
	if summary && p.SummaryThinking != "" {
		// Per-model summary policy overrides the client default on native
		// compaction requests only. Count using the mode actually forwarded.
		req.Thinking = &messages.ThinkingConfig{Type: p.SummaryThinking}
	}
	if summary {
		// Keep Claude's summary protocol, adding a model-neutral retention focus.
		// Count the augmented request, including this instruction, before forwarding.
		for i := len(req.Messages) - 1; i >= 0; i-- {
			m := &req.Messages[i]
			if m.Role != "user" {
				continue
			}
			const focus = "\n\nRetention requirement: Begin your summary with an explicit VERIFIED FACTS section. Copy the exact values of facts the user asked you to remember from earlier tool results, including names, identifiers, numbers and codewords. Do not merely say that facts exist or must be reported. Then list completed work and the next action. Omit repetitive bulk data. Never invent a missing value."
			if text, ok := m.ContentAsString(); ok {
				_ = m.SetContentString(text + focus)
			} else if blocks, ok := m.ContentAsBlocks(); ok {
				for j := len(blocks) - 1; j >= 0; j-- {
					if text, ok := blocks[j].(*messages.TextBlock); ok {
						text.Text += focus
						break
					}
				}
				_ = m.SetContentBlocks(blocks)
			}
			break
		}
	}
	count, err := h.countRequest(r.Context(), req)
	if err != nil {
		httpapi.WriteError(w, httpapi.DialectAnthropic, 503, "token counting unavailable; refusing unchecked context")
		return true
	}
	limit := p.CompactAt
	if summary {
		limit = p.ContextWindow - p.MaxOutputTokens
	}
	if count >= limit {
		h.log.Info("claude context budget reached", "session_id", r.Header.Get("X-Claude-Code-Session-Id"), "model", req.Model, "input_tokens", count, "limit", limit, "summary", summary)
		// Omit a numerical token-gap hint: the policy limit is deliberately below
		// the physical window. A gap hint can cause Claude to truncate history before
		// it even attempts summarization. Token counts remain accurate in logs.
		httpapi.WriteError(w, httpapi.DialectAnthropic, http.StatusBadRequest, fmt.Sprintf("prompt is too long for the configured %d input-token budget; compact the conversation", limit))
		return true
	}
	if req.MaxTokens > p.MaxOutputTokens {
		req.MaxTokens = p.MaxOutputTokens
	}
	h.log.Info("claude context budget checked", "session_id", r.Header.Get("X-Claude-Code-Session-Id"), "model", req.Model, "input_tokens", count, "limit", limit, "summary", summary)
	return false
}
