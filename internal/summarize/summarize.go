// Package summarize runs the compaction summarization call against Ollama.
package summarize

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

// Result is a completed summarization.
type Result struct {
	Summary      string
	Model        string // summarizer model actually used
	InputTokens  int
	OutputTokens int
}

// Summarizer produces conversation summaries via Ollama /v1/chat/completions.
type Summarizer struct {
	client *upstream.Client

	// Model overrides the summarizer model; empty means use the model passed
	// per call (the conversation's own model).
	Model   string
	Timeout time.Duration
}

// New returns a Summarizer using client.
func New(client *upstream.Client, model string, timeout time.Duration) *Summarizer {
	return &Summarizer{client: client, Model: model, Timeout: timeout}
}

// Summarize condenses transcript using instructions (the full replacement
// prompt) and the request model as fallback summarizer.
func (s *Summarizer) Summarize(ctx context.Context, requestModel, instructions, transcript string) (*Result, error) {
	model := s.Model
	if model == "" {
		model = requestModel
	}
	if strings.TrimSpace(instructions) == "" {
		instructions = DefaultInstructions
	}

	ctx, cancel := context.WithTimeout(ctx, s.Timeout)
	defer cancel()

	body, err := json.Marshal(map[string]any{
		"model": model,
		"messages": []map[string]string{
			{"role": "system", "content": instructions},
			{"role": "user", "content": "Compact the following conversation:\n\n" + transcript},
		},
		"stream": false,
		// Summarization never wants chain-of-thought: on thinking models (e.g.
		// qwen3.x) the reasoning preamble is generated at full cost and blows
		// SUMMARIZE_TIMEOUT, silently no-opping compaction. A no-op for models
		// that don't reason, so it is safe to send unconditionally.
		"reasoning_effort": "none",
	})
	if err != nil {
		return nil, fmt.Errorf("marshal summarize request: %w", err)
	}

	resp, err := s.client.PostJSON(ctx, "/v1/chat/completions", body)
	if err != nil {
		return nil, fmt.Errorf("summarize call: %w", err)
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("summarizer returned %d: %s", resp.StatusCode, truncate(resp.Body, 200))
	}

	var decoded struct {
		Choices []struct {
			Message struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
		Usage struct {
			PromptTokens     int `json:"prompt_tokens"`
			CompletionTokens int `json:"completion_tokens"`
		} `json:"usage"`
	}
	if err := json.Unmarshal(resp.Body, &decoded); err != nil {
		return nil, fmt.Errorf("decode summarizer response: %w", err)
	}
	if len(decoded.Choices) == 0 || strings.TrimSpace(decoded.Choices[0].Message.Content) == "" {
		return nil, fmt.Errorf("summarizer produced no content")
	}

	return &Result{
		Summary:      strings.TrimSpace(decoded.Choices[0].Message.Content),
		Model:        model,
		InputTokens:  decoded.Usage.PromptTokens,
		OutputTokens: decoded.Usage.CompletionTokens,
	}, nil
}

func truncate(b []byte, n int) string {
	if len(b) > n {
		return string(b[:n]) + "..."
	}
	return string(b)
}
