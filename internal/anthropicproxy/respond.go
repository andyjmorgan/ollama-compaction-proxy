package anthropicproxy

import (
	"encoding/json"
	"fmt"

	"github.com/andyjmorgan/slipspace-gateway/models"
	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
)

// compactionResult is a completed summarization plus its minted blob, carried
// from the trigger step to response synthesis.
type compactionResult struct {
	summary string
	blob    string
	usage   summarize.Result
}

// compactionBlock builds the wire content block for a completed compaction.
// The block registry upstream is sealed, so this is an UnknownBlock — which
// marshals to exactly {"type":"compaction","content":...,"encrypted_content":...}.
func compactionBlock(c *compactionResult) messages.ContentBlock {
	content, _ := json.Marshal(c.summary)
	encrypted, _ := json.Marshal(c.blob)
	return &messages.UnknownBlock{
		Type: "compaction",
		DynamicProperties: models.DynamicProperties{
			Extra: map[string]json.RawMessage{
				"content":           content,
				"encrypted_content": encrypted,
			},
		},
	}
}

// iterations builds usage.iterations for a compaction turn. Top-level tokens
// exclude the compaction iteration, per the contract.
func iterations(c *compactionResult, model string, msgUsage *messages.Usage) []messages.IterationUsage {
	its := []messages.IterationUsage{{
		Type:         "compaction",
		InputTokens:  c.usage.InputTokens,
		OutputTokens: c.usage.OutputTokens,
	}}
	if msgUsage != nil {
		its = append(its, messages.IterationUsage{
			Type:         "message",
			Model:        model,
			InputTokens:  msgUsage.InputTokens,
			OutputTokens: msgUsage.OutputTokens,
		})
	}
	return its
}

// synthesizeResponse rewrites an upstream non-streaming response body:
// minted outer id, compaction block prepended, usage.iterations attached.
func synthesizeResponse(upstreamBody []byte, c *compactionResult) ([]byte, error) {
	var resp messages.MessagesResponse
	if err := json.Unmarshal(upstreamBody, &resp); err != nil {
		return nil, fmt.Errorf("parse upstream response: %w", err)
	}

	resp.ID = httpapi.MintID("msg")
	resp.Content = append([]messages.ContentBlock{compactionBlock(c)}, resp.Content...)
	resp.Usage.Iterations = iterations(c, resp.Model, &resp.Usage)

	return json.Marshal(resp)
}

// pauseResponse builds the block-only response for pause_after_compaction.
func pauseResponse(model string, c *compactionResult) ([]byte, error) {
	stop := "compaction"
	resp := messages.MessagesResponse{
		ID:         httpapi.MintID("msg"),
		Type:       "message",
		Role:       "assistant",
		Model:      model,
		Content:    []messages.ContentBlock{compactionBlock(c)},
		StopReason: &stop,
		Usage: messages.Usage{
			InputTokens:  0,
			OutputTokens: 0,
			Iterations:   iterations(c, model, nil),
		},
	}
	return json.Marshal(resp)
}
