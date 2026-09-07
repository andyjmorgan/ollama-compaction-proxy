package openaiproxy

import (
	"encoding/json"
	"fmt"

	"github.com/andyjmorgan/slipspace-gateway/models"
	"github.com/andyjmorgan/slipspace-gateway/protocols/openai/responses"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
)

// compactionResult carries a completed summarization to response synthesis.
type compactionResult struct {
	summary string
	blob    string
	itemID  string
	usage   summarize.Result
}

// newCompactionResult mints the output item id alongside the blob.
func newCompactionResult(summary, blob string, usage summarize.Result) *compactionResult {
	return &compactionResult{summary: summary, blob: blob, itemID: httpapi.MintID("ci"), usage: usage}
}

// outputItem builds the wire compaction output item.
func (c *compactionResult) outputItem() responses.OutputItem {
	encrypted, _ := json.Marshal(c.blob)
	return responses.OutputItem{
		Type:             "compaction",
		ID:               c.itemID,
		EncryptedContent: encrypted,
		DynamicProperties: models.DynamicProperties{
			Extra: map[string]json.RawMessage{
				"created_by": json.RawMessage("null"),
			},
		},
	}
}

// outputItemJSON is the raw form used in stream events.
func (c *compactionResult) outputItemJSON(withContent bool) json.RawMessage {
	m := map[string]any{
		"type":       "compaction",
		"id":         c.itemID,
		"created_by": nil,
	}
	if withContent {
		m["encrypted_content"] = c.blob
	} else {
		m["encrypted_content"] = nil
	}
	raw, _ := json.Marshal(m)
	return raw
}

// compactionUsage is the namespaced extension reporting summarizer spend.
func (c *compactionResult) compactionUsage() json.RawMessage {
	raw, _ := json.Marshal(map[string]int{
		"input_tokens":  c.usage.InputTokens,
		"output_tokens": c.usage.OutputTokens,
	})
	return raw
}

// synthesizeResponse rewrites an upstream non-streaming response: minted id,
// compaction item prepended to output, compaction_usage extension attached.
func synthesizeResponse(upstreamBody []byte, c *compactionResult) ([]byte, error) {
	var resp responses.ResponsesResponse
	if err := json.Unmarshal(upstreamBody, &resp); err != nil {
		return nil, fmt.Errorf("parse upstream response: %w", err)
	}

	resp.ID = httpapi.MintID("resp")
	resp.Output = append([]responses.OutputItem{c.outputItem()}, resp.Output...)
	if resp.Extra == nil {
		resp.Extra = map[string]json.RawMessage{}
	}
	resp.Extra["compaction_usage"] = c.compactionUsage()

	return json.Marshal(resp)
}
