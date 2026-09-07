// Package tokencount calls the ollama-tokenizer-service for exact input-token
// counts, used for compaction trigger decisions and /v1/messages/count_tokens.
package tokencount

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"time"
)

// Client talks to one tokenizer-service instance.
type Client struct {
	baseURL string
	hc      *http.Client
}

// New returns a Client for the tokenizer service at baseURL.
func New(baseURL string) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		// Cold tokenizer loads fetch a multi-MB vocabulary; warm calls are
		// sub-millisecond.
		hc: &http.Client{Timeout: 2 * time.Minute},
	}
}

// Count returns the exact input-token count for a request body. The body is
// any JSON object the tokenizer service's extractor understands — both the
// Anthropic and OpenAI wire shapes map directly onto its fields.
func (c *Client) Count(ctx context.Context, body []byte) (int, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/count-tokens", bytes.NewReader(body))
	if err != nil {
		return 0, fmt.Errorf("build count request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.hc.Do(req)
	if err != nil {
		return 0, fmt.Errorf("call tokenizer: %w", err)
	}
	defer resp.Body.Close()

	var decoded struct {
		Tokens int    `json:"tokens"`
		Error  string `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return 0, fmt.Errorf("decode tokenizer response: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return 0, fmt.Errorf("tokenizer returned %d: %s", resp.StatusCode, decoded.Error)
	}
	return decoded.Tokens, nil
}
