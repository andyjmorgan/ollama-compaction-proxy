// Package ollama is a thin client for the subset of the Ollama HTTP API this
// service needs: /api/show, which supplies both the model's rendering metadata
// and its GGUF tokenizer vocabulary.
package ollama

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
)

// ErrModelNotFound reports that Ollama does not have the requested model
// installed. Callers map this to a 404.
var ErrModelNotFound = errors.New("model not found")

// Client talks to a single Ollama instance.
type Client struct {
	baseURL string
	hc      *http.Client
}

// New returns a Client for the Ollama instance at baseURL.
func New(baseURL string, hc *http.Client) *Client {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &Client{baseURL: strings.TrimRight(baseURL, "/"), hc: hc}
}

// Show is the part of Ollama's /api/show response this service uses.
//
// ModelInfo is left as raw JSON because a verbose response carries the full
// tokenizer vocabulary: for gemma4 that is 262k tokens and 515k merges, ~12MB.
// Decoding it into map[string]any would box roughly 780k values for nothing;
// the tokenizer package unmarshals only the keys it needs.
type Show struct {
	Modelfile    string                     `json:"modelfile"`
	Template     string                     `json:"template"`
	System       string                     `json:"system"`
	Capabilities []string                   `json:"capabilities"`
	ModelInfo    map[string]json.RawMessage `json:"model_info"`
}

// Thinking reports whether the model supports thinking. Ollama turns thinking
// on by default for such models when a request does not say otherwise, and
// that changes what gets rendered, so it changes the token count.
func (s *Show) Thinking() bool {
	return slices.Contains(s.Capabilities, "thinking")
}

// Renderer returns the name of the Go-native renderer this model declares, or
// "" if it renders via its Go template instead.
//
// Ollama 0.32.8 defines a Renderer field on its ShowResponse type but does not
// populate it, so the name has to come from the RENDERER directive in the
// rendered modelfile. Models such as gemma4 and muse-glimmer pair a RENDERER
// with a stub "{{ .Prompt }}" template, so missing this means falling back to a
// template that renders nothing but the bare prompt.
func (s *Show) Renderer() string {
	for line := range strings.SplitSeq(s.Modelfile, "\n") {
		if name, ok := strings.CutPrefix(strings.TrimSpace(line), "RENDERER "); ok {
			return strings.TrimSpace(name)
		}
	}
	return ""
}

// ShowVerbose fetches model metadata including the full tokenizer vocabulary.
func (c *Client) ShowVerbose(ctx context.Context, model string) (*Show, error) {
	body, err := json.Marshal(map[string]any{"model": model, "verbose": true})
	if err != nil {
		return nil, fmt.Errorf("marshal show request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+"/api/show", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build show request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call ollama: %w", err)
	}
	defer resp.Body.Close()

	switch resp.StatusCode {
	case http.StatusOK:
	case http.StatusNotFound:
		return nil, fmt.Errorf("%q: %w", model, ErrModelNotFound)
	default:
		snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 512))
		return nil, fmt.Errorf("ollama /api/show returned %s: %s", resp.Status, bytes.TrimSpace(snippet))
	}

	var show Show
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return nil, fmt.Errorf("decode show response: %w", err)
	}
	return &show, nil
}
