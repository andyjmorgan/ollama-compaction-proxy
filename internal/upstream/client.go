// Package upstream is the HTTP client for the Ollama compat endpoints, plus
// the SSE plumbing shared by every streaming path.
package upstream

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// Client talks to one Ollama instance.
type Client struct {
	baseURL string
	json    *http.Client // bounded timeout for non-streaming calls
	stream  *http.Client // no overall timeout: streams are long-lived
}

// New returns a Client for the Ollama instance at baseURL.
func New(baseURL string, jsonTimeout time.Duration) *Client {
	return &Client{
		baseURL: strings.TrimRight(baseURL, "/"),
		json:    &http.Client{Timeout: jsonTimeout},
		stream:  &http.Client{},
	}
}

// Response is a completed non-streaming upstream exchange.
type Response struct {
	StatusCode int
	Header     http.Header
	Body       []byte
}

// PostJSON forwards body to path and reads the whole response.
func (c *Client) PostJSON(ctx context.Context, path string, body []byte) (*Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.json.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call upstream: %w", err)
	}
	defer resp.Body.Close()

	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read upstream response: %w", err)
	}
	return &Response{StatusCode: resp.StatusCode, Header: resp.Header, Body: data}, nil
}

// StreamResponse is an in-flight streaming upstream exchange. The caller must
// Close it.
type StreamResponse struct {
	StatusCode int
	Header     http.Header

	// Frames reads the SSE stream. Nil when StatusCode is not 200 — read
	// ErrorBody instead.
	Frames *FrameReader

	// ErrorBody holds the full body of a non-200 response.
	ErrorBody []byte

	body io.Closer
}

// Close releases the upstream connection.
func (s *StreamResponse) Close() {
	if s.body != nil {
		s.body.Close()
	}
}

// PostStream forwards body to path expecting an SSE response. A non-200
// status is returned with its body fully read (upstreams send plain JSON
// errors even on would-be streaming requests).
func (c *Client) PostStream(ctx context.Context, path string, body []byte) (*StreamResponse, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.baseURL+path, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("build upstream request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	resp, err := c.stream.Do(req)
	if err != nil {
		return nil, fmt.Errorf("call upstream: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		data, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return &StreamResponse{StatusCode: resp.StatusCode, Header: resp.Header, ErrorBody: data}, nil
	}

	return &StreamResponse{
		StatusCode: resp.StatusCode,
		Header:     resp.Header,
		Frames:     NewFrameReader(resp.Body),
		body:       resp.Body,
	}, nil
}
