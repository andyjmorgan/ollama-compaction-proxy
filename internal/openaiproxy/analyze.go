// Package openaiproxy serves the OpenAI Responses dialect with native
// context compaction in front of Ollama's /v1/responses compat endpoint.
package openaiproxy

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/andyjmorgan/slipspace-gateway/protocols/openai/responses"
)

// contextManagementEntry is one element of the request's context_management
// array. Slipspace doesn't model the param (it lands in DynamicProperties),
// so this is our own shape.
type contextManagementEntry struct {
	Type             string `json:"type"`
	CompactThreshold *int   `json:"compact_threshold,omitempty"`
}

// item is one scanned input item: its raw bytes plus the peeked fields the
// proxy routes on.
type item struct {
	raw  json.RawMessage
	typ  string
	role string
}

// isUserMessage reports whether the item survives compaction verbatim.
func (it item) isUserMessage() bool {
	return (it.typ == "message" || it.typ == "") && it.role == "user"
}

// analysis is everything the handler needs to route one request.
type analysis struct {
	items []item

	// stringInput is set when input was a bare string (no items possible).
	stringInput bool

	// compactionEntry is the request's compaction config, nil if none.
	compactionEntry *contextManagementEntry
	threshold       int

	// lastCompaction is the index in items of the last round-tripped
	// compaction item, -1 if none.
	lastCompaction int
	incomingBlob   string

	// forced reports a compaction_trigger item (validated final).
	forced bool
}

// analyze inspects a parsed request. Returns an error for contract
// violations that must 400.
func analyze(req *responses.ResponsesRequest, defaultThreshold, minTrigger int) (*analysis, error) {
	a := &analysis{lastCompaction: -1}

	if raw, ok := req.Extra["context_management"]; ok {
		var entries []contextManagementEntry
		if err := json.Unmarshal(raw, &entries); err != nil {
			return nil, fmt.Errorf("invalid context_management: %w", err)
		}
		for i := range entries {
			if entries[i].Type == "compaction" {
				a.compactionEntry = &entries[i]
				break
			}
		}
	}
	if a.compactionEntry != nil {
		a.threshold = defaultThreshold
		if a.compactionEntry.CompactThreshold != nil && *a.compactionEntry.CompactThreshold > 0 {
			a.threshold = *a.compactionEntry.CompactThreshold
		}
		a.threshold = max(a.threshold, minTrigger)
	}

	if len(req.Input) == 0 {
		a.stringInput = true
		return a, nil
	}
	var s string
	if err := json.Unmarshal(req.Input, &s); err == nil {
		a.stringInput = true
		return a, nil
	}

	var rawItems []json.RawMessage
	if err := json.Unmarshal(req.Input, &rawItems); err != nil {
		return nil, fmt.Errorf("input must be a string or an array of items")
	}

	for i, raw := range rawItems {
		var peek struct {
			Type string `json:"type"`
			Role string `json:"role"`
		}
		if err := json.Unmarshal(raw, &peek); err != nil {
			return nil, fmt.Errorf("input[%d]: not an object", i)
		}
		it := item{raw: raw, typ: peek.Type, role: peek.Role}
		a.items = append(a.items, it)

		switch peek.Type {
		case "compaction":
			var c struct {
				EncryptedContent string `json:"encrypted_content"`
			}
			if err := json.Unmarshal(raw, &c); err != nil || strings.TrimSpace(c.EncryptedContent) == "" {
				return nil, fmt.Errorf("input[%d]: compaction item missing encrypted_content", i)
			}
			a.lastCompaction = i
			a.incomingBlob = strings.TrimSpace(c.EncryptedContent)
		case "compaction_trigger":
			if i != len(rawItems)-1 {
				return nil, fmt.Errorf("compaction_trigger must be the final input item")
			}
			a.forced = true
		}
	}

	return a, nil
}
