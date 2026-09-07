// Package anthropicproxy serves the Anthropic Messages dialect with native
// server-side compaction (beta compact-2026-01-12 semantics) in front of
// Ollama's /v1/messages compat endpoint.
package anthropicproxy

import (
	"encoding/json"
	"strings"

	"github.com/andyjmorgan/slipspace-gateway/protocols/anthropic/messages"
)

// incomingBlock locates the last compaction block a client round-tripped.
type incomingBlock struct {
	msgIdx   int
	blockIdx int

	// content is the block's plaintext summary: nil when the field was null
	// or absent (Anthropic's failed-compaction marker).
	content *string

	// encrypted is the opaque blob we minted on a previous turn.
	encrypted string
}

// analysis is everything the handler needs to route one request.
type analysis struct {
	// edit is the first compact_20260112 edit, nil when the request carries
	// none. Other edit types are accepted and ignored.
	edit *messages.CompactEdit

	// trigger is the resolved input-token trigger for edit.
	trigger int

	// incoming is the last round-tripped compaction block, nil if none.
	incoming *incomingBlock

	// hasNullContent reports a message with absent/null content, which
	// Ollama rejects with a 400 and we normalize instead.
	hasNullContent bool
}

// needsRewrite reports whether the request can be forwarded byte-for-byte.
func (a *analysis) needsRewrite() bool {
	return a.edit != nil || a.incoming != nil || a.hasNullContent
}

// analyze inspects a parsed request. defaultTrigger and minTrigger come from
// config.
func analyze(req *messages.MessagesRequest, defaultTrigger, minTrigger int) *analysis {
	a := &analysis{}

	if req.ContextManagement != nil {
		for _, edit := range req.ContextManagement.Edits {
			if ce, ok := edit.(*messages.CompactEdit); ok {
				a.edit = ce
				break
			}
		}
	}
	if a.edit != nil {
		a.trigger = resolveTrigger(a.edit.Trigger, defaultTrigger, minTrigger)
	}

	for mi := range req.Messages {
		raw := req.Messages[mi].Content
		if len(raw) == 0 || string(raw) == "null" {
			a.hasNullContent = true
			continue
		}
		blocks, ok := req.Messages[mi].ContentAsBlocks()
		if !ok {
			continue
		}
		for bi, block := range blocks {
			if block.BlockType() != "compaction" {
				continue
			}
			ub, ok := block.(*messages.UnknownBlock)
			if !ok {
				continue
			}
			ib := &incomingBlock{msgIdx: mi, blockIdx: bi}
			if raw, ok := ub.Extra["content"]; ok && string(raw) != "null" {
				var s string
				if err := json.Unmarshal(raw, &s); err == nil {
					ib.content = &s
				}
			}
			if raw, ok := ub.Extra["encrypted_content"]; ok {
				var s string
				_ = json.Unmarshal(raw, &s)
				ib.encrypted = strings.TrimSpace(s)
			}
			// Last one wins, per the contract.
			a.incoming = ib
		}
	}

	return a
}

// resolveTrigger extracts the input_tokens trigger value, applying the default
// and the configured floor.
func resolveTrigger(raw json.RawMessage, defaultTrigger, minTrigger int) int {
	value := defaultTrigger
	if len(raw) > 0 {
		var t struct {
			Value int `json:"value"`
		}
		if err := json.Unmarshal(raw, &t); err == nil && t.Value > 0 {
			value = t.Value
		}
	}
	return max(value, minTrigger)
}
