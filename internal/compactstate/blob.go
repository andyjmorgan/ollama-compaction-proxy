// Package compactstate encodes the proxy's compaction state into the opaque
// encrypted_content string both provider dialects round-trip.
//
// The encoding is integrity-protected, not encrypted: the payload is the
// client's own conversation summary, so confidentiality against the client
// buys nothing, while an HMAC lets the proxy reject tampered or foreign state.
package compactstate

import (
	"crypto/hmac"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// Blob is the state carried inside encrypted_content.
type Blob struct {
	V         int    `json:"v"`
	Kind      string `json:"kind"` // "anthropic" | "openai", informational
	Summary   string `json:"summary"`
	Model     string `json:"model"`     // conversation model at compaction time
	SumModel  string `json:"sum_model"` // summarizer model actually used
	CreatedAt int64  `json:"created_at"`
	InTokens  int    `json:"in_tokens"`
	OutTokens int    `json:"out_tokens"`
}

// ErrMalformed reports a blob that is not this proxy's encoding at all.
var ErrMalformed = errors.New("malformed compaction state")

// ErrBadMAC reports a blob whose signature does not verify — tampered, or
// minted under a different key.
var ErrBadMAC = errors.New("compaction state failed integrity check")

// Encode signs and serializes b.
func Encode(b Blob, key []byte) (string, error) {
	b.V = 1
	payload, err := json.Marshal(b)
	if err != nil {
		return "", fmt.Errorf("marshal blob: %w", err)
	}
	mac := sign(payload, key)
	return base64.RawURLEncoding.EncodeToString(append(append(payload, '.'), mac...)), nil
}

// Decode verifies and parses s.
func Decode(s string, key []byte) (Blob, error) {
	raw, err := base64.RawURLEncoding.DecodeString(strings.TrimSpace(s))
	if err != nil {
		return Blob{}, ErrMalformed
	}
	// The payload is JSON and cannot contain a bare '.', so the last dot
	// separates payload from MAC.
	dot := strings.LastIndexByte(string(raw), '.')
	if dot < 0 {
		return Blob{}, ErrMalformed
	}
	payload, mac := raw[:dot], raw[dot+1:]

	if !hmac.Equal(mac, sign(payload, key)) {
		return Blob{}, ErrBadMAC
	}

	var b Blob
	if err := json.Unmarshal(payload, &b); err != nil {
		return Blob{}, ErrMalformed
	}
	return b, nil
}

func sign(payload, key []byte) []byte {
	h := hmac.New(sha256.New, key)
	h.Write(payload)
	return []byte(hex.EncodeToString(h.Sum(nil)))
}
