package httpapi

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"net/http"
)

// Dialect selects which provider's error envelope a handler speaks.
type Dialect int

const (
	DialectAnthropic Dialect = iota
	DialectOpenAI
)

// anthropicErrorType maps an HTTP status to Anthropic's error.type value,
// mirroring the mapping Ollama's own compat layer uses.
func anthropicErrorType(status int) string {
	switch status {
	case http.StatusBadRequest:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusRequestEntityTooLarge:
		return "request_too_large"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case http.StatusServiceUnavailable, 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

func openaiErrorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge:
		return "invalid_request_error"
	case http.StatusNotFound:
		return "not_found_error"
	default:
		return "api_error"
	}
}

// WriteError emits msg in the dialect's error envelope with the given status.
func WriteError(w http.ResponseWriter, dialect Dialect, status int, msg string) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)

	switch dialect {
	case DialectAnthropic:
		_ = json.NewEncoder(w).Encode(map[string]any{
			"type": "error",
			"error": map[string]any{
				"type":    anthropicErrorType(status),
				"message": msg,
			},
			"request_id": "req_" + randHex(12),
		})
	default:
		_ = json.NewEncoder(w).Encode(map[string]any{
			"error": map[string]any{
				"message": msg,
				"type":    openaiErrorType(status),
				"param":   nil,
				"code":    nil,
			},
		})
	}
}

// randHex returns n random bytes as a hex string.
func randHex(n int) string {
	b := make([]byte, n)
	_, _ = rand.Read(b)
	return hex.EncodeToString(b)
}

// MintID returns a crypto-random wire ID with the given prefix, e.g.
// MintID("msg") -> "msg_5f0c...".
func MintID(prefix string) string {
	return prefix + "_" + randHex(12)
}
