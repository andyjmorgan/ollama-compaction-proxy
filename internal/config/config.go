// Package config loads the proxy's configuration from the environment.
package config

import (
	"crypto/rand"
	"fmt"
	"log/slog"
	"os"
	"strconv"
	"time"
)

// Config is the proxy's runtime configuration.
type Config struct {
	ListenAddr   string
	OllamaURL    string
	TokenizerURL string

	// CompactModel is the model used for summarization. Empty means use the
	// model of the request being compacted.
	CompactModel string

	// ModelPrefix, when non-empty, is stripped from incoming model names —
	// e.g. "agent/gemma4:e4b" resolves to "gemma4:e4b". This lets a fronting
	// gateway route by prefix (models: ["agent/*"] → this proxy) without
	// needing dynamic model-rewrite support of its own. Empty disables.
	ModelPrefix string

	// HMACKey signs compaction blobs. Persist it across restarts or OpenAI
	// clients holding old blobs will be rejected.
	HMACKey []byte

	// DefaultTriggerAnthropic is the input-token trigger when a compact edit
	// carries none (Anthropic documents 150000).
	DefaultTriggerAnthropic int

	// DefaultThresholdOpenAI is the compact_threshold when the request's
	// compaction entry carries none (OpenAI documents 200000).
	DefaultThresholdOpenAI int

	// MinTrigger floors client-supplied triggers, preventing pathological
	// every-turn compaction. Kept low so tests can force tiny watermarks.
	MinTrigger int

	MaxBodyBytes     int64
	SummarizeTimeout time.Duration
	UpstreamTimeout  time.Duration
	LogLevel         slog.Level
}

// StripModelPrefix removes the configured prefix from model if present.
func (c *Config) StripModelPrefix(model string) (string, bool) {
	if c.ModelPrefix == "" || len(model) <= len(c.ModelPrefix) {
		return model, false
	}
	if model[:len(c.ModelPrefix)] == c.ModelPrefix {
		return model[len(c.ModelPrefix):], true
	}
	return model, false
}

// FromEnv builds a Config from environment variables, applying defaults.
func FromEnv() (*Config, error) {
	cfg := &Config{
		ListenAddr:              env("LISTEN_ADDR", ":8082"),
		OllamaURL:               env("OLLAMA_URL", "http://127.0.0.1:11434"),
		TokenizerURL:            env("TOKENIZER_URL", "http://127.0.0.1:8081"),
		CompactModel:            os.Getenv("COMPACT_MODEL"),
		ModelPrefix:             env("MODEL_PREFIX", "agent/"),
		DefaultTriggerAnthropic: envInt("COMPACT_DEFAULT_TRIGGER_ANTHROPIC", 150000),
		DefaultThresholdOpenAI:  envInt("COMPACT_DEFAULT_THRESHOLD_OPENAI", 200000),
		MinTrigger:              envInt("COMPACT_MIN_TRIGGER", 1024),
		MaxBodyBytes:            int64(envInt("MAX_BODY_BYTES", 64<<20)),
		SummarizeTimeout:        envDuration("SUMMARIZE_TIMEOUT", 5*time.Minute),
		UpstreamTimeout:         envDuration("UPSTREAM_TIMEOUT", 10*time.Minute),
		LogLevel:                logLevel(),
	}

	if key := os.Getenv("COMPACT_HMAC_KEY"); key != "" {
		cfg.HMACKey = []byte(key)
	} else {
		cfg.HMACKey = make([]byte, 32)
		if _, err := rand.Read(cfg.HMACKey); err != nil {
			return nil, fmt.Errorf("generate ephemeral HMAC key: %w", err)
		}
		slog.Warn("COMPACT_HMAC_KEY not set: using a random per-boot key; " +
			"compaction blobs will not survive a restart")
	}

	if cfg.MinTrigger < 1 {
		return nil, fmt.Errorf("COMPACT_MIN_TRIGGER must be positive")
	}
	return cfg, nil
}

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return fallback
}

func envDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}

func logLevel() slog.Level {
	var lvl slog.Level
	if err := lvl.UnmarshalText([]byte(os.Getenv("LOG_LEVEL"))); err != nil {
		return slog.LevelInfo
	}
	return lvl
}
