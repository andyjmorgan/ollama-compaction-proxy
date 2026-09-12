// Command server runs the Ollama compaction proxy.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/internal/anthropicproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/chatproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/config"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/openaiproxy"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/summarize"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/tokencount"
	"github.com/andyjmorgan/ollama-compaction-proxy/internal/upstream"
)

func main() {
	cfg, err := config.FromEnv()
	if err != nil {
		slog.Error("configuration invalid", "error", err)
		os.Exit(1)
	}
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: cfg.LogLevel}))
	slog.SetDefault(log)

	up := upstream.New(cfg.OllamaURL, cfg.UpstreamTimeout)
	counter := tokencount.New(cfg.TokenizerURL)
	summarizer := summarize.New(up, cfg.CompactModel, cfg.SummarizeTimeout)

	anthropic := anthropicproxy.New(cfg, up, counter, summarizer, log)
	openaiHandler := openaiproxy.New(cfg, up, counter, summarizer, log)
	chat := chatproxy.New(up, cfg.ModelPrefix, log)

	router := httpapi.NewRouter(httpapi.Routes{
		Messages:         anthropic.Messages,
		Models:           anthropic.Models,
		CountTokens:      anthropic.CountTokens,
		Responses:        openaiHandler.Responses,
		ResponsesCompact: openaiHandler.Compact,
		ChatCompletions:  chat.ChatCompletions,
		Health:           health(cfg),
	}, cfg.MaxBodyBytes)

	srv := &http.Server{
		Addr:              cfg.ListenAddr,
		Handler:           router,
		ReadHeaderTimeout: 10 * time.Second,
		// No WriteTimeout: streams are long-lived; per-request contexts and
		// the upstream's lifecycle bound them instead.
		IdleTimeout: 2 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("listening", "addr", cfg.ListenAddr, "ollama_url", cfg.OllamaURL,
			"tokenizer_url", cfg.TokenizerURL, "compact_model", cfg.CompactModel)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Error("server failed", "error", err)
			stop()
		}
	}()

	<-ctx.Done()
	log.Info("shutting down")

	shutdownCtx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()
	if err := srv.Shutdown(shutdownCtx); err != nil {
		log.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}
}

// health reports proxy liveness plus upstream and tokenizer reachability.
func health(cfg *config.Config) http.HandlerFunc {
	probe := func(ctx context.Context, url string) string {
		req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
		if err != nil {
			return "error"
		}
		resp, err := (&http.Client{Timeout: 3 * time.Second}).Do(req)
		if err != nil {
			return "unreachable"
		}
		resp.Body.Close()
		return "ok"
	}

	return func(w http.ResponseWriter, r *http.Request) {
		ctx := r.Context()
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]string{
			"status":    "ok",
			"ollama":    probe(ctx, cfg.OllamaURL+"/api/version"),
			"tokenizer": probe(ctx, cfg.TokenizerURL+"/health"),
		})
	}
}
