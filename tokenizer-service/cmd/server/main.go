// Command server runs the Ollama token-count service.
package main

import (
	"context"
	"errors"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/httpapi"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/ollama"
	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/tokenizer"
)

func main() {
	log := slog.New(slog.NewJSONHandler(os.Stderr, &slog.HandlerOptions{Level: logLevel()}))

	ollamaURL := env("OLLAMA_URL", "http://127.0.0.1:11434")
	listenAddr := env("LISTEN_ADDR", ":8080")

	// Fetching a verbose /api/show response means pulling a full vocabulary —
	// around 12MB for gemma4 — so allow well beyond a normal API timeout.
	client := ollama.New(ollamaURL, &http.Client{Timeout: 2 * time.Minute})

	srv := &http.Server{
		Addr:              listenAddr,
		Handler:           httpapi.New(tokenizer.NewProvider(client), log),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       2 * time.Minute,
		WriteTimeout:      3 * time.Minute,
		IdleTimeout:       2 * time.Minute,
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	go func() {
		log.Info("listening", "addr", listenAddr, "ollama_url", ollamaURL)
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

func env(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
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
