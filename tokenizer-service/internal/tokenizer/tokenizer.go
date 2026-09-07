// Package tokenizer builds and caches tokenizers for locally installed Ollama
// models.
//
// The vocabulary always comes from the installed model's own GGUF metadata, via
// /api/show — never from a downloaded Hugging Face tokenizer, which could drift
// from the artifact actually being served. Tokenization runs on CPU using
// Ollama's tokenizer package; no weights are loaded and no inference happens.
package tokenizer

import (
	"context"
	"fmt"
	"strings"
	"sync"

	olm "github.com/ollama/ollama/tokenizer"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/ollama"
)

// Algorithm names the tokenization scheme chosen for a model, for logging and
// for the parity harness.
type Algorithm string

const (
	AlgoBPE         Algorithm = "bpe"
	AlgoSPMUnigram  Algorithm = "spm-unigram"
	AlgoSPMBytePair Algorithm = "spm-bpe"
	AlgoWordPiece   Algorithm = "wordpiece"
)

// Model is everything the service caches about one Ollama model: how to render
// a prompt for it, and how to tokenize the result.
type Model struct {
	ID       string
	Renderer string
	Template string

	// Thinking reports the model's thinking capability, which Ollama enables
	// by default and which changes what is rendered.
	Thinking bool

	// System is the model's own default system prompt, used when a request
	// supplies none.
	System string

	Algorithm Algorithm
	Pre       string

	tok olm.Tokenizer

	// leadingBOS is the model's BOS token as text, set only when the model
	// also adds BOS automatically. See Count.
	leadingBOS string
}

// Count returns the number of input tokens text costs for this model,
// including any special tokens the model adds automatically.
func (m *Model) Count(text string) (int, error) {
	// Several renderers write the BOS token into the rendered prompt
	// themselves — glimmer emits <|begin_of_text|>, gemma4 emits <bos>. When
	// the model ALSO declares add_bos_token, encoding that text with special
	// tokens enabled would count BOS twice. Drop the literal one and let the
	// tokenizer add it back, so exactly one is counted and EOS handling is
	// left alone.
	if m.leadingBOS != "" {
		text = strings.TrimPrefix(text, m.leadingBOS)
	}

	ids, err := m.tok.Encode(text, true)
	if err != nil {
		return 0, fmt.Errorf("tokenize: %w", err)
	}
	return len(ids), nil
}

// Provider loads models on demand and caches them for the process lifetime.
//
// The cache is unbounded, as the spec allows: the number of installed models is
// small and bounded. Each cached model holds its vocabulary plus the lazily
// built lookup maps — roughly 60-100MB for a 262k-token vocabulary like
// gemma4's. Failures are not cached, so a model pulled after a 404 works on the
// next request.
type Provider struct {
	client *ollama.Client

	mu      sync.Mutex
	entries map[string]*entry
}

type entry struct {
	done  chan struct{}
	model *Model
	err   error
}

// NewProvider returns a Provider backed by an Ollama instance.
func NewProvider(client *ollama.Client) *Provider {
	return &Provider{client: client, entries: make(map[string]*entry)}
}

// ForModel returns the cached model, loading it if necessary. The bool reports
// whether the model was already cached. Concurrent callers for the same model
// share a single load.
func (p *Provider) ForModel(ctx context.Context, id string) (*Model, bool, error) {
	p.mu.Lock()
	if e, ok := p.entries[id]; ok {
		p.mu.Unlock()
		select {
		case <-e.done:
			return e.model, true, e.err
		case <-ctx.Done():
			return nil, true, ctx.Err()
		}
	}

	e := &entry{done: make(chan struct{})}
	p.entries[id] = e
	p.mu.Unlock()

	e.model, e.err = p.load(ctx, id)
	if e.err != nil {
		// Don't cache failures: the model may be pulled, or Ollama restarted.
		p.mu.Lock()
		delete(p.entries, id)
		p.mu.Unlock()
	}
	close(e.done)

	return e.model, false, e.err
}

func (p *Provider) load(ctx context.Context, id string) (*Model, error) {
	show, err := p.client.ShowVerbose(ctx, id)
	if err != nil {
		return nil, err
	}

	meta, err := parseVocabulary(show.ModelInfo)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", id, err)
	}

	tok, algo, err := build(meta)
	if err != nil {
		return nil, fmt.Errorf("model %q: %w", id, err)
	}

	return &Model{
		ID:         id,
		Renderer:   show.Renderer(),
		Template:   show.Template,
		Thinking:   show.Thinking(),
		System:     show.System,
		Algorithm:  algo,
		Pre:        meta.pre,
		tok:        tok,
		leadingBOS: leadingBOS(meta.vocab),
	}, nil
}

// leadingBOS returns the model's BOS token as text when the model adds BOS
// automatically, and "" otherwise. A model that does not auto-add BOS must
// keep any literal BOS in its rendered prompt.
func leadingBOS(v *olm.Vocabulary) string {
	if !v.AddBOS || len(v.BOS) == 0 {
		return ""
	}
	id := int(v.BOS[0])
	if id < 0 || id >= len(v.Values) {
		return ""
	}
	return v.Values[id]
}

// build selects a tokenizer implementation from the model's declared family.
func build(m *meta) (olm.Tokenizer, Algorithm, error) {
	switch m.family {
	case "gpt2":
		bpe := olm.NewBytePairEncoding(m.vocab, patternsFor(m.pre, m.vocab.Values)...)
		return &bpe, AlgoBPE, nil

	case "llama", "spm", "sentencepiece":
		// A SentencePiece vocabulary carrying merges is a BPE model using
		// SentencePiece space normalization rather than a unigram model.
		// gemma4 ships both merges and scores; the parity harness settles
		// which path matches Ollama.
		if len(m.vocab.Merges) > 0 {
			bpe := olm.NewBytePairEncodingWithOptions(m.vocab, nil, olm.WithSentencePieceNormalizer())
			return &bpe, AlgoSPMBytePair, nil
		}
		if len(m.vocab.Scores) < len(m.vocab.Values) {
			return nil, "", fmt.Errorf("tokenizer %q has neither merges nor complete scores", m.family)
		}
		spm := olm.NewSentencePiece(m.vocab)
		return &spm, AlgoSPMUnigram, nil

	case "bert", "wordpiece":
		wpm := olm.NewWordPiece(m.vocab, true)
		return &wpm, AlgoWordPiece, nil

	default:
		return nil, "", fmt.Errorf("unsupported tokenizer %q", m.family)
	}
}
