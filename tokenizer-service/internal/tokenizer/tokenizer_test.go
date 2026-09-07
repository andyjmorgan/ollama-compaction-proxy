package tokenizer

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/andyjmorgan/ollama-compaction-proxy/tokenizer-service/internal/testsupport"
)

// rawInfo converts fixture metadata to the raw-JSON form /api/show yields.
func rawInfo(t *testing.T, info map[string]any) map[string]json.RawMessage {
	t.Helper()

	out := make(map[string]json.RawMessage, len(info))
	for k, v := range info {
		raw, err := json.Marshal(v)
		if err != nil {
			t.Fatalf("marshal %s: %v", k, err)
		}
		out[k] = raw
	}
	return out
}

func TestParseVocabulary(t *testing.T) {
	meta, err := parseVocabulary(rawInfo(t, testsupport.ByteVocabulary()))
	if err != nil {
		t.Fatalf("parseVocabulary: %v", err)
	}

	if meta.family != "gpt2" {
		t.Errorf("family = %q, want gpt2", meta.family)
	}
	if meta.pre != "default" {
		t.Errorf("pre = %q, want default", meta.pre)
	}
	if got, want := len(meta.vocab.Values), 258; got != want {
		t.Errorf("vocab size = %d, want %d", got, want)
	}
	if len(meta.vocab.Types) != len(meta.vocab.Values) {
		t.Errorf("Types len = %d, want %d", len(meta.vocab.Types), len(meta.vocab.Values))
	}
	if len(meta.vocab.BOS) != 1 || len(meta.vocab.EOS) != 1 {
		t.Errorf("BOS = %v, EOS = %v, want one each", meta.vocab.BOS, meta.vocab.EOS)
	}
	if meta.vocab.AddBOS {
		t.Error("AddBOS = true, want false per the fixture metadata")
	}
}

// gemma4 declares three stop tokens via eos_token_ids alongside a scalar
// eos_token_id; all must survive, without duplicates.
func TestParseVocabularyMergesEOSList(t *testing.T) {
	info := testsupport.ByteVocabulary()
	info["tokenizer.ggml.eos_token_id"] = 1
	info["tokenizer.ggml.eos_token_ids"] = []int32{1, 106, 50}

	meta, err := parseVocabulary(rawInfo(t, info))
	if err != nil {
		t.Fatalf("parseVocabulary: %v", err)
	}

	want := []int32{1, 106, 50}
	if len(meta.vocab.EOS) != len(want) {
		t.Fatalf("EOS = %v, want %v", meta.vocab.EOS, want)
	}
	for i, id := range want {
		if meta.vocab.EOS[i] != id {
			t.Errorf("EOS[%d] = %d, want %d", i, meta.vocab.EOS[i], id)
		}
	}
}

// A short token_type array would panic on lookup during encoding.
func TestParseVocabularyPadsShortTypes(t *testing.T) {
	info := testsupport.ByteVocabulary()
	info["tokenizer.ggml.token_type"] = []int32{1, 1, 1}

	meta, err := parseVocabulary(rawInfo(t, info))
	if err != nil {
		t.Fatalf("parseVocabulary: %v", err)
	}
	if len(meta.vocab.Types) != len(meta.vocab.Values) {
		t.Errorf("Types len = %d, want %d", len(meta.vocab.Types), len(meta.vocab.Values))
	}
}

func TestParseVocabularyErrors(t *testing.T) {
	cases := []struct {
		name string
		info map[string]any
		want string
	}{
		{"no metadata", map[string]any{}, "model_info missing"},
		{"no tokenizer family", map[string]any{"general.architecture": "gemma4"}, "tokenizer.ggml.model missing"},
		{"no tokens", map[string]any{"tokenizer.ggml.model": "gpt2"}, "tokenizer.ggml.tokens missing"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, err := parseVocabulary(rawInfo(t, tc.info))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

func TestBuildSelectsAlgorithm(t *testing.T) {
	cases := []struct {
		name   string
		family string
		merges []string
		scores bool
		want   Algorithm
	}{
		{"gpt2 is byte-pair", "gpt2", nil, false, AlgoBPE},
		{"llama with merges is sentencepiece bpe", "llama", []string{"a b"}, true, AlgoSPMBytePair},
		{"llama without merges is unigram", "llama", nil, true, AlgoSPMUnigram},
		{"spm alias", "spm", nil, true, AlgoSPMUnigram},
		{"bert is wordpiece", "bert", nil, false, AlgoWordPiece},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := testsupport.ByteVocabulary()
			info["tokenizer.ggml.model"] = tc.family
			info["tokenizer.ggml.merges"] = tc.merges
			if tc.scores {
				scores := make([]float32, 258)
				info["tokenizer.ggml.scores"] = scores
			}

			meta, err := parseVocabulary(rawInfo(t, info))
			if err != nil {
				t.Fatalf("parseVocabulary: %v", err)
			}
			_, algo, err := build(meta)
			if err != nil {
				t.Fatalf("build: %v", err)
			}
			if algo != tc.want {
				t.Errorf("algorithm = %q, want %q", algo, tc.want)
			}
		})
	}
}

func TestBuildRejectsUnusableTokenizers(t *testing.T) {
	cases := []struct {
		name   string
		family string
		want   string
	}{
		{"unknown family", "rwkv", "unsupported tokenizer"},
		{"sentencepiece without merges or scores", "llama", "neither merges nor complete scores"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			info := testsupport.ByteVocabulary()
			info["tokenizer.ggml.model"] = tc.family

			meta, err := parseVocabulary(rawInfo(t, info))
			if err != nil {
				t.Fatalf("parseVocabulary: %v", err)
			}
			if _, _, err := build(meta); err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Errorf("err = %v, want it to mention %q", err, tc.want)
			}
		})
	}
}

// The byte-level fixture yields one token per byte, which makes the sanity
// properties checkable without pinning arbitrary counts.
func TestCountIsPositiveAndMonotonic(t *testing.T) {
	meta, err := parseVocabulary(rawInfo(t, testsupport.ByteVocabulary()))
	if err != nil {
		t.Fatalf("parseVocabulary: %v", err)
	}
	tok, algo, err := build(meta)
	if err != nil {
		t.Fatalf("build: %v", err)
	}
	m := &Model{Algorithm: algo, tok: tok}

	short, err := m.Count("hello")
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if short <= 0 {
		t.Errorf("Count(hello) = %d, want > 0", short)
	}

	long, err := m.Count(strings.Repeat("hello world ", 40))
	if err != nil {
		t.Fatalf("Count: %v", err)
	}
	if long <= short {
		t.Errorf("Count(long) = %d, want > %d", long, short)
	}
}

func TestPatternsForFallsBackToOllamaDefault(t *testing.T) {
	if got := patternsFor("a-variant-that-does-not-exist", nil); got != nil {
		t.Errorf("patternsFor(unknown) = %v, want nil so Ollama applies its default", got)
	}
	for _, pre := range []string{"default", "llama3", "llama4", "qwen2"} {
		if len(patternsFor(pre, nil)) == 0 {
			t.Errorf("patternsFor(%q) is empty, want a pattern", pre)
		}
	}
}

// A vocabulary carrying the o200k_harmony markers is tokenized with the GPT-4o
// pretokenizer even though it declares the default variant.
func TestPatternsForDetectsO200kVocabulary(t *testing.T) {
	plain := []string{"hello", "world"}
	o200k := append(append([]string{}, plain...), "<|return|>", "<|call|>", "<|constrain|>")

	if got, want := patternsFor("default", o200k), gpt4oPretokenizer; len(got) != len(want) || got[0] != want[0] {
		t.Errorf("o200k vocabulary got the default pretokenizer, want the GPT-4o one")
	}
	if got := patternsFor("default", plain); len(got) != len(pretokenizers["default"]) {
		t.Errorf("plain vocabulary should keep the default pretokenizer, got %d patterns", len(got))
	}
	// An explicitly named variant is never overridden.
	if got, want := patternsFor("qwen2", o200k), pretokenizers["qwen2"]; got[0] != want[0] {
		t.Error("a declared variant must win over vocabulary detection")
	}
}
