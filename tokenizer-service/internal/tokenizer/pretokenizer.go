package tokenizer

// Pretokenizer patterns keyed by the GGUF `tokenizer.ggml.pre` value.
//
// Ollama's NewBytePairEncoding takes regex patterns rather than a variant
// name, so the mapping has to live here. It is keyed on a value the model's
// own metadata declares — not on a model name — so adding a model never means
// touching this file, and an unrecognized variant falls through to Ollama's
// built-in byte-level default rather than failing.
//
// Patterns follow llama.cpp's pretokenizer table. Each entry is validated
// against Ollama's own prompt_eval_count by the parity harness; treat an entry
// as provisional until it appears there.
var pretokenizers = map[string][]string{
	// llama.cpp's fallback for BPE models that name no variant. Note it is
	// four patterns applied in sequence, and that its GPT-2 alternation ends
	// at \s+(?!\S) — it has no trailing \s+ branch, unlike the pattern Ollama
	// falls back to on its own.
	"default": {
		`[\p{P}\$\+<=>\^~\|]+`,
		`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`,
		`\p{N}+`,
		`[0-9][0-9][0-9]`,
	},

	// Llama 3 family. Digits group in runs of up to three.
	"llama3":    llama3Pretokenizer,
	"llama-bpe": llama3Pretokenizer,

	// GPT-4o family. llama.cpp maps llama4 (muse-glimmer) here rather than to
	// llama3: it splits on letter case and, unlike every other variant, keeps
	// a trailing "/" with a punctuation run — which is why it differs from
	// llama3 mainly on JSON-shaped text such as tool schemas.
	"gpt-4o": gpt4oPretokenizer,
	"llama4": gpt4oPretokenizer,

	// Qwen 2 family. As llama3, except digits are split individually.
	"qwen2": {
		`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
	},
}

var llama3Pretokenizer = []string{
	`(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
}

// gpt4oPretokenizer is llama.cpp's original tokenizer.json form rather than the
// character-class workaround it uses for C++ std::regex, since regexp2 supports
// Unicode property classes directly.
var gpt4oPretokenizer = []string{
	`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|` +
		`[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|` +
		`\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
}

// o200kMarkers are control tokens unique to the o200k_harmony vocabulary.
//
// Models carrying that vocabulary — gpt-oss among them — declare
// `tokenizer.ggml.pre = default`, but llama.cpp tokenizes them with the GPT-4o
// pretokenizer, not the default one. Taking the declared variant at face value
// overcounts such models by up to 12%. This looks at the vocabulary the model
// ships rather than at its name, so it stays metadata-driven.
var o200kMarkers = []string{"<|return|>", "<|call|>", "<|constrain|>"}

// patternsFor returns the pretokenizer patterns for a model. An empty result
// tells Ollama to apply its own default.
func patternsFor(pre string, vocab []string) []string {
	if pre == "" || pre == "default" {
		if isO200k(vocab) {
			return gpt4oPretokenizer
		}
	}
	return pretokenizers[pre]
}

func isO200k(vocab []string) bool {
	// The markers sit at the end of the vocabulary, after the byte and text
	// tokens, so scan backwards over the control-token block rather than the
	// whole 200k-entry list.
	const tail = 2048
	start := max(len(vocab)-tail, 0)

	found := 0
	for _, want := range o200kMarkers {
		for _, got := range vocab[start:] {
			if got == want {
				found++
				break
			}
		}
	}
	return found == len(o200kMarkers)
}
