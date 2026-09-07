package tokenizer

import (
	"encoding/json"
	"fmt"

	olm "github.com/ollama/ollama/tokenizer"
)

// GGUF tokenizer metadata keys, as returned by /api/show with verbose set.
const (
	keyModel     = "tokenizer.ggml.model"
	keyPre       = "tokenizer.ggml.pre"
	keyTokens    = "tokenizer.ggml.tokens"
	keyTokenType = "tokenizer.ggml.token_type"
	keyScores    = "tokenizer.ggml.scores"
	keyMerges    = "tokenizer.ggml.merges"
	keyBOS       = "tokenizer.ggml.bos_token_id"
	keyEOS       = "tokenizer.ggml.eos_token_id"
	keyEOSList   = "tokenizer.ggml.eos_token_ids"
	keyAddBOS    = "tokenizer.ggml.add_bos_token"
	keyAddEOS    = "tokenizer.ggml.add_eos_token"
)

// meta is the tokenizer description carried in a model's GGUF metadata.
type meta struct {
	family string // tokenizer.ggml.model: gpt2, llama, bert, ...
	pre    string // tokenizer.ggml.pre: the pretokenizer variant
	vocab  *olm.Vocabulary
}

// parseVocabulary builds an Ollama Vocabulary from raw /api/show model_info.
//
// Only the tokenizer keys are unmarshalled. A verbose response for gemma4 is
// about 12MB and holds 262k tokens plus 515k merges, so decoding the whole
// object generically would be wasteful.
func parseVocabulary(info map[string]json.RawMessage) (*meta, error) {
	if len(info) == 0 {
		return nil, fmt.Errorf("model_info missing from /api/show response")
	}

	m := &meta{vocab: &olm.Vocabulary{}}

	if err := decode(info, keyModel, &m.family); err != nil {
		return nil, err
	}
	if m.family == "" {
		return nil, fmt.Errorf("%s missing: model has no tokenizer metadata", keyModel)
	}
	if err := decode(info, keyPre, &m.pre); err != nil {
		return nil, err
	}

	if err := decode(info, keyTokens, &m.vocab.Values); err != nil {
		return nil, err
	}
	if len(m.vocab.Values) == 0 {
		return nil, fmt.Errorf("%s missing or empty", keyTokens)
	}
	if err := decode(info, keyTokenType, &m.vocab.Types); err != nil {
		return nil, err
	}
	if err := decode(info, keyScores, &m.vocab.Scores); err != nil {
		return nil, err
	}
	if err := decode(info, keyMerges, &m.vocab.Merges); err != nil {
		return nil, err
	}

	// Types indexes in lockstep with Values throughout the tokenizers; a short
	// slice would panic on lookup, so pad rather than trust the metadata.
	for len(m.vocab.Types) < len(m.vocab.Values) {
		m.vocab.Types = append(m.vocab.Types, olm.TOKEN_TYPE_NORMAL)
	}

	var bos, eos int32
	hasBOS := decode(info, keyBOS, &bos) == nil && info[keyBOS] != nil
	hasEOS := decode(info, keyEOS, &eos) == nil && info[keyEOS] != nil
	if hasBOS {
		m.vocab.BOS = []int32{bos}
	}
	if hasEOS {
		m.vocab.EOS = []int32{eos}
	}

	// Some models declare several stop tokens; gemma4 has three.
	var eosList []int32
	if err := decode(info, keyEOSList, &eosList); err == nil {
		for _, id := range eosList {
			if !contains(m.vocab.EOS, id) {
				m.vocab.EOS = append(m.vocab.EOS, id)
			}
		}
	}

	if err := decode(info, keyAddBOS, &m.vocab.AddBOS); err != nil {
		return nil, err
	}
	if err := decode(info, keyAddEOS, &m.vocab.AddEOS); err != nil {
		return nil, err
	}

	return m, nil
}

// decode unmarshals one optional key. A missing key leaves dst untouched.
func decode(info map[string]json.RawMessage, key string, dst any) error {
	raw, ok := info[key]
	if !ok || len(raw) == 0 || string(raw) == "null" {
		return nil
	}
	if err := json.Unmarshal(raw, dst); err != nil {
		return fmt.Errorf("decode %s: %w", key, err)
	}
	return nil
}

func contains(ids []int32, id int32) bool {
	for _, v := range ids {
		if v == id {
			return true
		}
	}
	return false
}
