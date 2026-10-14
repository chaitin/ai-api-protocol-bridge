package protocolbridge

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

// decodeToolInputJSON parses a tool call's input with every number kept as a
// json.Number, so encoding it again writes the literal the model produced.
// A float64 round trip would round an integer past 2^53 (a nanosecond
// timestamp, a snowflake id) and rewrite 2.5e3 as 2500, silently changing the
// arguments a tool is called with. ok is false unless raw is exactly one JSON
// value.
func decodeToolInputJSON(raw []byte) (input any, ok bool) {
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	if err := decoder.Decode(&input); err != nil {
		return nil, false
	}
	if _, err := decoder.Token(); !errors.Is(err, io.EOF) {
		return nil, false
	}
	return input, true
}

// UnmarshalJSON decodes a content block with its tool input's numbers kept as
// written; see decodeToolInputJSON. Every other field decodes as before.
func (b *anthropicContentBlock) UnmarshalJSON(data []byte) error {
	type plain anthropicContentBlock
	block := struct {
		*plain
		Input json.RawMessage `json:"input,omitempty"`
	}{plain: (*plain)(b)}
	if err := json.Unmarshal(data, &block); err != nil {
		return err
	}
	b.Input = nil
	if len(block.Input) == 0 {
		return nil
	}
	input, ok := decodeToolInputJSON(block.Input)
	if !ok {
		return errors.New("decode anthropic content block input: not a single JSON value")
	}
	b.Input = input
	return nil
}
