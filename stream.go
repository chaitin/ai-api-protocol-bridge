package protocolbridge

import (
	"encoding/json"
	"errors"
)

// ErrStreamTruncated reports an upstream stream that ended without a terminal
// event: the provider's connection dropped mid-generation, or it closed the
// stream without saying the response was finished.
//
// A decoder surfaces it as a StreamError part from Close. Without that, a
// truncated upstream is indistinguishable from a short but successful answer,
// because the encoder synthesizes a normal finish for any stream that merely
// stops.
var ErrStreamTruncated = errors.New("protocolbridge: upstream stream ended without a terminal event")

// rawStreamText renders a StreamRaw value for the wire.
//
// A raw value is either text a decoder could not parse, which is already JSON,
// or the decoded event struct it did not model. Rendering the struct with
// fmt.Sprint produced Go syntax — braces, field names, "<nil>" — and the chat
// encoder put that in the assistant's content, so a client saw the package's
// internals as the model's answer.
func rawStreamText(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case []byte:
		return string(typed)
	case json.RawMessage:
		return string(typed)
	}
	encoded, err := json.Marshal(value)
	if err != nil {
		return ""
	}
	return string(encoded)
}

// truncatedStreamError is the StreamError value a decoder emits for an
// upstream that ended early. The keys are the ones the error readers use, so
// every target protocol can render it.
func truncatedStreamError() any {
	return map[string]any{"code": "upstream_stream_truncated", "message": ErrStreamTruncated.Error()}
}

type RawStreamEvent struct {
	Event string
	Data  []byte
	ID    string
	Retry *int
}

type StreamPart struct {
	Type StreamPartType `json:"type"`

	ID string `json:"id,omitempty"`

	Delta string `json:"delta,omitempty"`

	ToolName   string `json:"tool_name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`

	Input any `json:"input,omitempty"`

	Output *ToolResultOutput `json:"output,omitempty"`

	File *FilePart `json:"file,omitempty"`

	FinishReason FinishReason `json:"finish_reason,omitempty"`

	Usage Usage `json:"usage,omitempty"`

	// Warnings lists what this part lost. No decoder in this package populates
	// it: a mid-stream loss has nowhere to go, because the only thing a
	// StreamEncoder can hand back is wire frames, and the client protocols have
	// no field for it. A stream that ends badly is reported as a StreamError
	// instead. The field is kept so a decoder can start reporting, but reading it
	// today always yields nil.
	Warnings []Warning `json:"warnings,omitempty"`

	Error any `json:"error,omitempty"`

	ProviderMetadata map[string]any `json:"provider_metadata,omitempty"`

	RawValue any `json:"raw_value,omitempty"`
}

type StreamPartType string

const (
	StreamStart StreamPartType = "stream-start"

	StreamTextStart StreamPartType = "text-start"
	StreamTextDelta StreamPartType = "text-delta"
	StreamTextEnd   StreamPartType = "text-end"

	StreamReasoningStart StreamPartType = "reasoning-start"
	StreamReasoningDelta StreamPartType = "reasoning-delta"
	StreamReasoningEnd   StreamPartType = "reasoning-end"

	StreamToolInputStart StreamPartType = "tool-input-start"
	StreamToolInputDelta StreamPartType = "tool-input-delta"
	StreamToolInputEnd   StreamPartType = "tool-input-end"

	StreamToolCall   StreamPartType = "tool-call"
	StreamToolResult StreamPartType = "tool-result"

	StreamFile StreamPartType = "file"

	StreamResponseMetadata StreamPartType = "response-metadata"

	StreamFinish StreamPartType = "finish"

	StreamRaw StreamPartType = "raw"

	StreamError StreamPartType = "error"
)
