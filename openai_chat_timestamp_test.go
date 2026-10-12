package protocolbridge

import (
	"encoding/json"
	"testing"
)

// Every streamed chat chunk carries "created", and a provider uses one value
// for the whole completion. The encoder used to write a literal 0 into each
// chunk, which is what these tests pin down.

func TestOpenAIChatStreamEncoderStampsOneNonZeroCreated(t *testing.T) {
	encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	parts := []StreamPart{
		{Type: StreamStart, ID: "chatcmpl-1"},
		{Type: StreamTextDelta, Delta: "hello"},
		{Type: StreamToolInputStart, ToolCallID: "call_1", ToolName: "Read"},
		{Type: StreamToolInputDelta, ToolCallID: "call_1", Delta: `{"file_path":"a.go"}`},
		{Type: StreamFinish, FinishReason: FinishToolCalls, Usage: Usage{InputTokens: intPtr(10), OutputTokens: intPtr(2)}},
	}

	var seen []float64
	for _, part := range parts {
		events, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(%s) error = %v", part.Type, err)
		}
		for _, event := range events {
			payload := rawStreamEventMap(t, event)
			created, ok := payload["created"].(float64)
			if !ok {
				t.Fatalf("event %q has no numeric created: %+v", event.Event, payload)
			}
			seen = append(seen, created)
		}
	}

	if len(seen) == 0 {
		t.Fatal("encoder produced no events")
	}
	for i, created := range seen {
		if created == 0 {
			t.Fatalf("event %d has created 0", i)
		}
		if created != seen[0] {
			t.Fatalf("event %d has created %v, want the stream's %v", i, created, seen[0])
		}
	}
}

func TestOpenAIChatStreamEncoderHonoursRequestedCreated(t *testing.T) {
	encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4o", Created: 1735689600})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamTextDelta, Delta: "hi"})
	if err != nil {
		t.Fatalf("Encode() error = %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %+v", events)
	}

	payload := rawStreamEventMap(t, events[0])
	if payload["created"] != float64(1735689600) {
		t.Fatalf("created = %v", payload["created"])
	}
}

func TestOpenAIChatEncodeResponseStampsCreated(t *testing.T) {
	resp := &LLMResponse{
		Protocol:     ProtocolAnthropicMessages,
		Model:        "claude-sonnet-4",
		Role:         RoleAssistant,
		Content:      []Part{{Type: PartText, Text: &TextPart{Text: "hi"}}},
		FinishReason: FinishStop,
	}

	raw, err := NewOpenAIChatAdapter().EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-4o", Created: 1735689600})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["created"] != float64(1735689600) {
		t.Fatalf("created = %v", decoded["created"])
	}

	// Without an explicit value the field must still be present and non-zero,
	// because a chat completion always carries one.
	raw, err = NewOpenAIChatAdapter().EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	created, ok := decoded["created"].(float64)
	if !ok || created == 0 {
		t.Fatalf("created = %v, want a stamped timestamp", decoded["created"])
	}
}
