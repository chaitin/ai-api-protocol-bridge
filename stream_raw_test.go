package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// OpenAI opens every chat stream with one chunk carrying both the model and the
// role. The decoder used to answer that with two StreamStart parts, so every
// encoder published its opening event twice: a Responses client saw
// response.created and response.in_progress repeated, and an Anthropic client
// saw message_start repeated. No hand-written assertion noticed, because each
// looked only for the event it cared about; the golden snapshot did.
func TestOpenAIChatStreamDecoderEmitsOneStartForTheOpeningChunk(t *testing.T) {
	decoder, err := NewOpenAIChatAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	parts, err := decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Decode() error = %v", err)
	}

	starts := 0
	for _, part := range parts {
		if part.Type != StreamStart {
			continue
		}
		starts++
		if part.ProviderMetadata["model"] != "gpt-5.4" {
			t.Fatalf("start metadata = %+v", part.ProviderMetadata)
		}
		if part.ProviderMetadata["role"] != "assistant" {
			t.Fatalf("start metadata = %+v, want the role folded in", part.ProviderMetadata)
		}
	}
	if starts != 1 {
		t.Fatalf("StreamStart parts = %d, want 1: %+v", starts, parts)
	}
}

func TestResponsesClientSeesOneOpeningEvent(t *testing.T) {
	events := relayChatStreamToResponses(t, goldenChatUpstreamChunks)

	counts := map[string]int{}
	for _, event := range events {
		counts[event.Event]++
	}
	for _, name := range []string{"response.created", "response.in_progress"} {
		if counts[name] != 1 {
			t.Fatalf("%s emitted %d times, want 1: %v", name, counts[name], counts)
		}
	}
}

// A raw frame is an upstream event this package does not model. It is not model
// output, so it must not become assistant content — which is what rendering it
// with fmt.Sprint did, putting Go syntax like "{ 0 <nil> }" in the answer.
func TestRawStreamFramesAreNotRenderedAsContent(t *testing.T) {
	t.Run("rawStreamText re-encodes a struct", func(t *testing.T) {
		type event struct {
			Type  string `json:"type"`
			Index int    `json:"index"`
		}
		got := rawStreamText(event{Type: "some_event", Index: 3})
		if got != `{"type":"some_event","index":3}` {
			t.Fatalf("rawStreamText() = %q", got)
		}
		if strings.Contains(got, "<nil>") || strings.Contains(got, "{ ") {
			t.Fatalf("rawStreamText() = %q, want JSON", got)
		}
	})

	t.Run("rawStreamText passes text through", func(t *testing.T) {
		if got := rawStreamText(`{"a":1}`); got != `{"a":1}` {
			t.Fatalf("rawStreamText() = %q", got)
		}
		if got := rawStreamText(nil); got != "" {
			t.Fatalf("rawStreamText(nil) = %q", got)
		}
	})

	t.Run("chat encoder keeps the frame out of content", func(t *testing.T) {
		encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4", Created: 1})
		if err != nil {
			t.Fatalf("NewStreamEncoder() error = %v", err)
		}
		events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
		if err != nil {
			t.Fatalf("Encode() error = %v", err)
		}
		if len(events) != 1 {
			t.Fatalf("events = %+v", events)
		}
		var chunk openAIChatStreamChunk
		if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		if chunk.Raw != `{"type":"some_event"}` {
			t.Fatalf("raw = %q", chunk.Raw)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta != nil && choice.Delta.Content != nil {
				t.Fatalf("raw frame reached content: %q", *choice.Delta.Content)
			}
		}
	})

	t.Run("anthropic encoder renders the frame as JSON", func(t *testing.T) {
		encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet-4"})
		if err != nil {
			t.Fatalf("NewStreamEncoder() error = %v", err)
		}
		events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
		if err != nil {
			t.Fatalf("Encode() error = %v", err)
		}
		if len(events) != 1 || !strings.Contains(string(events[0].Data), `\"type\":\"some_event\"`) {
			t.Fatalf("events = %s", events[0].Data)
		}
		if strings.Contains(string(events[0].Data), "<nil>") {
			t.Fatalf("Go syntax on the wire: %s", events[0].Data)
		}
	})

	t.Run("responses encoder renders the frame as JSON", func(t *testing.T) {
		encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4"})
		if err != nil {
			t.Fatalf("NewStreamEncoder() error = %v", err)
		}
		events, err := encoder.Encode(StreamPart{Type: StreamRaw, RawValue: map[string]any{"type": "some_event"}})
		if err != nil {
			t.Fatalf("Encode() error = %v", err)
		}
		if len(events) != 1 || !strings.Contains(string(events[0].Data), `\"type\":\"some_event\"`) {
			t.Fatalf("events = %s", events[0].Data)
		}
		if strings.Contains(string(events[0].Data), "<nil>") {
			t.Fatalf("Go syntax on the wire: %s", events[0].Data)
		}
	})
}
