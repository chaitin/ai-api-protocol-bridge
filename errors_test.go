package protocolbridge

import (
	"encoding/json"
	"testing"
)

// The Responses API reports a failed response in two different places, and the
// decoder used to read neither of them: response.failed nests its detail under
// response.error, while the bare error event carries code/message at the top
// level. These tests pin both, plus the outbound shapes each protocol requires.

func TestOpenAIResponsesDecodeResponseFailedReadsNestedError(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{
		Event: "response.failed",
		Data: []byte(`{"type":"response.failed","sequence_number":5,"response":{"id":"resp_1",` +
			`"status":"failed","error":{"code":"server_error","message":"upstream exploded"}}}`),
	})
	if err != nil {
		t.Fatalf("Decode(response.failed) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamError {
		t.Fatalf("parts = %+v", parts)
	}

	value, ok := parts[0].Error.(map[string]any)
	if !ok {
		t.Fatalf("error = %#v, want the decoded response.error object", parts[0].Error)
	}
	if value["message"] != "upstream exploded" || value["code"] != "server_error" {
		t.Fatalf("error = %+v", value)
	}
}

func TestOpenAIResponsesDecodeBareErrorEventReadsTopLevelFields(t *testing.T) {
	decoder, err := NewOpenAIResponsesAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{
		Event: "error",
		Data:  []byte(`{"type":"error","code":"rate_limit_exceeded","message":"slow down","sequence_number":2}`),
	})
	if err != nil {
		t.Fatalf("Decode(error) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamError {
		t.Fatalf("parts = %+v", parts)
	}

	value, ok := parts[0].Error.(map[string]any)
	if !ok {
		t.Fatalf("error = %#v, want a synthesized object", parts[0].Error)
	}
	if value["message"] != "slow down" || value["code"] != "rate_limit_exceeded" {
		t.Fatalf("error = %+v", value)
	}
}

func TestOpenAIResponsesEncodeStreamErrorUsesTopLevelFields(t *testing.T) {
	encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{
		Type:  StreamError,
		Error: map[string]any{"code": "server_error", "message": "upstream exploded"},
	})
	if err != nil {
		t.Fatalf("Encode(StreamError) error = %v", err)
	}
	if len(events) != 1 || events[0].Event != "error" {
		t.Fatalf("events = %+v", events)
	}

	payload := rawStreamEventMap(t, events[0])
	if payload["message"] != "upstream exploded" || payload["code"] != "server_error" {
		t.Fatalf("payload = %+v", payload)
	}
	if _, nested := payload["error"]; nested {
		t.Fatalf("error event must not nest its detail: %+v", payload)
	}
}

func TestOpenAIResponsesEncodeFailedFinishNestsResponseError(t *testing.T) {
	encoder, err := NewOpenAIResponsesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{
		Type:         StreamFinish,
		FinishReason: FinishError,
		Error:        map[string]any{"type": "api_error", "message": "boom"},
	})
	if err != nil {
		t.Fatalf("Encode(StreamFinish) error = %v", err)
	}
	if len(events) != 1 || events[0].Event != "response.failed" {
		t.Fatalf("events = %+v", events)
	}

	var event openAIResponsesStreamEvent
	if err := json.Unmarshal(events[0].Data, &event); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if event.Response == nil || event.Response.Status != "failed" {
		t.Fatalf("event = %+v", event)
	}
	value, ok := event.Response.Error.(map[string]any)
	if !ok {
		t.Fatalf("response.error = %#v, want an object", event.Response.Error)
	}
	if value["message"] != "boom" || value["code"] != "api_error" {
		t.Fatalf("response.error = %+v", value)
	}
}

func TestOpenAIChatEncodeStreamErrorReadsMessageObject(t *testing.T) {
	encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4o"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{
		Type:  StreamError,
		Error: map[string]any{"type": "api_error", "message": "boom"},
	})
	if err != nil {
		t.Fatalf("Encode(StreamError) error = %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("events = %+v", events)
	}

	payload := rawStreamEventMap(t, events[0])
	inner, ok := payload["error"].(map[string]any)
	if !ok {
		t.Fatalf("error = %#v", payload["error"])
	}
	if inner["message"] != "boom" {
		t.Fatalf("error.message = %v, want the provider message rather than a Go map rendering", inner["message"])
	}
	if inner["type"] != "api_error" {
		t.Fatalf("error.type = %v", inner["type"])
	}
}

func TestAnthropicEncodeStreamErrorSanitizesType(t *testing.T) {
	cases := []struct {
		name     string
		value    any
		wantType string
	}{
		{name: "openai code becomes api_error", value: map[string]any{"code": "server_error", "message": "boom"}, wantType: "api_error"},
		{name: "known anthropic type is kept", value: map[string]any{"type": "invalid_request_error", "message": "bad"}, wantType: "invalid_request_error"},
		{name: "reverse family wrapper is unwrapped", value: map[string]any{"error": map[string]any{"type": "overloaded_error", "message": "busy"}}, wantType: "overloaded_error"},
		{name: "plain string", value: "something broke", wantType: "api_error"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			encoder, err := NewAnthropicMessagesAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "claude-sonnet-4"})
			if err != nil {
				t.Fatalf("NewStreamEncoder() error = %v", err)
			}
			events, err := encoder.Encode(StreamPart{Type: StreamError, Error: tc.value})
			if err != nil {
				t.Fatalf("Encode(StreamError) error = %v", err)
			}
			if len(events) != 1 || events[0].Event != "error" {
				t.Fatalf("events = %+v", events)
			}

			payload := rawStreamEventMap(t, events[0])
			inner, ok := payload["error"].(map[string]any)
			if !ok {
				t.Fatalf("error = %#v", payload["error"])
			}
			if inner["type"] != tc.wantType {
				t.Fatalf("error.type = %v, want %v", inner["type"], tc.wantType)
			}
			if inner["message"] == "" || inner["message"] == "unknown error" {
				t.Fatalf("error.message = %v", inner["message"])
			}
		})
	}
}
