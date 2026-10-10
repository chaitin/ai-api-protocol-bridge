package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// A chat client pointed at a Responses upstream is the last cell, and the one
// where the two protocols look most alike. The risk is not the request mapping
// — the Responses adapter already reads what the chat decoder produces — but
// the stream: Responses reports usage on response.completed, which also ends the
// stream, while a chat client expects finish_reason and usage in two separate
// chunks and then the [DONE] sentinel.

// chatClientRequest is one turn of a chat-speaking agent: a system preamble, a
// tool schema, an assistant tool call and its result.
const chatClientRequest = `{
  "model": "gpt-4.1",
  "messages": [
    {"role": "system", "content": "You are a coding agent running in a sandbox."},
    {"role": "user", "content": "List the files in the working directory."},
    {"role": "assistant", "content": null, "tool_calls": [{"id": "call_01ABC", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":[\"bash\",\"-lc\",\"ls -1\"]}"}}]},
    {"role": "tool", "tool_call_id": "call_01ABC", "content": "README.md\nmain.go\n"}
  ],
  "tools": [{"type": "function", "function": {"name": "shell", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}}}],
  "tool_choice": "auto",
  "stream": true
}`

// responsesInputItemKind names an input item. A plain message carries no "type"
// of its own — the protocol infers it from the presence of a role — so the
// absent case is the message case.
func responsesInputItemKind(item map[string]any) string {
	if kind, ok := item["type"].(string); ok && kind != "" {
		return kind
	}
	if _, ok := item["role"]; ok {
		return "message"
	}
	return ""
}

func chatToResponsesRequest(t *testing.T, body string) (*LLMRequest, map[string]any) {
	t.Helper()

	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolOpenAIResponses)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(chat, responses) ok = false, want true")
	}
	req, err := NewOpenAIChatAdapter().DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	raw, err := bridge.EncodeUpstreamRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", raw, err)
	}
	return req, decoded
}

func TestChatToResponsesBridgeRelaysChatRequest(t *testing.T) {
	_, decoded := chatToResponsesRequest(t, chatClientRequest)

	if !strings.Contains(decoded["instructions"].(string), "coding agent running in a sandbox") {
		t.Fatalf("instructions = %v, want the chat system message", decoded["instructions"])
	}

	input, ok := decoded["input"].([]any)
	if !ok {
		t.Fatalf("input = %#v", decoded["input"])
	}
	kinds := make([]string, 0, len(input))
	for _, entry := range input {
		kinds = append(kinds, responsesInputItemKind(entry.(map[string]any)))
	}
	want := []string{"message", "function_call", "function_call_output"}
	if len(kinds) != len(want) {
		t.Fatalf("input item kinds = %v, want %v", kinds, want)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("input item kinds = %v, want %v", kinds, want)
		}
	}

	call := input[1].(map[string]any)
	if call["call_id"] != "call_01ABC" || call["name"] != "shell" {
		t.Fatalf("function_call = %+v", call)
	}
	if !strings.Contains(call["arguments"].(string), "ls -1") {
		t.Fatalf("function_call arguments = %v", call["arguments"])
	}
	output := input[2].(map[string]any)
	if output["call_id"] != "call_01ABC" || output["output"] != "README.md\nmain.go\n" {
		t.Fatalf("function_call_output = %+v", output)
	}

	// Chat nests the schema under "function"; Responses keeps it flat.
	tools, ok := decoded["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v", decoded["tools"])
	}
	tool := tools[0].(map[string]any)
	if tool["type"] != "function" || tool["name"] != "shell" {
		t.Fatalf("tool = %+v", tool)
	}
	if _, nested := tool["function"]; nested {
		t.Fatalf("tool = %+v, want the flat Responses shape", tool)
	}
	if _, ok := tool["parameters"].(map[string]any); !ok {
		t.Fatalf("tool parameters = %#v", tool["parameters"])
	}
}

func relayResponsesStreamToChat(t *testing.T, events []string) []RawStreamEvent {
	t.Helper()

	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolOpenAIResponses)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(chat, responses) ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	var ordered []RawStreamEvent
	for _, body := range events {
		var envelope struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(body), &envelope); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", body, err)
		}
		parts, err := decoder.Decode(RawStreamEvent{Event: envelope.Type, Data: []byte(body)})
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", body, err)
		}
		for _, part := range parts {
			encoded, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", part.Type, err)
			}
			ordered = append(ordered, encoded...)
		}
	}
	closed, err := decoder.Close()
	if err != nil {
		t.Fatalf("decoder Close() error = %v", err)
	}
	for _, part := range closed {
		encoded, err := encoder.Encode(part)
		if err != nil {
			t.Fatalf("Encode(%s) error = %v", part.Type, err)
		}
		ordered = append(ordered, encoded...)
	}
	tail, err := encoder.Close()
	if err != nil {
		t.Fatalf("encoder Close() error = %v", err)
	}
	return append(ordered, tail...)
}

// responsesTextStream is a realistic Responses stream: the created envelope, the
// message item with its output_text deltas, and the completed envelope that
// carries usage.
var responsesTextStream = []string{
	`{"type":"response.created","response":{"id":"resp_1","object":"response","status":"in_progress","model":"gpt-5.4"}}`,
	`{"type":"response.output_item.added","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"in_progress","content":[]}}`,
	`{"type":"response.content_part.added","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`,
	`{"type":"response.output_text.delta","item_id":"msg_1","output_index":0,"content_index":0,"delta":"README.md"}`,
	`{"type":"response.output_text.done","item_id":"msg_1","output_index":0,"content_index":0,"text":"README.md"}`,
	`{"type":"response.content_part.done","item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"README.md"}}`,
	`{"type":"response.output_item.done","output_index":0,"item":{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"README.md"}]}}`,
	`{"type":"response.completed","response":{"id":"resp_1","object":"response","status":"completed","model":"gpt-5.4","output":[{"id":"msg_1","type":"message","role":"assistant","status":"completed","content":[{"type":"output_text","text":"README.md"}]}],"usage":{"input_tokens":1200,"input_tokens_details":{"cached_tokens":1024},"output_tokens":8,"total_tokens":1208}}}`,
}

// The client must receive finish_reason and usage in the order chat expects:
// text, then the finish chunk, then a usage-only chunk, then [DONE].
func TestChatToResponsesBridgeStreamsUsageAfterFinish(t *testing.T) {
	events := relayResponsesStreamToChat(t, responsesTextStream)

	types := make([]string, 0, len(events))
	for _, event := range events {
		if event.Event != "" {
			types = append(types, "event:"+event.Event)
			continue
		}
		types = append(types, "data")
	}

	if len(events) == 0 || string(events[len(events)-1].Data) != "[DONE]" {
		t.Fatalf("stream does not end with [DONE]: %v", types)
	}

	var text strings.Builder
	finishAt, usageAt := -1, -1
	for i, event := range events {
		if string(event.Data) == "[DONE]" {
			continue
		}
		var chunk openAIChatStreamChunk
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", event.Data, err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta != nil && choice.Delta.Content != nil {
				text.WriteString(*choice.Delta.Content)
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				if finishAt >= 0 {
					t.Fatalf("finish_reason reported twice: %v", types)
				}
				finishAt = i
				if *choice.FinishReason != "stop" {
					t.Fatalf("finish_reason = %q", *choice.FinishReason)
				}
			}
		}
		if chunk.Usage != nil {
			if usageAt >= 0 {
				t.Fatalf("usage reported twice: %v", types)
			}
			usageAt = i
		}
	}

	if text.String() != "README.md" {
		t.Fatalf("streamed text = %q", text.String())
	}
	if finishAt < 0 {
		t.Fatalf("no finish_reason chunk: %v", types)
	}
	if usageAt < 0 {
		t.Fatalf("no usage chunk: %v", types)
	}
	if usageAt < finishAt {
		t.Fatalf("usage chunk at %d precedes finish_reason at %d: %v", usageAt, finishAt, types)
	}

	// The usage that Responses reports on response.completed must survive as the
	// chat totals, with the cached portion broken out separately.
	var chunk openAIChatStreamChunk
	if err := json.Unmarshal(events[usageAt].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chunk.Usage.PromptTokens == nil || *chunk.Usage.PromptTokens != 1200 {
		t.Fatalf("prompt_tokens = %v, want the full 1200 the client was told", chunk.Usage.PromptTokens)
	}
	if chunk.Usage.CompletionTokens == nil || *chunk.Usage.CompletionTokens != 8 {
		t.Fatalf("completion_tokens = %v", chunk.Usage.CompletionTokens)
	}
	if chunk.Usage.TotalTokens == nil || *chunk.Usage.TotalTokens != 1208 {
		t.Fatalf("total_tokens = %v", chunk.Usage.TotalTokens)
	}
	if chunk.Usage.PromptTokensDetails == nil || chunk.Usage.PromptTokensDetails.CachedTokens == nil || *chunk.Usage.PromptTokensDetails.CachedTokens != 1024 {
		t.Fatalf("prompt_tokens_details = %+v", chunk.Usage.PromptTokensDetails)
	}
}

// Tool calls have to survive the same re-ordering: codex-style upstreams stream
// the arguments as deltas, and a chat client reads them back as fragments.
func TestChatToResponsesBridgeStreamsToolCall(t *testing.T) {
	events := relayResponsesStreamToChat(t, []string{
		`{"type":"response.created","response":{"id":"resp_2","object":"response","status":"in_progress","model":"gpt-5.4"}}`,
		`{"type":"response.output_item.added","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"in_progress","name":"shell","call_id":"call_01ABC","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"{\"command\":"}`,
		`{"type":"response.function_call_arguments.delta","item_id":"fc_1","output_index":0,"delta":"[\"ls\"]}"}`,
		`{"type":"response.function_call_arguments.done","item_id":"fc_1","output_index":0,"arguments":"{\"command\":[\"ls\"]}"}`,
		`{"type":"response.output_item.done","output_index":0,"item":{"id":"fc_1","type":"function_call","status":"completed","name":"shell","call_id":"call_01ABC","arguments":"{\"command\":[\"ls\"]}"}}`,
		`{"type":"response.completed","response":{"id":"resp_2","object":"response","status":"completed","model":"gpt-5.4","output":[],"usage":{"input_tokens":40,"output_tokens":6,"total_tokens":46}}}`,
	})

	arguments := strings.Builder{}
	finishReason := ""
	for _, event := range events {
		if string(event.Data) == "[DONE]" {
			continue
		}
		var chunk openAIChatStreamChunk
		if err := json.Unmarshal(event.Data, &chunk); err != nil {
			t.Fatalf("Unmarshal(%s) error = %v", event.Data, err)
		}
		for _, choice := range chunk.Choices {
			if choice.Delta != nil {
				for _, call := range choice.Delta.ToolCalls {
					if call.ID != "" && call.Function.Name != "shell" {
						t.Fatalf("tool call = %+v", call)
					}
					arguments.WriteString(call.Function.Arguments)
				}
			}
			if choice.FinishReason != nil && *choice.FinishReason != "" {
				finishReason = *choice.FinishReason
			}
		}
	}
	if arguments.String() != `{"command":["ls"]}` {
		t.Fatalf("streamed arguments = %q", arguments.String())
	}
	if finishReason != "tool_calls" {
		t.Fatalf("finish_reason = %q, want tool_calls", finishReason)
	}
}

// A tool-only assistant turn must not also produce an empty message item. This
// is a property of the Responses encoder rather than of the bridge, so it is
// pinned on the adapter directly: the chat and Anthropic decoders both hand it
// an assistant message whose only part is a tool call.
func TestOpenAIResponsesEncodeRequestOmitsEmptyAssistantMessage(t *testing.T) {
	adapter := NewOpenAIResponsesAdapter()
	req := &LLMRequest{
		Protocol: ProtocolOpenAIChat,
		Model:    "gpt-5.4",
		Prompt: []Message{
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "ls"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "shell", Input: map[string]any{"command": []any{"ls"}}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultText, Text: "main.go"}}}}},
		},
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	input := decoded["input"].([]any)
	kinds := make([]string, 0, len(input))
	for _, entry := range input {
		item := entry.(map[string]any)
		kinds = append(kinds, responsesInputItemKind(item))
		if responsesInputItemKind(item) != "message" {
			continue
		}
		content, ok := item["content"].([]any)
		if !ok || len(content) == 0 {
			t.Fatalf("empty message item sent upstream: %+v", item)
		}
	}
	want := []string{"message", "function_call", "function_call_output"}
	if len(kinds) != len(want) {
		t.Fatalf("input item kinds = %v, want %v", kinds, want)
	}
	for i := range want {
		if kinds[i] != want[i] {
			t.Fatalf("input item kinds = %v, want %v", kinds, want)
		}
	}
}

func TestChatToResponsesBridgeDecodesResponse(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIChat, ProtocolOpenAIResponses)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol() ok = false")
	}
	resp, err := bridge.DecodeUpstreamResponse([]byte(`{
	  "id": "resp_9",
	  "object": "response",
	  "status": "completed",
	  "model": "gpt-5.4",
	  "output": [{"id": "msg_9", "type": "message", "role": "assistant", "status": "completed", "content": [{"type": "output_text", "text": "done"}]}],
	  "usage": {"input_tokens": 1200, "input_tokens_details": {"cached_tokens": 1024}, "output_tokens": 8, "total_tokens": 1208}
	}`))
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.Protocol != ProtocolOpenAIResponses {
		t.Fatalf("protocol = %s", resp.Protocol)
	}

	raw, err := NewOpenAIChatAdapter().EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	choice := decoded["choices"].([]any)[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v", choice["finish_reason"])
	}
	if choice["message"].(map[string]any)["content"] != "done" {
		t.Fatalf("message = %+v", choice["message"])
	}
	usage := decoded["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(1200) || usage["completion_tokens"] != float64(8) {
		t.Fatalf("usage = %+v", usage)
	}
	if usage["prompt_tokens_details"].(map[string]any)["cached_tokens"] != float64(1024) {
		t.Fatalf("usage = %+v", usage)
	}
}
