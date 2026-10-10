package protocolbridge

import (
	"encoding/json"
	"strings"
	"testing"
)

// The fixtures in this file are shaped like the traffic a host relays when a
// coding agent that speaks OpenAI Responses — codex — is pointed at an upstream
// that speaks Chat Completions. Endpoint hosts and credentials are removed; the
// bodies keep their real structure, including the fields codex sets by
// convention: an instructions preamble, flat tool schemas, an encrypted
// reasoning item echoed back from the previous turn, and a trailing usage-only
// stream chunk.

// codexRequest is the first turn: a preamble, one user instruction, and the
// shell tool codex ships with.
const codexRequest = `{
  "model": "gpt-5.4",
  "instructions": "You are a coding agent running in a sandbox. Follow the user's request.",
  "input": [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List the files in the working directory."}]}
  ],
  "tools": [
    {"type": "function", "name": "shell", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}, "strict": false}
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "reasoning": {"effort": "medium", "summary": "auto"},
  "include": ["reasoning.encrypted_content"],
  "store": false,
  "stream": true
}`

// codexToolResultRequest is the follow-up turn. It replays what codex sends
// back: the reasoning item the upstream produced, the function call it made,
// and the output the runtime collected. A coding agent exercises this mapping
// on almost every turn.
const codexToolResultRequest = `{
  "model": "gpt-5.4",
  "instructions": "You are a coding agent running in a sandbox.",
  "input": [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "List the files in the working directory."}]},
    {"type": "reasoning", "id": "rs_01ABC", "summary": [{"type": "summary_text", "text": "The user wants a directory listing."}], "encrypted_content": "gAAAAABm"},
    {"type": "function_call", "id": "fc_01ABC", "call_id": "call_01ABC", "name": "shell", "arguments": "{\"command\":[\"bash\",\"-lc\",\"ls -1\"]}"},
    {"type": "function_call_output", "call_id": "call_01ABC", "output": "README.md\nmain.go\n"}
  ],
  "tools": [
    {"type": "function", "name": "shell", "description": "Run a shell command.", "parameters": {"type": "object", "properties": {"command": {"type": "array", "items": {"type": "string"}}}, "required": ["command"]}, "strict": false}
  ],
  "tool_choice": "auto",
  "parallel_tool_calls": false,
  "reasoning": {"effort": "medium", "summary": "auto"},
  "include": ["reasoning.encrypted_content"],
  "store": false,
  "stream": true
}`

func responsesToChatRequest(t *testing.T, body string, opts EncodeRequestOptions) (*LLMRequest, map[string]any) {
	t.Helper()

	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIResponses, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(responses, chat) ok = false, want true")
	}
	req, err := NewOpenAIResponsesAdapter().DecodeRequest([]byte(body))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	raw, err := bridge.EncodeUpstreamRequest(req, opts)
	if err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal(%s) error = %v", raw, err)
	}
	return req, decoded
}

func chatMessagesOf(t *testing.T, decoded map[string]any) []map[string]any {
	t.Helper()
	entries, ok := decoded["messages"].([]any)
	if !ok {
		t.Fatalf("messages = %#v", decoded["messages"])
	}
	messages := make([]map[string]any, 0, len(entries))
	for _, entry := range entries {
		message, ok := entry.(map[string]any)
		if !ok {
			t.Fatalf("message = %#v", entry)
		}
		messages = append(messages, message)
	}
	return messages
}

// TestResponsesToChatBridgeRelaysCodexFirstTurn takes the first-turn request
// through the path a host uses: decode the Responses body, then bridge the
// neutral request to a chat-completions upstream.
func TestResponsesToChatBridgeRelaysCodexFirstTurn(t *testing.T) {
	req, decoded := responsesToChatRequest(t, codexRequest, EncodeRequestOptions{Model: "gpt-5.4"})

	messages := chatMessagesOf(t, decoded)
	if len(messages) != 2 {
		t.Fatalf("messages = %+v", messages)
	}
	if messages[0]["role"] != "system" || !strings.Contains(messages[0]["content"].(string), "coding agent running in a sandbox") {
		t.Fatalf("system message = %+v, want the Responses instructions preamble", messages[0])
	}
	if messages[1]["role"] != "user" || messages[1]["content"] != "List the files in the working directory." {
		t.Fatalf("user message = %+v", messages[1])
	}

	// Codex sends flat tool schemas; chat wants them nested under "function".
	tools, ok := decoded["tools"].([]any)
	if !ok || len(tools) != 1 {
		t.Fatalf("tools = %#v", decoded["tools"])
	}
	tool := tools[0].(map[string]any)
	function, ok := tool["function"].(map[string]any)
	if !ok || tool["type"] != "function" || function["name"] != "shell" {
		t.Fatalf("tool = %+v", tool)
	}
	if _, ok := function["parameters"].(map[string]any); !ok {
		t.Fatalf("tool parameters = %#v", function["parameters"])
	}

	if decoded["stream"] != true {
		t.Fatalf("stream = %v", decoded["stream"])
	}
	// Chat only reports usage when it is asked to.
	options, ok := decoded["stream_options"].(map[string]any)
	if !ok || options["include_usage"] != true {
		t.Fatalf("stream_options = %#v", decoded["stream_options"])
	}

	// Nothing was dropped on the first turn.
	if len(req.Warnings) != 0 {
		t.Fatalf("warnings = %+v", req.Warnings)
	}
}

// TestResponsesToChatBridgeReplaysCodexToolResultTurn covers the turn shape a
// coding agent hits constantly: reasoning item, function call, function output.
func TestResponsesToChatBridgeReplaysCodexToolResultTurn(t *testing.T) {
	req, decoded := responsesToChatRequest(t, codexToolResultRequest, EncodeRequestOptions{Model: "gpt-5.4"})

	messages := chatMessagesOf(t, decoded)
	roles := make([]string, 0, len(messages))
	for _, message := range messages {
		roles = append(roles, message["role"].(string))
	}
	want := []string{"system", "user", "assistant", "tool"}
	if len(roles) != len(want) {
		t.Fatalf("roles = %v, want %v", roles, want)
	}
	for i := range want {
		if roles[i] != want[i] {
			t.Fatalf("roles = %v, want %v", roles, want)
		}
	}

	assistant := messages[2]
	calls, ok := assistant["tool_calls"].([]any)
	if !ok || len(calls) != 1 {
		t.Fatalf("tool_calls = %#v", assistant["tool_calls"])
	}
	call := calls[0].(map[string]any)
	if call["id"] != "call_01ABC" || call["type"] != "function" {
		t.Fatalf("tool call = %+v", call)
	}
	if call["function"].(map[string]any)["name"] != "shell" {
		t.Fatalf("tool call = %+v", call)
	}

	tool := messages[3]
	if tool["tool_call_id"] != "call_01ABC" {
		t.Fatalf("tool message = %+v", tool)
	}
	if tool["content"] != "README.md\nmain.go\n" {
		t.Fatalf("tool content = %v", tool["content"])
	}

	// The reasoning item has no chat-completions field. It is reported, at
	// informational severity, because the answer does not depend on it.
	if len(req.Warnings) != 1 {
		t.Fatalf("warnings = %+v", req.Warnings)
	}
	warning := req.Warnings[0]
	if warning.Code != LossDroppedReasoning || warning.Severity != SeverityInfo {
		t.Fatalf("warning = %+v", warning)
	}
	if warning.From != ProtocolOpenAIResponses || warning.To != ProtocolOpenAIChat {
		t.Fatalf("warning directions = %s -> %s", warning.From, warning.To)
	}

	// Strict refuses anything that dropped content, and an informational notice
	// is not content, so it must still go through.
	bridge, _ := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIResponses, ProtocolOpenAIChat)
	strictReq, err := NewOpenAIResponsesAdapter().DecodeRequest([]byte(codexToolResultRequest))
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}
	if _, err := bridge.EncodeUpstreamRequest(strictReq, EncodeRequestOptions{Model: "gpt-5.4", LossPolicy: LossPolicyStrict}); err != nil {
		t.Fatalf("EncodeUpstreamRequest() error = %v, want strict to tolerate an informational notice", err)
	}
}

// chatStreamChunk renders one chat-completions SSE frame.
func chatStreamChunk(t *testing.T, body string) RawStreamEvent {
	t.Helper()
	return RawStreamEvent{Event: "message", Data: []byte(body)}
}

func relayChatStreamToResponses(t *testing.T, chunks []string) []RawStreamEvent {
	t.Helper()

	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIResponses, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol(responses, chat) ok = false, want true")
	}
	decoder, err := bridge.NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}
	encoder, err := bridge.NewStreamEncoder(StreamEncodeOptions{Model: "gpt-5.4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	var events []RawStreamEvent
	for _, chunk := range chunks {
		parts, err := decoder.Decode(chatStreamChunk(t, chunk))
		if err != nil {
			t.Fatalf("Decode(%s) error = %v", chunk, err)
		}
		for _, part := range parts {
			encoded, err := encoder.Encode(part)
			if err != nil {
				t.Fatalf("Encode(%s) error = %v", part.Type, err)
			}
			events = append(events, encoded...)
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
		events = append(events, encoded...)
	}
	tail, err := encoder.Close()
	if err != nil {
		t.Fatalf("encoder Close() error = %v", err)
	}
	return append(events, tail...)
}

// TestResponsesToChatBridgeStreamsUsageIntoCompleted covers the ordering trap.
// A chat upstream reports usage in a chunk of its own, after finish_reason,
// while response.completed is the only Responses event that can carry usage.
// The finish therefore has to be held until the usage arrives, or the response
// is published as complete with no token counts at all.
func TestResponsesToChatBridgeStreamsUsageIntoCompleted(t *testing.T) {
	events := relayChatStreamToResponses(t, []string{
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"README.md"},"finish_reason":null}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
		`{"id":"chatcmpl-1","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[],"usage":{"prompt_tokens":1200,"completion_tokens":8,"total_tokens":1208,"prompt_tokens_details":{"cached_tokens":1024}}}`,
		`[DONE]`,
	})

	types := make([]string, 0, len(events))
	for _, event := range events {
		types = append(types, event.Event)
	}

	completedAt := -1
	for i, eventType := range types {
		if eventType == "response.completed" {
			if completedAt >= 0 {
				t.Fatalf("response.completed emitted twice: %v", types)
			}
			completedAt = i
		}
	}
	if completedAt < 0 {
		t.Fatalf("no response.completed in %v", types)
	}
	if completedAt != len(events)-1 {
		t.Fatalf("response.completed is not last: %v", types)
	}

	var completed map[string]any
	if err := json.Unmarshal(events[completedAt].Data, &completed); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	response := completed["response"].(map[string]any)
	usage, ok := response["usage"].(map[string]any)
	if !ok {
		t.Fatalf("completed response carries no usage: %+v", response)
	}
	if usage["input_tokens"] != float64(1200) || usage["output_tokens"] != float64(8) {
		t.Fatalf("usage = %+v", usage)
	}
	details, ok := usage["input_tokens_details"].(map[string]any)
	if !ok || details["cached_tokens"] != float64(1024) {
		t.Fatalf("input_tokens_details = %#v", usage["input_tokens_details"])
	}

	// The text the client reconstructs must survive the re-ordering.
	var text strings.Builder
	for _, event := range events {
		if event.Event != "response.output_text.delta" {
			continue
		}
		var delta map[string]any
		if err := json.Unmarshal(event.Data, &delta); err != nil {
			t.Fatalf("Unmarshal() error = %v", err)
		}
		text.WriteString(delta["delta"].(string))
	}
	if text.String() != "README.md" {
		t.Fatalf("streamed text = %q", text.String())
	}
}

// A chat upstream that reports usage before finish_reason must not lose it, and
// one that never reports usage must still finish.
func TestResponsesToChatBridgeHandlesUnusualUsageOrdering(t *testing.T) {
	t.Run("usage before finish", func(t *testing.T) {
		events := relayChatStreamToResponses(t, []string{
			`{"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`,
			`{"id":"chatcmpl-2","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`[DONE]`,
		})
		assertCompletedUsage(t, events, 10, 2, true)
	})

	t.Run("no usage at all", func(t *testing.T) {
		events := relayChatStreamToResponses(t, []string{
			`{"id":"chatcmpl-3","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}`,
			`{"id":"chatcmpl-3","object":"chat.completion.chunk","created":1,"model":"gpt-5.4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`,
			`[DONE]`,
		})
		assertCompletedUsage(t, events, 0, 0, false)
	})
}

func assertCompletedUsage(t *testing.T, events []RawStreamEvent, inputTokens, outputTokens int, wantUsage bool) {
	t.Helper()

	if len(events) == 0 || events[len(events)-1].Event != "response.completed" {
		types := make([]string, 0, len(events))
		for _, event := range events {
			types = append(types, event.Event)
		}
		t.Fatalf("last event is not response.completed: %v", types)
	}
	var completed map[string]any
	if err := json.Unmarshal(events[len(events)-1].Data, &completed); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	response := completed["response"].(map[string]any)
	if !wantUsage {
		if response["usage"] != nil {
			t.Fatalf("usage = %+v, want none reported", response["usage"])
		}
		return
	}
	usage, ok := response["usage"].(map[string]any)
	if !ok {
		t.Fatalf("completed response carries no usage: %+v", response)
	}
	if usage["input_tokens"] != float64(inputTokens) || usage["output_tokens"] != float64(outputTokens) {
		t.Fatalf("usage = %+v, want %d/%d", usage, inputTokens, outputTokens)
	}
}

// A non-streaming chat response must bridge to a Responses body the encoder can
// finish, including the tool call codex is waiting for.
func TestResponsesToChatBridgeDecodesChatToolCall(t *testing.T) {
	bridge, ok := NewCrossFamilyBridgeForProtocol(ProtocolOpenAIResponses, ProtocolOpenAIChat)
	if !ok {
		t.Fatal("NewCrossFamilyBridgeForProtocol() ok = false")
	}
	resp, err := bridge.DecodeUpstreamResponse([]byte(`{
	  "id": "chatcmpl-9",
	  "object": "chat.completion",
	  "created": 1,
	  "model": "gpt-5.4",
	  "choices": [{"index": 0, "message": {"role": "assistant", "content": null, "tool_calls": [{"id": "call_9", "type": "function", "function": {"name": "shell", "arguments": "{\"command\":[\"ls\"]}"}}]}, "finish_reason": "tool_calls"}],
	  "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14}
	}`))
	if err != nil {
		t.Fatalf("DecodeUpstreamResponse() error = %v", err)
	}
	if resp.Protocol != ProtocolOpenAIChat {
		t.Fatalf("protocol = %s", resp.Protocol)
	}

	raw, err := NewOpenAIResponsesAdapter().EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	output := decoded["output"].([]any)
	call := output[0].(map[string]any)
	if call["type"] != "function_call" || call["name"] != "shell" || call["call_id"] != "call_9" {
		t.Fatalf("function_call = %+v", call)
	}
}
