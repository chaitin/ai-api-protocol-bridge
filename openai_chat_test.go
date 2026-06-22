package protocolbridge

import (
	"encoding/json"
	"errors"
	"testing"
)

func TestOpenAIChatDecodeRequest(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	raw := []byte(`{
		"model":"gpt-4.1",
		"messages":[
			{"role":"system","content":"You are helpful."},
			{"role":"developer","content":"Follow project rules."},
			{"role":"user","content":[{"type":"text","text":"Hello"},{"type":"image_url","image_url":{"url":"data:image/png;base64,abc","detail":"high"}},{"type":"file","file":{"file_data":"Zm9v","filename":"notes.txt"}}]},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Shanghai\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"Sunny"}
		],
		"max_completion_tokens":128,
		"temperature":0.7,
		"stop":["END"],
		"tools":[{"type":"function","function":{"name":"get_weather","description":"Get weather.","parameters":{"type":"object"},"strict":true}}],
		"tool_choice":{"type":"function","function":{"name":"get_weather"}},
		"stream_options":{"include_usage":true},
		"metadata":{"trace_id":"trace_1"},
		"user":"dev-user",
		"service_tier":"flex",
		"parallel_tool_calls":false,
		"logprobs":true,
		"top_logprobs":2,
		"n":2,
		"store":true,
		"prediction":{"type":"content","content":"Hello"},
		"modalities":["text"],
		"audio":{"voice":"alloy","format":"mp3"},
		"stream":true
	}`)

	req, err := adapter.DecodeRequest(raw)
	if err != nil {
		t.Fatalf("DecodeRequest() error = %v", err)
	}

	if req.Model != "gpt-4.1" {
		t.Fatalf("Model = %q", req.Model)
	}
	if len(req.Prompt) != 5 {
		t.Fatalf("Prompt length = %d", len(req.Prompt))
	}
	if req.Prompt[0].Role != RoleSystem || req.Prompt[0].Parts[0].Text.Text != "You are helpful." {
		t.Fatalf("system message = %+v", req.Prompt[0])
	}
	if req.Prompt[1].Role != RoleDeveloper || req.Prompt[1].Parts[0].Text.Text != "Follow project rules." {
		t.Fatalf("developer message = %+v", req.Prompt[1])
	}
	if req.Prompt[2].Parts[0].Text.Text != "Hello" {
		t.Fatalf("user text = %+v", req.Prompt[2].Parts[0])
	}
	image := req.Prompt[2].Parts[1].File
	if image.Type != FileImage || image.MediaType != "image/png" || image.Data != "abc" || image.Detail != "high" {
		t.Fatalf("user image = %+v", image)
	}
	file := req.Prompt[2].Parts[2].File
	if file.Type != FileDocument || file.Data != "Zm9v" || file.Filename != "notes.txt" {
		t.Fatalf("user file = %+v", file)
	}
	toolCall := req.Prompt[3].Parts[0].ToolCall
	if toolCall.ToolCallID != "call_1" || toolCall.ToolName != "get_weather" {
		t.Fatalf("tool call = %+v", toolCall)
	}
	toolInput := toolCall.Input.(map[string]any)
	if toolInput["city"] != "Shanghai" {
		t.Fatalf("tool input = %+v", toolInput)
	}
	toolResult := req.Prompt[4].Parts[0].ToolResult
	if toolResult.ToolCallID != "call_1" || toolResult.Output.Text != "Sunny" {
		t.Fatalf("tool result = %+v", toolResult)
	}
	if req.MaxOutputTokens == nil || *req.MaxOutputTokens != 128 {
		t.Fatalf("MaxOutputTokens = %v", req.MaxOutputTokens)
	}
	if len(req.StopSequences) != 1 || req.StopSequences[0] != "END" {
		t.Fatalf("StopSequences = %+v", req.StopSequences)
	}
	if len(req.Tools) != 1 || req.Tools[0].InputSchema["type"] != "object" || req.Tools[0].Strict == nil || !*req.Tools[0].Strict {
		t.Fatalf("Tools = %+v", req.Tools)
	}
	if req.ToolChoice == nil || req.ToolChoice.Type != ToolChoiceTool || req.ToolChoice.ToolName != "get_weather" {
		t.Fatalf("ToolChoice = %+v", req.ToolChoice)
	}
	if req.ParallelToolCalls == nil || *req.ParallelToolCalls {
		t.Fatalf("ParallelToolCalls = %+v", req.ParallelToolCalls)
	}
	if req.CandidateCount == nil || *req.CandidateCount != 2 {
		t.Fatalf("CandidateCount = %+v", req.CandidateCount)
	}
	if !req.Stream {
		t.Fatal("Stream = false")
	}
	if req.ProviderOptions != nil {
		t.Fatalf("ProviderOptions = %+v", req.ProviderOptions)
	}
}

func TestOpenAIChatEncodeRequest(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	maxTokens := 128
	temperature := 0.7
	strict := true
	candidateCount := 2
	parallelToolCalls := false
	req := &LLMRequest{
		Model: "chaitin/gpt-5.5",
		Prompt: []Message{
			{Role: RoleSystem, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "You are helpful."}}}},
			{Role: RoleDeveloper, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Follow project rules."}}}},
			{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}, {Type: PartFile, File: &FilePart{Type: FileImage, MediaType: "image/png", Data: "abc", Detail: "high"}}, {Type: PartFile, File: &FilePart{Type: FileDocument, Data: "Zm9v", Filename: "notes.txt"}}}},
			{Role: RoleAssistant, Parts: []Part{{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: "call_1", ToolName: "get_weather", Input: map[string]any{"city": "Shanghai"}}}}},
			{Role: RoleTool, Parts: []Part{{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: "call_1", Output: ToolResultOutput{Type: ToolResultText, Text: "Sunny"}}}}},
		},
		MaxOutputTokens: &maxTokens,
		Temperature:     &temperature,
		CandidateCount:  &candidateCount,
		StopSequences:   []string{"END"},
		Tools: []Tool{{
			Type:        ToolFunction,
			Name:        "get_weather",
			Description: "Get weather.",
			InputSchema: map[string]any{"type": "object"},
			Strict:      &strict,
		}},
		ToolChoice:        &ToolChoice{Type: ToolChoiceRequired},
		ParallelToolCalls: &parallelToolCalls,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{Model: "gpt-5.5"})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["model"] != "gpt-5.5" {
		t.Fatalf("model = %v", decoded["model"])
	}
	if decoded["max_completion_tokens"] != float64(128) {
		t.Fatalf("max_completion_tokens = %v", decoded["max_completion_tokens"])
	}
	if _, ok := decoded["max_tokens"]; ok {
		t.Fatalf("max_tokens should be omitted: %+v", decoded)
	}
	if decoded["tool_choice"] != "required" {
		t.Fatalf("tool_choice = %v", decoded["tool_choice"])
	}
	if decoded["parallel_tool_calls"] != false {
		t.Fatalf("parallel_tool_calls = %v", decoded["parallel_tool_calls"])
	}
	if decoded["n"] != float64(2) {
		t.Fatalf("n = %v", decoded["n"])
	}
	for _, key := range []string{"metadata", "user", "service_tier", "logprobs", "top_logprobs", "store", "prediction", "modalities", "audio"} {
		if _, ok := decoded[key]; ok {
			t.Fatalf("provider-specific field %q should be omitted: %+v", key, decoded)
		}
	}
	toolFunction := decoded["tools"].([]any)[0].(map[string]any)["function"].(map[string]any)
	if toolFunction["strict"] != true {
		t.Fatalf("tool function = %+v", toolFunction)
	}

	messages := decoded["messages"].([]any)
	if len(messages) != 5 {
		t.Fatalf("messages length = %d", len(messages))
	}
	developer := messages[1].(map[string]any)
	if developer["role"] != "developer" || developer["content"] != "Follow project rules." {
		t.Fatalf("developer message = %+v", developer)
	}
	user := messages[2].(map[string]any)
	userContent := user["content"].([]any)
	if userContent[0].(map[string]any)["type"] != "text" {
		t.Fatalf("user content = %+v", userContent)
	}
	encodedImage := userContent[1].(map[string]any)
	imageURL := encodedImage["image_url"].(map[string]any)
	if encodedImage["type"] != "image_url" || imageURL["url"] != "data:image/png;base64,abc" || imageURL["detail"] != "high" {
		t.Fatalf("encoded image = %+v", encodedImage)
	}
	encodedFile := userContent[2].(map[string]any)
	fileObject := encodedFile["file"].(map[string]any)
	if encodedFile["type"] != "file" || fileObject["file_data"] != "Zm9v" || fileObject["filename"] != "notes.txt" {
		t.Fatalf("encoded file = %+v", encodedFile)
	}
	assistant := messages[3].(map[string]any)
	toolCalls := assistant["tool_calls"].([]any)
	function := toolCalls[0].(map[string]any)["function"].(map[string]any)
	if function["arguments"] != `{"city":"Shanghai"}` {
		t.Fatalf("arguments = %v", function["arguments"])
	}
	toolMessage := messages[4].(map[string]any)
	if toolMessage["role"] != "tool" || toolMessage["tool_call_id"] != "call_1" || toolMessage["content"] != "Sunny" {
		t.Fatalf("tool message = %+v", toolMessage)
	}
}

func TestOpenAIChatEncodeRequestUsesDefaultForNegativeMaxTokens(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	maxTokens := -1
	req := &LLMRequest{
		Model:           "chaitin/gpt-5.5",
		Prompt:          []Message{{Role: RoleUser, Parts: []Part{{Type: PartText, Text: &TextPart{Text: "Hello"}}}}},
		MaxOutputTokens: &maxTokens,
	}

	raw, err := adapter.EncodeRequest(req, EncodeRequestOptions{})
	if err != nil {
		t.Fatalf("EncodeRequest() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["max_completion_tokens"] != float64(defaultMaxOutputTokens) {
		t.Fatalf("max_completion_tokens = %v", decoded["max_completion_tokens"])
	}
}

func TestOpenAIChatDecodeResponse(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	raw := []byte(`{
		"id":"chatcmpl_1",
		"object":"chat.completion",
		"created":1710000000,
		"model":"gpt-4.1",
		"choices":[{"index":0,"message":{"role":"assistant","content":"Hello back"},"finish_reason":"stop"},{"index":1,"message":{"role":"assistant","content":"Alternative"},"finish_reason":"length"}],
		"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"prompt_tokens_details":{"cached_tokens":3},"completion_tokens_details":{"reasoning_tokens":2}}
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}

	if resp.ID != "chatcmpl_1" || resp.Model != "gpt-4.1" {
		t.Fatalf("response identity = %+v", resp)
	}
	if resp.FinishReason != FinishStop {
		t.Fatalf("FinishReason = %q", resp.FinishReason)
	}
	if len(resp.Content) != 1 || resp.Content[0].Text.Text != "Hello back" {
		t.Fatalf("content = %+v", resp.Content)
	}
	if len(resp.Choices) != 2 || resp.Choices[1].Content[0].Text.Text != "Alternative" || resp.Choices[1].FinishReason != FinishLength {
		t.Fatalf("choices = %+v", resp.Choices)
	}
	if *resp.Usage.InputTokens != 10 || *resp.Usage.OutputTokens != 5 {
		t.Fatalf("usage = %+v", resp.Usage)
	}
	if *resp.Usage.CachedInputTokens != 3 || *resp.Usage.ReasoningTokens != 2 {
		t.Fatalf("usage details = %+v", resp.Usage)
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 7 || billingUsage.CachedInputTokens != 3 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIChatDecodeResponseClampsBillingUsage(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	raw := []byte(`{
		"id":"chatcmpl_1",
		"object":"chat.completion",
		"model":"gpt-4.1",
		"choices":[{"index":0,"message":{"role":"assistant","content":"Hello"},"finish_reason":"stop"}],
		"usage":{"prompt_tokens":3,"completion_tokens":5,"prompt_tokens_details":{"cached_tokens":10}}
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	billingUsage := resp.BillingUsage()
	if billingUsage.InputTokens != 0 || billingUsage.CachedInputTokens != 10 || billingUsage.OutputTokens != 5 {
		t.Fatalf("billing usage = %+v", billingUsage)
	}
}

func TestOpenAIChatDecodeResponseRefusal(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	raw := []byte(`{
		"id":"chatcmpl_1",
		"object":"chat.completion",
		"model":"gpt-4.1",
		"choices":[{"index":0,"message":{"role":"assistant","refusal":"I'm sorry, I cannot assist with that request."},"finish_reason":"stop"}]
	}`)

	resp, err := adapter.DecodeResponse(raw)
	if err != nil {
		t.Fatalf("DecodeResponse() error = %v", err)
	}
	if len(resp.Content) != 1 || resp.Content[0].Type != PartRefusal || resp.Content[0].Refusal.Text != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("content = %+v", resp.Content)
	}
}

func TestOpenAIChatEncodeResponse(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	inputTokens := 10
	outputTokens := 5
	cachedInputTokens := 3
	totalInputTokens := inputTokens + cachedInputTokens
	resp := &LLMResponse{
		Protocol: ProtocolOpenAIChat,
		ID:       "chatcmpl_1",
		Model:    "chaitin/gpt-5.5",
		Role:     RoleAssistant,
		Content: []Part{
			{Type: PartReasoning, Reasoning: &ReasoningPart{Text: "I should answer briefly."}},
			{Type: PartText, Text: &TextPart{Text: "Hello back"}},
		},
		FinishReason: FinishStop,
		Usage: Usage{
			InputTokens:       &totalInputTokens,
			CachedInputTokens: &cachedInputTokens,
			OutputTokens:      &outputTokens,
		},
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{Model: "gpt-5.5"})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if decoded["object"] != "chat.completion" || decoded["model"] != "gpt-5.5" {
		t.Fatalf("response = %+v", decoded)
	}
	choices := decoded["choices"].([]any)
	choice := choices[0].(map[string]any)
	if choice["finish_reason"] != "stop" {
		t.Fatalf("finish_reason = %v", choice["finish_reason"])
	}
	message := choice["message"].(map[string]any)
	if message["role"] != "assistant" || message["content"] != "Hello back" {
		t.Fatalf("message = %+v", message)
	}
	if _, ok := message["reasoning_content"]; ok {
		t.Fatalf("message = %+v", message)
	}
	usage := decoded["usage"].(map[string]any)
	if usage["prompt_tokens"] != float64(13) || usage["completion_tokens"] != float64(5) || usage["total_tokens"] != float64(18) {
		t.Fatalf("usage = %+v", usage)
	}
	promptDetails := usage["prompt_tokens_details"].(map[string]any)
	if promptDetails["cached_tokens"] != float64(3) {
		t.Fatalf("prompt_tokens_details = %+v", promptDetails)
	}
}

func TestOpenAIChatEncodeResponseRefusal(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	resp := &LLMResponse{
		Protocol:     ProtocolOpenAIChat,
		ID:           "chatcmpl_1",
		Model:        "gpt-4.1",
		Role:         RoleAssistant,
		Content:      []Part{{Type: PartRefusal, Refusal: &RefusalPart{Text: "I'm sorry, I cannot assist with that request."}}},
		FinishReason: FinishStop,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	message := decoded["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if message["refusal"] != "I'm sorry, I cannot assist with that request." {
		t.Fatalf("message = %+v", message)
	}
	if _, ok := message["content"]; ok {
		t.Fatalf("content should be omitted: %+v", message)
	}
}

func TestOpenAIChatEncodeResponseWithChoices(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	resp := &LLMResponse{
		Protocol: ProtocolOpenAIChat,
		ID:       "chatcmpl_1",
		Model:    "gpt-4.1",
		Role:     RoleAssistant,
		Content:  []Part{{Type: PartText, Text: &TextPart{Text: "First"}}},
		Choices: []LLMChoice{
			{Index: 0, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "First"}}}, FinishReason: FinishStop},
			{Index: 1, Role: RoleAssistant, Content: []Part{{Type: PartText, Text: &TextPart{Text: "Second"}}}, FinishReason: FinishLength},
		},
		FinishReason: FinishStop,
	}

	raw, err := adapter.EncodeResponse(resp, EncodeResponseOptions{})
	if err != nil {
		t.Fatalf("EncodeResponse() error = %v", err)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	choices := decoded["choices"].([]any)
	if len(choices) != 2 {
		t.Fatalf("choices length = %d", len(choices))
	}
	second := choices[1].(map[string]any)
	secondMessage := second["message"].(map[string]any)
	if second["index"] != float64(1) || second["finish_reason"] != "length" || secondMessage["content"] != "Second" {
		t.Fatalf("second choice = %+v", second)
	}
}

func TestOpenAIChatEncodeError(t *testing.T) {
	adapter := NewOpenAIChatAdapter()
	raw, status := adapter.EncodeError(errors.New("bad request"))
	if status != 400 {
		t.Fatalf("status = %d", status)
	}

	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	openAIError := decoded["error"].(map[string]any)
	if openAIError["message"] != "bad request" || openAIError["type"] != "protocol_bridge_error" {
		t.Fatalf("error = %+v", openAIError)
	}
}

func TestOpenAIChatStreamDecoder(t *testing.T) {
	decoder, err := NewOpenAIChatAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Decode(role) error = %v", err)
	}
	if len(parts) < 1 || parts[0].Type != StreamStart {
		t.Fatalf("role parts = %+v", parts)
	}

	content := "Hi"
	parts, err = decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Decode(content) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamTextDelta || parts[0].Delta != content {
		t.Fatalf("content parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)})
	if err != nil {
		t.Fatalf("Decode(finish) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamFinish || parts[0].FinishReason != FinishStop {
		t.Fatalf("finish parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Data: []byte("[DONE]")})
	if err != nil {
		t.Fatalf("Decode(DONE) error = %v", err)
	}
	if len(parts) != 0 {
		t.Fatalf("DONE parts = %+v", parts)
	}
}

func TestOpenAIChatStreamDecoderToolCall(t *testing.T) {
	decoder, err := NewOpenAIChatAdapter().NewStreamDecoder(StreamDecodeOptions{})
	if err != nil {
		t.Fatalf("NewStreamDecoder() error = %v", err)
	}

	parts, err := decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"get_weather","arguments":""}}]},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Decode(tool start) error = %v", err)
	}
	if len(parts) < 1 {
		t.Fatalf("tool start parts = %+v", parts)
	}
	var foundStart bool
	for _, p := range parts {
		if p.Type == StreamToolInputStart && p.ToolCallID == "call_1" && p.ToolName == "get_weather" {
			foundStart = true
		}
	}
	if !foundStart {
		t.Fatalf("tool start not found in parts = %+v", parts)
	}

	parts, err = decoder.Decode(RawStreamEvent{Data: []byte(`{"id":"chatcmpl-1","object":"chat.completion.chunk","model":"gpt-4","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\"city\":"}}]},"finish_reason":null}]}`)})
	if err != nil {
		t.Fatalf("Decode(tool delta) error = %v", err)
	}
	if len(parts) != 1 || parts[0].Type != StreamToolInputDelta || parts[0].Delta != `{"city":` {
		t.Fatalf("tool delta parts = %+v", parts)
	}
}

func TestOpenAIChatStreamEncoder(t *testing.T) {
	encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}

	events, err := encoder.Encode(StreamPart{Type: StreamStart, ID: "chatcmpl-1"})
	if err != nil {
		t.Fatalf("Encode(StreamStart) error = %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("StreamStart events = %+v", events)
	}
	var chunk openAIChatStreamChunk
	if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chunk.Object != "chat.completion.chunk" || chunk.ID != "chatcmpl-1" {
		t.Fatalf("chunk = %+v", chunk)
	}

	events, err = encoder.Encode(StreamPart{Type: StreamTextDelta, Delta: "Hi"})
	if err != nil {
		t.Fatalf("Encode(StreamTextDelta) error = %v", err)
	}
	if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chunk.Choices[0].Delta.Content == nil || *chunk.Choices[0].Delta.Content != "Hi" {
		t.Fatalf("delta content = %+v", chunk.Choices[0].Delta)
	}

	inputTokens := 13
	outputTokens := 5
	cachedTokens := 3
	events, err = encoder.Encode(StreamPart{
		Type:         StreamFinish,
		FinishReason: FinishStop,
		Usage: Usage{
			InputTokens:       &inputTokens,
			OutputTokens:      &outputTokens,
			CachedInputTokens: &cachedTokens,
		},
	})
	if err != nil {
		t.Fatalf("Encode(StreamFinish) error = %v", err)
	}
	if len(events) != 1 {
		t.Fatalf("StreamFinish events = %+v", events)
	}
	if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if chunk.Choices[0].FinishReason == nil || *chunk.Choices[0].FinishReason != "stop" {
		t.Fatalf("finish reason = %+v", chunk.Choices[0].FinishReason)
	}
	if chunk.Usage == nil || chunk.Usage.PromptTokens == nil || *chunk.Usage.PromptTokens != 13 || chunk.Usage.CompletionTokens == nil || *chunk.Usage.CompletionTokens != 5 {
		t.Fatalf("finish usage = %+v", chunk.Usage)
	}
	if chunk.Usage.PromptTokensDetails == nil || chunk.Usage.PromptTokensDetails.CachedTokens == nil || *chunk.Usage.PromptTokensDetails.CachedTokens != 3 {
		t.Fatalf("finish cached tokens = %+v", chunk.Usage)
	}

	events, err = encoder.Close()
	if err != nil {
		t.Fatalf("Close() error = %v", err)
	}
	if len(events) != 1 || string(events[0].Data) != "[DONE]" {
		t.Fatalf("Close events = %+v", events)
	}
}

func TestOpenAIChatStreamEncoderToolCall(t *testing.T) {
	encoder, err := NewOpenAIChatAdapter().NewStreamEncoder(StreamEncodeOptions{Model: "gpt-4"})
	if err != nil {
		t.Fatalf("NewStreamEncoder() error = %v", err)
	}
	_, _ = encoder.Encode(StreamPart{Type: StreamStart, ID: "chatcmpl-1"})

	events, err := encoder.Encode(StreamPart{Type: StreamToolInputStart, ToolCallID: "call_1", ToolName: "get_weather"})
	if err != nil {
		t.Fatalf("Encode(ToolInputStart) error = %v", err)
	}
	var chunk openAIChatStreamChunk
	if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(chunk.Choices) == 0 || chunk.Choices[0].Delta == nil || len(chunk.Choices[0].Delta.ToolCalls) == 0 {
		t.Fatalf("tool start chunk = %+v", chunk)
	}
	tc := chunk.Choices[0].Delta.ToolCalls[0]
	if tc.ID != "call_1" || tc.Function.Name != "get_weather" {
		t.Fatalf("tool call = %+v", tc)
	}

	events, err = encoder.Encode(StreamPart{Type: StreamToolInputDelta, ToolCallID: "call_1", Delta: `{"city":`})
	if err != nil {
		t.Fatalf("Encode(ToolInputDelta) error = %v", err)
	}
	if err := json.Unmarshal(events[0].Data, &chunk); err != nil {
		t.Fatalf("Unmarshal() error = %v", err)
	}
	if len(chunk.Choices[0].Delta.ToolCalls) == 0 || chunk.Choices[0].Delta.ToolCalls[0].Function.Arguments != `{"city":` {
		t.Fatalf("tool delta = %+v", chunk.Choices[0].Delta.ToolCalls[0])
	}

	events, err = encoder.Encode(StreamPart{Type: StreamToolInputEnd, ToolCallID: "call_1"})
	if err != nil {
		t.Fatalf("Encode(ToolInputEnd) error = %v", err)
	}
	if len(events) != 0 {
		t.Fatalf("ToolInputEnd events should be empty = %+v", events)
	}
}
