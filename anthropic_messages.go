package protocolbridge

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"regexp"
	"strings"
)

var claudeCodeVolatileSystemPromptPattern = regexp.MustCompile(`^(?:x-anthropic-billing-header:\s*)?cc_version=[^;\n]+;\s*cc_entrypoint=[^;\n]+;\s*cch=[^;\n]+;\s*$`)

type AnthropicMessagesAdapter struct{}

func NewAnthropicMessagesAdapter() AnthropicMessagesAdapter {
	return AnthropicMessagesAdapter{}
}

func (a AnthropicMessagesAdapter) Protocol() Protocol {
	return ProtocolAnthropicMessages
}

func (a AnthropicMessagesAdapter) DecodeRequest(raw []byte) (*LLMRequest, error) {
	var request anthropicRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return nil, fmt.Errorf("decode anthropic messages request: %w", err)
	}

	prompt := make([]Message, 0, len(request.Messages)+1)
	if system := request.systemText(); strings.TrimSpace(system) != "" {
		prompt = append(prompt, Message{
			Role:  RoleSystem,
			Parts: []Part{{Type: PartText, Text: &TextPart{Text: system}}},
		})
	}
	for _, message := range request.Messages {
		decoded, err := decodeAnthropicMessage(message)
		if err != nil {
			return nil, err
		}
		prompt = append(prompt, decoded)
	}

	return &LLMRequest{
		Protocol:              ProtocolAnthropicMessages,
		Model:                 request.Model,
		Prompt:                prompt,
		MaxOutputTokens:       maxOutputTokensOrDefault(request.MaxTokens),
		Temperature:           request.Temperature,
		StopSequences:         request.StopSequences,
		TopP:                  request.TopP,
		TopK:                  request.TopK,
		ResponseFormat:        decodeAnthropicOutputConfig(request.OutputConfig),
		Cache:                 decodeAnthropicCache(request),
		Reasoning:             decodeAnthropicThinking(request.Thinking),
		ReasoningBudgetTokens: decodeAnthropicThinkingBudget(request.Thinking),
		ReasoningEffort:       decodeAnthropicReasoningEffort(request.Thinking),
		Tools:                 decodeAnthropicTools(request.Tools),
		ToolChoice:            decodeAnthropicToolChoice(request.ToolChoice),
		ParallelToolCalls:     decodeAnthropicParallelToolCalls(request.ToolChoice),
		Stream:                request.Stream,
		Metadata:              decodeAnthropicMetadata(request.Metadata),
	}, nil
}

func (a AnthropicMessagesAdapter) EncodeRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, errors.New("encode anthropic messages request: nil request")
	}

	model := req.Model
	if opts.Model != "" {
		model = opts.Model
	}

	request := anthropicRequest{
		Model:         model,
		MaxTokens:     maxOutputTokensOrDefault(req.MaxOutputTokens),
		Temperature:   req.Temperature,
		StopSequences: req.StopSequences,
		TopP:          req.TopP,
		TopK:          req.TopK,
		Tools:         encodeAnthropicTools(req.Tools),
		Stream:        req.Stream,
		Metadata:      encodeAnthropicMetadata(req.Metadata),
	}
	request.OutputConfig = encodeAnthropicOutputConfig(req.ResponseFormat)
	request.Thinking = encodeAnthropicThinking(req.Reasoning, req.ReasoningBudgetTokens, request.MaxTokens)
	request.ToolChoice = encodeAnthropicToolChoice(sanitizeAnthropicToolChoice(req.ToolChoice, request.Thinking), req.ParallelToolCalls)

	for _, message := range req.Prompt {
		if message.Role == RoleSystem || message.Role == RoleDeveloper {
			request.System = appendSystemText(request.System, joinTextParts(message.Parts))
			continue
		}
		encoded := encodeAnthropicMessage(message)
		if message.Role == RoleTool {
			encoded.Role = string(RoleUser)
		}
		if len(message.Parts) == 0 {
			continue
		}
		request.Messages = append(request.Messages, encoded)
	}

	applyAnthropicCache(&request, req.Cache)
	return json.Marshal(request)
}

func sanitizeAnthropicToolChoice(choice *ToolChoice, thinking any) *ToolChoice {
	if thinking == nil || choice == nil {
		return choice
	}
	if choice.Type == ToolChoiceRequired || choice.Type == ToolChoiceTool {
		return &ToolChoice{Type: ToolChoiceAuto}
	}
	return choice
}

func appendSystemText(system any, text string) any {
	if strings.TrimSpace(text) == "" {
		return system
	}
	if current, ok := system.(string); ok && current != "" {
		return current + "\n" + text
	}
	if system == nil {
		return text
	}
	return system
}

func (a AnthropicMessagesAdapter) DecodeResponse(raw []byte) (*LLMResponse, error) {
	var response anthropicResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("decode anthropic messages response: %w", err)
	}

	return &LLMResponse{
		Protocol:         ProtocolAnthropicMessages,
		ID:               response.ID,
		Model:            response.Model,
		Role:             RoleAssistant,
		Content:          decodeAnthropicContent(response.Content),
		FinishReason:     decodeAnthropicStopReason(response.StopReason),
		Usage:            decodeAnthropicUsage(response.Usage),
		ProviderMetadata: map[string]any{"type": response.Type},
	}, nil
}

func (a AnthropicMessagesAdapter) EncodeResponse(resp *LLMResponse, opts EncodeResponseOptions) ([]byte, error) {
	if resp == nil {
		return nil, errors.New("encode anthropic messages response: nil response")
	}

	model := resp.Model
	if opts.Model != "" {
		model = opts.Model
	}

	content, finishReason := firstResponseContent(resp)
	response := anthropicResponse{
		ID:         resp.ID,
		Type:       "message",
		Role:       string(RoleAssistant),
		Model:      model,
		Content:    encodeAnthropicContent(content),
		StopReason: encodeAnthropicStopReason(finishReason),
		Usage:      encodeAnthropicUsage(resp.Usage, resp.BillingUsage()),
	}
	if finishReason == FinishContentFilter {
		response.StopDetails = &anthropicStopDetails{Type: "refusal", Explanation: firstRefusalText(content)}
	}

	return json.Marshal(response)
}

func (a AnthropicMessagesAdapter) NewStreamDecoder(StreamDecodeOptions) (StreamDecoder, error) {
	return &anthropicStreamDecoder{}, nil
}

func (a AnthropicMessagesAdapter) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &anthropicStreamEncoder{model: opts.Model}, nil
}

func (a AnthropicMessagesAdapter) EncodeError(err error) ([]byte, int) {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	raw, marshalErr := json.Marshal(anthropicErrorResponse{
		Type: "error",
		Error: anthropicError{
			Type:    "invalid_request_error",
			Message: message,
		},
	})
	if marshalErr != nil {
		return []byte(`{"type":"error","error":{"type":"api_error","message":"failed to encode error"}}`), http.StatusInternalServerError
	}
	return raw, http.StatusBadRequest
}

func decodeAnthropicMessage(message anthropicMessage) (Message, error) {
	parts, err := decodeAnthropicRawContent(asRawMessage(message.Content))
	if err != nil {
		return Message{}, err
	}
	return Message{Role: Role(message.Role), Parts: parts}, nil
}

type anthropicStreamDecoder struct {
	blockTypes map[int]StreamPartType
	toolIDs    map[int]string
}

func (d *anthropicStreamDecoder) Decode(event RawStreamEvent) ([]StreamPart, error) {
	if len(bytes.TrimSpace(event.Data)) == 0 {
		return nil, nil
	}

	var raw anthropicStreamEvent
	if err := json.Unmarshal(event.Data, &raw); err != nil {
		return nil, fmt.Errorf("decode anthropic stream event: %w", err)
	}

	switch raw.Type {
	case "message_start":
		part := StreamPart{Type: StreamStart}
		if raw.Message != nil {
			part.ID = raw.Message.ID
			part.Usage = decodeAnthropicUsage(raw.Message.Usage)
			part.ProviderMetadata = map[string]any{"model": raw.Message.Model, "role": raw.Message.Role}
		}
		return []StreamPart{part}, nil
	case "content_block_start":
		parts := decodeAnthropicContentBlockStart(raw)
		if len(parts) > 0 {
			index := intValueOrZero(raw.Index)
			d.setBlockType(index, parts[0].Type)
			if parts[0].Type == StreamToolInputStart {
				d.setToolID(index, parts[0].ToolCallID)
			}
		}
		return parts, nil
	case "content_block_delta":
		return decodeAnthropicContentBlockDelta(raw, d.toolID(intValueOrZero(raw.Index))), nil
	case "content_block_stop":
		index := intValueOrZero(raw.Index)
		return []StreamPart{{Type: d.streamEndType(index), ID: streamIndexID(index)}}, nil
	case "message_delta":
		usage := anthropicUsage{}
		if raw.Usage != nil {
			usage = *raw.Usage
		}
		finish := StreamPart{Type: StreamFinish, FinishReason: decodeAnthropicStopReason(raw.Delta.StopReason), Usage: decodeAnthropicUsage(usage)}
		return []StreamPart{finish}, nil
	case "message_stop":
		return nil, nil
	case "error":
		return []StreamPart{{Type: StreamError, Error: raw.Error}}, nil
	case "ping":
		return nil, nil
	default:
		return []StreamPart{{Type: StreamRaw, RawValue: raw}}, nil
	}
}

func (d *anthropicStreamDecoder) Close() ([]StreamPart, error) {
	return nil, nil
}

func (d *anthropicStreamDecoder) setBlockType(index int, partType StreamPartType) {
	if d.blockTypes == nil {
		d.blockTypes = make(map[int]StreamPartType)
	}
	d.blockTypes[index] = partType
}

func (d *anthropicStreamDecoder) setToolID(index int, toolID string) {
	if strings.TrimSpace(toolID) == "" {
		return
	}
	if d.toolIDs == nil {
		d.toolIDs = make(map[int]string)
	}
	d.toolIDs[index] = toolID
}

func (d *anthropicStreamDecoder) toolID(index int) string {
	if d.toolIDs == nil {
		return ""
	}
	return d.toolIDs[index]
}

func (d *anthropicStreamDecoder) streamEndType(index int) StreamPartType {
	if d.blockTypes == nil {
		return StreamTextEnd
	}
	partType := d.blockTypes[index]
	delete(d.blockTypes, index)
	delete(d.toolIDs, index)
	switch partType {
	case StreamReasoningStart:
		return StreamReasoningEnd
	case StreamToolInputStart:
		return StreamToolInputEnd
	default:
		return StreamTextEnd
	}
}

func decodeAnthropicContentBlockStart(raw anthropicStreamEvent) []StreamPart {
	block := raw.ContentBlock
	index := intValueOrZero(raw.Index)
	switch block.Type {
	case "text":
		return []StreamPart{{Type: StreamTextStart, ID: streamIndexID(index)}}
	case "thinking":
		return []StreamPart{{Type: StreamReasoningStart, ID: streamIndexID(index)}}
	case "redacted_thinking":
		return []StreamPart{{Type: StreamReasoningStart, ID: streamIndexID(index), Delta: block.Data, ProviderMetadata: map[string]any{"redacted": true}}}
	case "tool_use":
		return []StreamPart{{Type: StreamToolInputStart, ID: streamIndexID(index), ToolCallID: block.ID, ToolName: block.Name}}
	default:
		return []StreamPart{{Type: StreamRaw, ID: streamIndexID(index), RawValue: block}}
	}
}

func decodeAnthropicContentBlockDelta(raw anthropicStreamEvent, toolID string) []StreamPart {
	if raw.Delta == nil {
		return nil
	}
	index := intValueOrZero(raw.Index)
	switch raw.Delta.Type {
	case "text_delta":
		return []StreamPart{{Type: StreamTextDelta, ID: streamIndexID(index), Delta: raw.Delta.Text}}
	case "thinking_delta":
		return []StreamPart{{Type: StreamReasoningDelta, ID: streamIndexID(index), Delta: raw.Delta.Thinking}}
	case "signature_delta":
		return []StreamPart{{Type: StreamReasoningDelta, ID: streamIndexID(index), ProviderMetadata: map[string]any{"signature": raw.Delta.Signature}}}
	case "input_json_delta":
		return []StreamPart{{Type: StreamToolInputDelta, ID: streamIndexID(index), ToolCallID: toolID, Delta: raw.Delta.PartialJSON}}
	default:
		return []StreamPart{{Type: StreamRaw, ID: streamIndexID(index), RawValue: raw.Delta}}
	}
}

type anthropicStreamEncoder struct {
	model        string
	nextIndex    int
	activeText   map[string]int
	activeReason map[string]int
	activeTool   map[string]int
	toolNames    map[string]string
	toolInputs   map[string]string
	started      bool
	finished     bool
}

func (e *anthropicStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	if e.activeText == nil {
		e.activeText = make(map[string]int)
		e.activeReason = make(map[string]int)
		e.activeTool = make(map[string]int)
		e.toolNames = make(map[string]string)
		e.toolInputs = make(map[string]string)
	}

	switch part.Type {
	case StreamStart:
		e.started = true
		message := anthropicStreamMessage{ID: part.ID, Type: "message", Role: string(RoleAssistant), Model: e.model, Content: []anthropicContentBlock{}, StopReason: nil, StopSequence: nil, Usage: encodeAnthropicUsage(part.Usage, billingUsageForProtocol(ProtocolAnthropicMessages, part.Usage))}
		if message.ID == "" {
			message.ID = "msg_stream"
		}
		payload := map[string]any{
			"type": "message_start",
			"message": map[string]any{
				"id":            message.ID,
				"type":          message.Type,
				"role":          message.Role,
				"content":       []any{},
				"model":         message.Model,
				"stop_reason":   nil,
				"stop_sequence": nil,
				"usage":         message.Usage,
			},
		}
		return singleAnthropicStreamPayloadEvent("message_start", payload)
	case StreamTextStart:
		idx := e.ensureIndex(e.activeText, part.ID)
		payload := map[string]any{
			"type":  "content_block_start",
			"index": idx,
			"content_block": map[string]any{
				"type": "text",
				"text": "",
			},
		}
		return singleAnthropicStreamPayloadEvent("content_block_start", payload)
	case StreamTextDelta:
		idx := e.ensureIndex(e.activeText, part.ID)
		delta := anthropicStreamDelta{Type: "text_delta", Text: part.Delta}
		return singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &delta})
	case StreamTextEnd:
		idx, key, ok := e.existingIndexAndKey(e.activeText, part.ID)
		if !ok {
			return nil, nil
		}
		delete(e.activeText, key)
		return singleAnthropicStreamEvent("content_block_stop", anthropicStreamEvent{Type: "content_block_stop", Index: intPtr(idx)})
	case StreamReasoningStart:
		idx := e.ensureIndex(e.activeReason, part.ID)
		blockType := "thinking"
		if redacted, ok := part.ProviderMetadata["redacted"].(bool); ok && redacted {
			blockType = "redacted_thinking"
		}
		block := map[string]any{"type": blockType}
		if blockType == "redacted_thinking" {
			block["data"] = part.Delta
		} else {
			block["thinking"] = ""
			block["signature"] = ""
		}
		payload := map[string]any{
			"type":          "content_block_start",
			"index":         idx,
			"content_block": block,
		}
		return singleAnthropicStreamPayloadEvent("content_block_start", payload)
	case StreamReasoningDelta:
		idx := e.ensureIndex(e.activeReason, part.ID)
		if redacted, ok := part.ProviderMetadata["redacted"].(bool); ok && redacted {
			delta := anthropicStreamDelta{Type: "redacted_thinking_delta", Text: part.Delta}
			return singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &delta})
		}
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" {
			delta := anthropicStreamDelta{Type: "signature_delta", Signature: signature}
			return singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &delta})
		}
		delta := anthropicStreamDelta{Type: "thinking_delta", Thinking: part.Delta}
		return singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &delta})
	case StreamReasoningEnd:
		idx, key, ok := e.existingIndexAndKey(e.activeReason, part.ID)
		if !ok {
			return nil, nil
		}
		delete(e.activeReason, key)
		return singleAnthropicStreamEvent("content_block_stop", anthropicStreamEvent{Type: "content_block_stop", Index: intPtr(idx)})
	case StreamToolInputStart:
		idx := e.ensureIndex(e.activeTool, part.ToolCallID)
		e.toolNames[part.ToolCallID] = part.ToolName
		block := anthropicContentBlock{Type: "tool_use", ID: part.ToolCallID, Name: part.ToolName, Input: map[string]any{}}
		return singleAnthropicStreamEvent("content_block_start", anthropicStreamEvent{Type: "content_block_start", Index: intPtr(idx), ContentBlock: &block})
	case StreamToolInputDelta:
		idx := e.ensureIndex(e.activeTool, part.ToolCallID)
		e.toolInputs[part.ToolCallID] += part.Delta
		delta := anthropicStreamDelta{Type: "input_json_delta", PartialJSON: part.Delta}
		return singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &delta})
	case StreamToolInputEnd:
		idx, key, ok := e.existingIndexAndKey(e.activeTool, part.ToolCallID)
		if !ok && part.ID != part.ToolCallID {
			idx, key, ok = e.existingIndexAndKey(e.activeTool, part.ID)
		}
		if !ok {
			return nil, nil
		}
		delete(e.activeTool, key)
		return singleAnthropicStreamEvent("content_block_stop", anthropicStreamEvent{Type: "content_block_stop", Index: intPtr(idx)})
	case StreamToolCall:
		return e.encodeToolCall(part)
	case StreamFinish:
		e.finished = true
		return e.encodeFinish(part), nil
	case StreamError:
		return singleAnthropicStreamEvent("error", anthropicStreamEvent{Type: "error", Error: part.Error})
	case StreamRaw:
		delta := anthropicStreamDelta{Type: "raw", Text: fmt.Sprint(part.RawValue)}
		return singleAnthropicStreamEvent("raw", anthropicStreamEvent{Type: "raw", Delta: &delta})
	default:
		return nil, nil
	}
}

func (e *anthropicStreamEncoder) Close() ([]RawStreamEvent, error) {
	if e.finished {
		return nil, nil
	}
	return e.encodeFinish(StreamPart{Type: StreamFinish, FinishReason: FinishStop}), nil
}

func (e *anthropicStreamEncoder) EncodeError(err error) []RawStreamEvent {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	events, _ := singleAnthropicStreamEvent("error", anthropicStreamEvent{Type: "error", Error: anthropicError{Type: "api_error", Message: message}})
	return events
}

func (e *anthropicStreamEncoder) encodeToolCall(part StreamPart) ([]RawStreamEvent, error) {
	toolID := part.ToolCallID
	if toolID == "" {
		toolID = part.ID
	}
	idx := e.ensureIndex(e.activeTool, toolID)
	name := part.ToolName
	if name == "" {
		name = e.toolNames[toolID]
	}
	input, err := encodeOpenAIToolInput(part.Input)
	if err != nil {
		return nil, err
	}
	block := anthropicContentBlock{Type: "tool_use", ID: toolID, Name: name, Input: map[string]any{}}
	start, err := singleAnthropicStreamEvent("content_block_start", anthropicStreamEvent{Type: "content_block_start", Index: intPtr(idx), ContentBlock: &block})
	if err != nil {
		return nil, err
	}
	streamDelta := anthropicStreamDelta{Type: "input_json_delta", PartialJSON: input}
	delta, err := singleAnthropicStreamEvent("content_block_delta", anthropicStreamEvent{Type: "content_block_delta", Index: intPtr(idx), Delta: &streamDelta})
	if err != nil {
		return nil, err
	}
	stop, err := singleAnthropicStreamEvent("content_block_stop", anthropicStreamEvent{Type: "content_block_stop", Index: intPtr(idx)})
	if err != nil {
		return nil, err
	}
	delete(e.activeTool, toolID)
	return append(append(start, delta...), stop...), nil
}

func (e *anthropicStreamEncoder) encodeFinish(part StreamPart) []RawStreamEvent {
	usage := encodeAnthropicUsage(part.Usage, billingUsageForProtocol(ProtocolAnthropicMessages, part.Usage))
	if usage.OutputTokens == nil {
		outputTokens := 0
		usage.OutputTokens = &outputTokens
	}
	if part.FinishReason == "" {
		part.FinishReason = FinishStop
	}
	events := make([]RawStreamEvent, 0, 2)
	payload := map[string]any{
		"type": "message_delta",
		"delta": map[string]any{
			"stop_reason":   encodeAnthropicStopReason(part.FinishReason),
			"stop_sequence": nil,
		},
		"usage": usage,
	}
	deltaEvent, err := singleAnthropicStreamPayloadEvent("message_delta", payload)
	if err == nil {
		events = append(events, deltaEvent...)
	}
	stopEvent, err := singleAnthropicStreamEvent("message_stop", anthropicStreamEvent{Type: "message_stop"})
	if err == nil {
		events = append(events, stopEvent...)
	}
	return events
}

func (e *anthropicStreamEncoder) ensureIndex(indexes map[string]int, id string) int {
	index, _ := e.ensureIndexAndKey(indexes, id)
	return index
}

func (e *anthropicStreamEncoder) existingIndexAndKey(indexes map[string]int, id string) (int, string, bool) {
	if id == "" && len(indexes) == 1 {
		for key, index := range indexes {
			return index, key, true
		}
	}
	if id == "" {
		return 0, "", false
	}
	index, ok := indexes[id]
	return index, id, ok
}

func (e *anthropicStreamEncoder) ensureIndexAndKey(indexes map[string]int, id string) (int, string) {
	if id == "" && len(indexes) == 1 {
		for key, index := range indexes {
			return index, key
		}
	}
	if id == "" {
		id = fmt.Sprintf("idx_%d", e.nextIndex)
	}
	if index, ok := indexes[id]; ok {
		return index, id
	}
	index := e.nextIndex
	e.nextIndex++
	indexes[id] = index
	return index, id
}

func singleAnthropicStreamEvent(event string, payload anthropicStreamEvent) ([]RawStreamEvent, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return []RawStreamEvent{{Event: event, Data: raw}}, nil
}

func singleAnthropicStreamPayloadEvent(event string, payload any) ([]RawStreamEvent, error) {
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return []RawStreamEvent{{Event: event, Data: raw}}, nil
}

func streamIndexID(index int) string {
	return fmt.Sprintf("content_%d", index)
}

func intPtr(value int) *int {
	return &value
}

func intValueOrZero(value *int) int {
	if value == nil {
		return 0
	}
	return *value
}

func encodeAnthropicMessage(message Message) anthropicMessage {
	return anthropicMessage{
		Role:    string(message.Role),
		Content: encodeAnthropicContent(message.Parts),
	}
}

func decodeAnthropicRawContent(raw json.RawMessage) ([]Part, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return []Part{{Type: PartText, Text: &TextPart{Text: text}}}, nil
	}

	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil {
		return nil, fmt.Errorf("decode anthropic content: %w", err)
	}
	return decodeAnthropicContent(blocks), nil
}

func decodeAnthropicContent(blocks []anthropicContentBlock) []Part {
	parts := make([]Part, 0, len(blocks))
	for _, block := range blocks {
		switch block.Type {
		case "text":
			parts = append(parts, Part{Type: PartText, Text: &TextPart{Text: block.Text}})
		case "thinking":
			parts = append(parts, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Text: block.Thinking, Signature: block.Signature}})
		case "redacted_thinking":
			parts = append(parts, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Redacted: block.Data, Signature: block.Signature}})
		case "image":
			parts = append(parts, Part{Type: PartFile, File: decodeAnthropicFile(block, FileImage)})
		case "document":
			parts = append(parts, Part{Type: PartFile, File: decodeAnthropicFile(block, FileDocument)})
		case "tool_use":
			parts = append(parts, Part{Type: PartToolCall, ToolCall: &ToolCallPart{ToolCallID: block.ID, ToolName: block.Name, Input: block.Input}})
		case "tool_result":
			parts = append(parts, Part{Type: PartToolResult, ToolResult: &ToolResultPart{ToolCallID: block.ToolUseID, Output: decodeAnthropicToolResult(block)}})
		}
	}
	return parts
}

func encodeAnthropicContent(parts []Part) []anthropicContentBlock {
	blocks := make([]anthropicContentBlock, 0, len(parts))
	for _, part := range parts {
		switch part.Type {
		case PartText:
			if part.Text != nil {
				blocks = append(blocks, anthropicContentBlock{Type: "text", Text: part.Text.Text})
			}
		case PartReasoning:
			if part.Reasoning != nil {
				if part.Reasoning.Redacted != "" {
					blocks = append(blocks, anthropicContentBlock{Type: "redacted_thinking", Data: part.Reasoning.Redacted})
				} else {
					blocks = append(blocks, anthropicContentBlock{Type: "thinking", Thinking: part.Reasoning.Text, Signature: part.Reasoning.Signature})
				}
			}
		case PartRefusal:
			if part.Refusal != nil {
				blocks = append(blocks, anthropicContentBlock{Type: "text", Text: part.Refusal.Text})
			}
		case PartFile:
			if part.File != nil {
				if block, ok := encodeAnthropicFile(part.File); ok {
					blocks = append(blocks, block)
				} else {
					blocks = append(blocks, anthropicContentBlock{Type: "text", Text: unsupportedFileWarningText(part.File)})
				}
			}
		case PartToolCall:
			if part.ToolCall != nil {
				blocks = append(blocks, anthropicContentBlock{Type: "tool_use", ID: part.ToolCall.ToolCallID, Name: part.ToolCall.ToolName, Input: part.ToolCall.Input})
			}
		case PartToolResult:
			if part.ToolResult != nil {
				blocks = append(blocks, encodeAnthropicToolResult(part.ToolResult))
			}
		}
	}
	return blocks
}

func decodeAnthropicFile(block anthropicContentBlock, fileType FilePartType) *FilePart {
	file := &FilePart{Type: fileType, Filename: block.Title}
	if source, ok := block.Source.(map[string]any); ok {
		if mediaType, ok := source["media_type"].(string); ok {
			file.MediaType = mediaType
		}
		if data, ok := source["data"].(string); ok {
			file.Data = data
		}
		if url, ok := source["url"].(string); ok {
			file.URL = url
		}
		if fileID, ok := source["file_id"].(string); ok {
			file.FileID = fileID
		}
	}
	return file
}

func encodeAnthropicFile(file *FilePart) (anthropicContentBlock, bool) {
	blockType := string(file.Type)
	if blockType == "" {
		blockType = "document"
	}
	source := map[string]any{}
	if file.URL != "" {
		source["type"] = "url"
		source["url"] = file.URL
	} else if file.Data != "" {
		if file.Type == FileImage {
			source["type"] = "base64"
			source["media_type"] = file.MediaType
			source["data"] = file.Data
		} else if file.MediaType == "application/pdf" {
			source["type"] = "base64"
			source["media_type"] = file.MediaType
			source["data"] = file.Data
		} else if file.MediaType == "text/plain" {
			source["type"] = "text"
			source["media_type"] = file.MediaType
			source["data"] = file.Data
		} else {
			return anthropicContentBlock{}, false
		}
	} else {
		return anthropicContentBlock{}, false
	}
	return anthropicContentBlock{Type: blockType, Source: source, Title: file.Filename}, true
}

func unsupportedFileWarningText(file *FilePart) string {
	if file == nil {
		return "[Proxy warning: an unsupported file input was omitted while converting to Anthropic Messages.]"
	}
	identifier := strings.TrimSpace(file.Filename)
	if identifier == "" {
		identifier = strings.TrimSpace(file.FileID)
	}
	if identifier == "" {
		identifier = strings.TrimSpace(file.MediaType)
	}
	if identifier == "" {
		identifier = "unknown file"
	}
	return fmt.Sprintf("[Proxy warning: file input %q could not be converted to Anthropic Messages and was omitted. Provide a URL, base64 image, PDF, or plain text content instead.]", identifier)
}

func decodeAnthropicToolResult(block anthropicContentBlock) ToolResultOutput {
	outputType := ToolResultText
	if block.IsError {
		outputType = ToolResultErrorText
	}
	if contentBlocks, ok := block.Content.([]any); ok {
		parts := make([]Part, 0, len(contentBlocks))
		for _, item := range contentBlocks {
			raw, err := json.Marshal(item)
			if err != nil {
				continue
			}
			var contentBlock anthropicContentBlock
			if err := json.Unmarshal(raw, &contentBlock); err != nil {
				continue
			}
			parts = append(parts, decodeAnthropicContent([]anthropicContentBlock{contentBlock})...)
		}
		return ToolResultOutput{Type: ToolResultContent, Content: parts}
	}
	text, _ := block.Content.(string)
	return ToolResultOutput{Type: outputType, Text: text}
}

func encodeAnthropicToolResult(result *ToolResultPart) anthropicContentBlock {
	block := anthropicContentBlock{Type: "tool_result", ToolUseID: result.ToolCallID}
	if result.Output.Type == ToolResultErrorText || result.Output.Type == ToolResultErrorJSON {
		block.IsError = true
	}
	if result.Output.Type == ToolResultContent {
		block.Content = encodeAnthropicContent(result.Output.Content)
	} else {
		block.Content = encodeToolResultText(result.Output)
	}
	return block
}

func decodeAnthropicThinking(raw any) *bool {
	if len(asRawMessage(raw)) == 0 || string(asRawMessage(raw)) == "null" {
		return nil
	}
	var thinking struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(asRawMessage(raw), &thinking); err != nil {
		return nil
	}
	enabled := thinking.Type == "enabled"
	return &enabled
}

func decodeAnthropicThinkingBudget(raw any) *int {
	if len(asRawMessage(raw)) == 0 || string(asRawMessage(raw)) == "null" {
		return nil
	}
	var thinking struct {
		Type         string `json:"type"`
		BudgetTokens *int   `json:"budget_tokens"`
	}
	if err := json.Unmarshal(asRawMessage(raw), &thinking); err != nil || thinking.Type != "enabled" {
		return nil
	}
	return thinking.BudgetTokens
}

func decodeAnthropicReasoningEffort(thinking any) string {
	if len(asRawMessage(thinking)) == 0 || string(asRawMessage(thinking)) == "null" {
		return ""
	}
	var decoded struct {
		Type string `json:"type"`
	}
	if err := json.Unmarshal(asRawMessage(thinking), &decoded); err != nil {
		return ""
	}
	switch decoded.Type {
	case "adaptive":
		return "high"
	case "disabled":
		return "none"
	default:
		return ""
	}
}

func encodeAnthropicThinking(reasoning *bool, budgetTokens *int, maxTokens *int) any {
	if reasoning == nil || !*reasoning {
		return nil
	}
	budget := anthropicThinkingBudgetTokens(budgetTokens, maxTokens)
	if budget == nil {
		return nil
	}
	return map[string]any{"type": "enabled", "budget_tokens": budget}
}

func decodeAnthropicCache(request anthropicRequest) *bool {
	if anthropicContentHasCacheControl(request.System) {
		enabled := true
		return &enabled
	}
	for _, message := range request.Messages {
		if anthropicContentHasCacheControl(message.Content) {
			enabled := true
			return &enabled
		}
	}
	return nil
}

func anthropicContentHasCacheControl(content any) bool {
	raw := asRawMessage(content)
	if len(raw) == 0 || string(raw) == "null" {
		return false
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err == nil {
		for _, block := range blocks {
			if block.CacheControl != nil {
				return true
			}
		}
	}
	return false
}

func applyAnthropicCache(request *anthropicRequest, cache *bool) {
	if request == nil || (cache != nil && !*cache) {
		return
	}
	cacheControl := &anthropicCacheControl{Type: "ephemeral"}
	if applyAnthropicCacheToSystem(request, cacheControl) {
		return
	}
	applyAnthropicCacheToMessages(request.Messages, cacheControl)
}

func applyAnthropicCacheToSystem(request *anthropicRequest, cacheControl *anthropicCacheControl) bool {
	if request == nil || request.System == nil {
		return false
	}
	if system, ok := request.System.(string); ok {
		if strings.TrimSpace(system) == "" {
			return false
		}
		request.System = []anthropicContentBlock{{Type: "text", Text: system, CacheControl: cacheControl}}
		return true
	}
	blocks, ok := anthropicContentBlocks(request.System)
	if !ok || len(blocks) == 0 {
		return false
	}
	blocks[len(blocks)-1].CacheControl = cacheControl
	request.System = blocks
	return true
}

func applyAnthropicCacheToMessages(messages []anthropicMessage, cacheControl *anthropicCacheControl) {
	for i := len(messages) - 1; i >= 0; i-- {
		blocks, ok := anthropicContentBlocks(messages[i].Content)
		if !ok || len(blocks) == 0 {
			continue
		}
		blocks[len(blocks)-1].CacheControl = cacheControl
		messages[i].Content = blocks
		return
	}
}

func anthropicContentBlocks(content any) ([]anthropicContentBlock, bool) {
	raw := asRawMessage(content)
	if len(raw) == 0 || string(raw) == "null" {
		return nil, false
	}
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if strings.TrimSpace(text) == "" {
			return nil, false
		}
		return []anthropicContentBlock{{Type: "text", Text: text}}, true
	}
	var blocks []anthropicContentBlock
	if err := json.Unmarshal(raw, &blocks); err != nil || len(blocks) == 0 {
		return nil, false
	}
	return blocks, true
}

func decodeAnthropicOutputConfig(config *anthropicOutputConfig) *ResponseFormat {
	if config == nil || config.Format == nil {
		return nil
	}
	if config.Format.Type != "json_schema" {
		return nil
	}
	return &ResponseFormat{Type: ResponseFormatJSON, Schema: config.Format.Schema, Strict: config.Format.Strict}
}

func encodeAnthropicOutputConfig(format *ResponseFormat) *anthropicOutputConfig {
	if format == nil || format.Type != ResponseFormatJSON || format.Schema == nil {
		return nil
	}
	return &anthropicOutputConfig{Format: &anthropicOutputFormat{Type: "json_schema", Schema: format.Schema, Strict: format.Strict}}
}

func anthropicThinkingBudgetTokens(budgetTokens *int, maxTokens *int) *int {
	maxOutputTokens := defaultMaxOutputTokens
	if maxTokens != nil && *maxTokens > 0 {
		maxOutputTokens = *maxTokens
	}
	budget := defaultThinkingBudgetTokens
	if budgetTokens != nil {
		budget = *budgetTokens
	}
	if budget < minAnthropicThinkingBudgetTokens || budget >= maxOutputTokens {
		return nil
	}
	return &budget
}

func encodeToolResultText(output ToolResultOutput) string {
	if output.Type == ToolResultContent {
		return joinTextParts(output.Content)
	}
	if output.Type == ToolResultJSON || output.Type == ToolResultErrorJSON {
		raw, err := json.Marshal(output.JSON)
		if err != nil {
			return "null"
		}
		return string(raw)
	}
	return output.Text
}

func decodeAnthropicTools(tools []anthropicTool) []Tool {
	decoded := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "" && tool.Type != "custom" {
			continue
		}
		decoded = append(decoded, Tool{Type: ToolFunction, Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, Strict: tool.Strict})
	}
	return decoded
}

func encodeAnthropicTools(tools []Tool) []anthropicTool {
	encoded := make([]anthropicTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != ToolFunction {
			continue
		}
		encoded = append(encoded, anthropicTool{Name: tool.Name, Description: tool.Description, InputSchema: tool.InputSchema, Strict: tool.Strict})
	}
	return encoded
}

func unsupportedAnthropicToolWarnings(tools []Tool) []string {
	warnings := make([]string, 0)
	for _, tool := range tools {
		if tool.Type == ToolFunction {
			continue
		}
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			name = string(tool.Type)
		}
		if name == "" {
			name = "provider-defined tool"
		}
		warnings = append(warnings, fmt.Sprintf("OpenAI Responses tool %q is not available when proxying to Anthropic and was omitted.", name))
	}
	return warnings
}

func decodeAnthropicToolChoice(choice *anthropicToolChoice) *ToolChoice {
	if choice == nil {
		return nil
	}
	switch choice.Type {
	case "auto":
		return &ToolChoice{Type: ToolChoiceAuto}
	case "none":
		return &ToolChoice{Type: ToolChoiceNone}
	case "any":
		return &ToolChoice{Type: ToolChoiceRequired}
	case "tool":
		return &ToolChoice{Type: ToolChoiceTool, ToolName: choice.Name}
	default:
		return nil
	}
}

func decodeAnthropicParallelToolCalls(choice *anthropicToolChoice) *bool {
	if choice == nil || choice.DisableParallelToolUse == nil {
		return nil
	}
	enabled := !*choice.DisableParallelToolUse
	return &enabled
}

func decodeAnthropicMetadata(metadata *anthropicMetadata) map[string]string {
	if metadata == nil || strings.TrimSpace(metadata.UserID) == "" {
		return nil
	}
	return map[string]string{"user_id": metadata.UserID}
}

func encodeAnthropicMetadata(metadata map[string]string) *anthropicMetadata {
	if strings.TrimSpace(metadata["user_id"]) == "" {
		return nil
	}
	return &anthropicMetadata{UserID: metadata["user_id"]}
}

func encodeAnthropicToolChoice(choice *ToolChoice, parallelToolCalls *bool) *anthropicToolChoice {
	if choice == nil && parallelToolCalls == nil {
		return nil
	}
	encoded := &anthropicToolChoice{Type: "auto"}
	if choice == nil {
		return withAnthropicParallelToolCalls(encoded, parallelToolCalls)
	}
	switch choice.Type {
	case ToolChoiceAuto:
		encoded = &anthropicToolChoice{Type: "auto"}
	case ToolChoiceNone:
		encoded = &anthropicToolChoice{Type: "none"}
	case ToolChoiceRequired:
		encoded = &anthropicToolChoice{Type: "any"}
	case ToolChoiceTool:
		encoded = &anthropicToolChoice{Type: "tool", Name: choice.ToolName}
	default:
		return nil
	}
	return withAnthropicParallelToolCalls(encoded, parallelToolCalls)
}

func withAnthropicParallelToolCalls(choice *anthropicToolChoice, parallelToolCalls *bool) *anthropicToolChoice {
	if choice == nil || parallelToolCalls == nil {
		return choice
	}
	disabled := !*parallelToolCalls
	choice.DisableParallelToolUse = &disabled
	return choice
}

func decodeAnthropicStopReason(reason string) FinishReason {
	switch reason {
	case "end_turn", "stop_sequence":
		return FinishStop
	case "max_tokens":
		return FinishLength
	case "tool_use":
		return FinishToolCalls
	case "refusal":
		return FinishContentFilter
	case "":
		return FinishUnknown
	default:
		return FinishOther
	}
}

func encodeAnthropicStopReason(reason FinishReason) string {
	switch reason {
	case FinishStop:
		return "end_turn"
	case FinishLength:
		return "max_tokens"
	case FinishToolCalls:
		return "tool_use"
	case FinishContentFilter:
		return "refusal"
	default:
		return "end_turn"
	}
}

func firstRefusalText(parts []Part) string {
	for _, part := range parts {
		if part.Type == PartRefusal && part.Refusal != nil {
			return strings.TrimSpace(part.Refusal.Text)
		}
	}
	return ""
}

func decodeAnthropicUsage(usage anthropicUsage) Usage {
	return Usage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens, CachedInputTokens: usage.CacheReadInputTokens, CacheCreationInputTokens: usage.CacheCreationInputTokens, CacheReadInputTokens: usage.CacheReadInputTokens}
}

func encodeAnthropicUsage(usage Usage, billingUsage BillingUsage) anthropicUsage {
	if hasBillingUsage(billingUsage) {
		inputTokens := billingUsage.InputTokens
		cachedInputTokens := billingUsage.CachedInputTokens
		outputTokens := billingUsage.OutputTokens
		return anthropicUsage{InputTokens: &inputTokens, OutputTokens: &outputTokens, CacheCreationInputTokens: usage.CacheCreationInputTokens, CacheReadInputTokens: &cachedInputTokens}
	}
	cacheReadTokens := usage.CacheReadInputTokens
	if cacheReadTokens == nil {
		cacheReadTokens = usage.CachedInputTokens
	}
	return anthropicUsage{InputTokens: usage.InputTokens, OutputTokens: usage.OutputTokens, CacheCreationInputTokens: usage.CacheCreationInputTokens, CacheReadInputTokens: cacheReadTokens}
}

type anthropicRequest struct {
	Model         string                 `json:"model"`
	System        any                    `json:"system,omitempty"`
	Messages      []anthropicMessage     `json:"messages"`
	MaxTokens     *int                   `json:"max_tokens,omitempty"`
	Temperature   *float64               `json:"temperature,omitempty"`
	StopSequences []string               `json:"stop_sequences,omitempty"`
	TopP          *float64               `json:"top_p,omitempty"`
	TopK          *int                   `json:"top_k,omitempty"`
	OutputConfig  *anthropicOutputConfig `json:"output_config,omitempty"`
	Tools         []anthropicTool        `json:"tools,omitempty"`
	ToolChoice    *anthropicToolChoice   `json:"tool_choice,omitempty"`
	Stream        bool                   `json:"stream,omitempty"`
	Thinking      any                    `json:"thinking,omitempty"`
	Metadata      *anthropicMetadata     `json:"metadata,omitempty"`
}

type anthropicMetadata struct {
	UserID string `json:"user_id,omitempty"`
}

func (r anthropicRequest) systemText() string {
	if system, ok := r.System.(string); ok {
		return filterAnthropicSystemText([]Part{{Type: PartText, Text: &TextPart{Text: system}}})
	}
	raw, err := json.Marshal(r.System)
	if err != nil {
		return ""
	}
	parts, err := decodeAnthropicRawContent(raw)
	if err != nil {
		return ""
	}
	return filterAnthropicSystemText(parts)
}

func filterAnthropicSystemText(parts []Part) string {
	texts := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type != PartText || part.Text == nil {
			texts = append(texts, part)
			continue
		}
		if isClaudeCodeVolatileSystemPrompt(part.Text.Text) {
			continue
		}
		texts = append(texts, part)
	}
	return joinTextParts(texts)
}

func isClaudeCodeVolatileSystemPrompt(text string) bool {
	return claudeCodeVolatileSystemPromptPattern.MatchString(strings.TrimSpace(text))
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}

func (m *anthropicMessage) UnmarshalJSON(raw []byte) error {
	var decoded struct {
		Role    string          `json:"role"`
		Content json.RawMessage `json:"content"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	m.Role = decoded.Role
	m.Content = decoded.Content
	return nil
}

type anthropicContentBlock struct {
	Type         string                 `json:"type"`
	Text         string                 `json:"text,omitempty"`
	Thinking     string                 `json:"thinking,omitempty"`
	Data         string                 `json:"data,omitempty"`
	Signature    string                 `json:"signature,omitempty"`
	ID           string                 `json:"id,omitempty"`
	Name         string                 `json:"name,omitempty"`
	Input        any                    `json:"input,omitempty"`
	ToolUseID    string                 `json:"tool_use_id,omitempty"`
	Content      any                    `json:"content,omitempty"`
	IsError      bool                   `json:"is_error,omitempty"`
	Source       any                    `json:"source,omitempty"`
	Title        string                 `json:"title,omitempty"`
	CacheControl *anthropicCacheControl `json:"cache_control,omitempty"`
}

type anthropicCacheControl struct {
	Type string `json:"type"`
	TTL  string `json:"ttl,omitempty"`
}

type anthropicTool struct {
	Type        string         `json:"type,omitempty"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
	Raw         map[string]any `json:"-"`
}

func (t *anthropicTool) UnmarshalJSON(raw []byte) error {
	var decoded map[string]any
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	if value, ok := decoded["type"].(string); ok {
		t.Type = value
	}
	if value, ok := decoded["name"].(string); ok {
		t.Name = value
	}
	if value, ok := decoded["description"].(string); ok {
		t.Description = value
	}
	if value, ok := decoded["input_schema"].(map[string]any); ok {
		t.InputSchema = value
	}
	if value, ok := decoded["strict"].(bool); ok {
		t.Strict = &value
	}
	t.Raw = decoded
	return nil
}

func (t anthropicTool) MarshalJSON() ([]byte, error) {
	if len(t.Raw) > 0 && t.Type != "" && t.Type != "custom" {
		config := make(map[string]any, len(t.Raw)+4)
		for key, value := range t.Raw {
			config[key] = value
		}
		config["type"] = t.Type
		if t.Name != "" {
			config["name"] = t.Name
		}
		if t.Strict != nil {
			config["strict"] = *t.Strict
		}
		return json.Marshal(config)
	}
	type alias anthropicTool
	return json.Marshal(struct {
		alias
		Raw any `json:"-"`
	}{alias: alias(t)})
}

type anthropicToolChoice struct {
	Type                   string `json:"type"`
	Name                   string `json:"name,omitempty"`
	DisableParallelToolUse *bool  `json:"disable_parallel_tool_use,omitempty"`
}

type anthropicOutputConfig struct {
	Format *anthropicOutputFormat `json:"format,omitempty"`
}

type anthropicOutputFormat struct {
	Type   string         `json:"type"`
	Schema map[string]any `json:"schema,omitempty"`
	Strict *bool          `json:"strict,omitempty"`
}

type anthropicResponse struct {
	ID          string                  `json:"id,omitempty"`
	Type        string                  `json:"type"`
	Role        string                  `json:"role"`
	Model       string                  `json:"model"`
	Content     []anthropicContentBlock `json:"content"`
	StopReason  string                  `json:"stop_reason,omitempty"`
	StopDetails *anthropicStopDetails   `json:"stop_details,omitempty"`
	Usage       anthropicUsage          `json:"usage,omitempty"`
}

type anthropicStopDetails struct {
	Type        string `json:"type"`
	Category    any    `json:"category,omitempty"`
	Explanation string `json:"explanation,omitempty"`
}

type anthropicUsage struct {
	InputTokens              *int `json:"input_tokens,omitempty"`
	OutputTokens             *int `json:"output_tokens,omitempty"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
}

type anthropicStreamEvent struct {
	Type         string                  `json:"type"`
	Index        *int                    `json:"index,omitempty"`
	Message      *anthropicStreamMessage `json:"message,omitempty"`
	ContentBlock *anthropicContentBlock  `json:"content_block,omitempty"`
	Delta        *anthropicStreamDelta   `json:"delta,omitempty"`
	Usage        *anthropicUsage         `json:"usage,omitempty"`
	Error        any                     `json:"error,omitempty"`
}

type anthropicStreamMessage struct {
	ID           string                  `json:"id,omitempty"`
	Type         string                  `json:"type,omitempty"`
	Role         string                  `json:"role,omitempty"`
	Model        string                  `json:"model,omitempty"`
	Content      []anthropicContentBlock `json:"content,omitempty"`
	StopReason   any                     `json:"stop_reason,omitempty"`
	StopSequence any                     `json:"stop_sequence,omitempty"`
	Usage        anthropicUsage          `json:"usage,omitempty"`
}

type anthropicStreamDelta struct {
	Type        string `json:"type,omitempty"`
	Text        string `json:"text,omitempty"`
	Thinking    string `json:"thinking,omitempty"`
	Signature   string `json:"signature,omitempty"`
	PartialJSON string `json:"partial_json,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

type anthropicErrorResponse struct {
	Type  string         `json:"type"`
	Error anthropicError `json:"error"`
}

type anthropicError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}
