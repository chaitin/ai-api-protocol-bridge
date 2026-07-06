package protocolbridge

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
)

var ErrStreamUnsupported = errors.New("protocolbridge: stream conversion is not implemented")

type OpenAIChatAdapter struct{}

func NewOpenAIChatAdapter() OpenAIChatAdapter {
	return OpenAIChatAdapter{}
}

func (a OpenAIChatAdapter) Protocol() Protocol {
	return ProtocolOpenAIChat
}

func (a OpenAIChatAdapter) DecodeRequest(raw []byte) (*LLMRequest, error) {
	var request openAIChatRequest
	if err := json.Unmarshal(raw, &request); err != nil {
		return nil, fmt.Errorf("decode openai chat request: %w", err)
	}

	maxOutputTokens := request.MaxTokens
	if request.MaxCompletionTokens != nil {
		maxOutputTokens = request.MaxCompletionTokens
	}
	maxOutputTokens = maxOutputTokensOrDefault(maxOutputTokens)

	llmRequest := &LLMRequest{
		Protocol:          ProtocolOpenAIChat,
		Model:             request.Model,
		Prompt:            make([]Message, 0, len(request.Messages)),
		MaxOutputTokens:   maxOutputTokens,
		Temperature:       request.Temperature,
		StopSequences:     decodeOpenAIStop(asRawMessage(request.Stop)),
		TopP:              request.TopP,
		PresencePenalty:   request.PresencePenalty,
		FrequencyPenalty:  request.FrequencyPenalty,
		Seed:              request.Seed,
		CandidateCount:    request.N,
		ResponseFormat:    decodeOpenAIResponseFormat(asRawMessage(request.ResponseFormat)),
		Reasoning:         decodeReasoningEffort(request.ReasoningEffort),
		Tools:             decodeOpenAITools(request.Tools),
		ToolChoice:        decodeOpenAIToolChoice(asRawMessage(request.ToolChoice)),
		ParallelToolCalls: request.ParallelToolCalls,
		Stream:            request.Stream,
	}

	for _, message := range request.Messages {
		decoded, err := decodeOpenAIChatMessage(message)
		if err != nil {
			return nil, err
		}
		llmRequest.Prompt = append(llmRequest.Prompt, decoded)
	}

	return llmRequest, nil
}

func (a OpenAIChatAdapter) EncodeRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, errors.New("encode openai chat request: nil request")
	}

	model := req.Model
	if opts.Model != "" {
		model = opts.Model
	}

	request := openAIChatRequest{
		Model:               model,
		MaxCompletionTokens: maxOutputTokensOrDefault(req.MaxOutputTokens),
		Temperature:         req.Temperature,
		Stop:                encodeOpenAIStop(req.StopSequences),
		TopP:                req.TopP,
		PresencePenalty:     req.PresencePenalty,
		FrequencyPenalty:    req.FrequencyPenalty,
		Seed:                req.Seed,
		N:                   req.CandidateCount,
		ResponseFormat:      encodeOpenAIResponseFormat(req.ResponseFormat),
		ReasoningEffort:     encodeReasoningEffort(req.Reasoning),
		StreamOptions:       encodeOpenAIStreamOptions(req.Stream),
		Tools:               encodeOpenAITools(req.Tools),
		ToolChoice:          encodeOpenAIToolChoice(req.ToolChoice),
		ParallelToolCalls:   req.ParallelToolCalls,
		Stream:              req.Stream,
	}
	for _, message := range req.Prompt {
		encoded, err := encodeOpenAIChatMessages(message)
		if err != nil {
			return nil, err
		}
		request.Messages = append(request.Messages, encoded...)
	}

	return json.Marshal(request)
}

func (a OpenAIChatAdapter) DecodeResponse(raw []byte) (*LLMResponse, error) {
	var response openAIChatResponse
	if err := json.Unmarshal(raw, &response); err != nil {
		return nil, fmt.Errorf("decode openai chat response: %w", err)
	}
	if len(response.Choices) == 0 {
		return nil, errors.New("decode openai chat response: choices is empty")
	}

	choices := make([]LLMChoice, 0, len(response.Choices))
	for _, choice := range response.Choices {
		decoded, err := decodeOpenAIChatMessage(choice.Message)
		if err != nil {
			return nil, err
		}
		choices = append(choices, LLMChoice{Index: choice.Index, Role: decoded.Role, Content: decoded.Parts, FinishReason: decodeOpenAIFinishReason(choice.FinishReason)})
	}

	firstChoice := choices[0]
	if firstChoice.Role == "" {
		firstChoice.Role = RoleAssistant
	}
	decodedContent := firstChoice.Content

	return &LLMResponse{
		Protocol:     ProtocolOpenAIChat,
		ID:           response.ID,
		Model:        response.Model,
		Role:         firstChoice.Role,
		Content:      decodedContent,
		Choices:      choices,
		FinishReason: firstChoice.FinishReason,
		Usage:        decodeOpenAIUsage(response.Usage),
		ProviderMetadata: map[string]any{
			"object":  response.Object,
			"created": response.Created,
		},
	}, nil
}

func (a OpenAIChatAdapter) EncodeResponse(resp *LLMResponse, opts EncodeResponseOptions) ([]byte, error) {
	if resp == nil {
		return nil, errors.New("encode openai chat response: nil response")
	}

	model := resp.Model
	if opts.Model != "" {
		model = opts.Model
	}

	choices := make([]openAIChatChoice, 0)
	if len(resp.Choices) > 0 {
		for _, choice := range resp.Choices {
			message, err := encodeOpenAIAssistantMessage(choice.Content)
			if err != nil {
				return nil, err
			}
			choices = append(choices, openAIChatChoice{Index: choice.Index, Message: message, FinishReason: encodeOpenAIFinishReason(choice.FinishReason)})
		}
	} else {
		message, err := encodeOpenAIAssistantMessage(resp.Content)
		if err != nil {
			return nil, err
		}
		choices = append(choices, openAIChatChoice{Index: 0, Message: message, FinishReason: encodeOpenAIFinishReason(resp.FinishReason)})
	}

	response := openAIChatResponse{
		ID:      resp.ID,
		Object:  "chat.completion",
		Model:   model,
		Choices: choices,
		Usage:   encodeOpenAIUsage(resp.Usage, resp.BillingUsage()),
	}

	return json.Marshal(response)
}

func (a OpenAIChatAdapter) NewStreamDecoder(StreamDecodeOptions) (StreamDecoder, error) {
	return &openAIChatStreamDecoder{}, nil
}

func (a OpenAIChatAdapter) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &openAIChatStreamEncoder{model: opts.Model, toolIndexes: make(map[string]int)}, nil
}

type openAIChatStreamChunk struct {
	ID      string                      `json:"id"`
	Object  string                      `json:"object"`
	Created int64                       `json:"created"`
	Model   string                      `json:"model"`
	Choices []openAIChatStreamChoice    `json:"choices"`
	Usage   *openAIChatStreamChunkUsage `json:"usage,omitempty"`
}

type openAIChatStreamChoice struct {
	Index        int                    `json:"index"`
	Delta        *openAIChatStreamDelta `json:"delta"`
	FinishReason *string                `json:"finish_reason"`
}

type openAIChatStreamDelta struct {
	Role      string                     `json:"role,omitempty"`
	Content   *string                    `json:"content,omitempty"`
	Reasoning *string                    `json:"reasoning_content,omitempty"`
	Refusal   *string                    `json:"refusal,omitempty"`
	ToolCalls []openAIChatStreamToolCall `json:"tool_calls,omitempty"`
}

type openAIChatStreamToolCall struct {
	Index    int                              `json:"index"`
	ID       string                           `json:"id,omitempty"`
	Type     string                           `json:"type,omitempty"`
	Function openAIChatStreamToolCallFunction `json:"function"`
}

type openAIChatStreamToolCallFunction struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

type openAIChatStreamChunkUsage struct {
	PromptTokens            *int                           `json:"prompt_tokens,omitempty"`
	CompletionTokens        *int                           `json:"completion_tokens,omitempty"`
	TotalTokens             *int                           `json:"total_tokens,omitempty"`
	PromptTokensDetails     *openAIPromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *openAICompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openAIChatStreamDecoder struct {
	started bool
}

func (d *openAIChatStreamDecoder) Decode(event RawStreamEvent) ([]StreamPart, error) {
	if len(bytes.TrimSpace(event.Data)) == 0 {
		return nil, nil
	}
	if string(event.Data) == "[DONE]" {
		return nil, nil
	}

	var chunk openAIChatStreamChunk
	if err := json.Unmarshal(event.Data, &chunk); err != nil {
		return []StreamPart{{Type: StreamRaw, RawValue: string(event.Data)}}, nil
	}

	parts := make([]StreamPart, 0)
	if !d.started {
		d.started = true
		part := StreamPart{Type: StreamStart, ID: chunk.ID, ProviderMetadata: map[string]any{"model": chunk.Model}}
		if chunk.Usage != nil {
			part.Usage = decodeOpenAIUsage(openAIUsage{
				PromptTokens:            chunk.Usage.PromptTokens,
				CompletionTokens:        chunk.Usage.CompletionTokens,
				TotalTokens:             chunk.Usage.TotalTokens,
				PromptTokensDetails:     chunk.Usage.PromptTokensDetails,
				CompletionTokensDetails: chunk.Usage.CompletionTokensDetails,
			})
		}
		parts = append(parts, part)
	}

	for _, choice := range chunk.Choices {
		if choice.Delta == nil {
			continue
		}
		if choice.Delta.Role != "" {
			parts = append(parts, StreamPart{Type: StreamStart, ID: chunk.ID, ProviderMetadata: map[string]any{"role": choice.Delta.Role}})
		}
		if choice.Delta.Reasoning != nil && *choice.Delta.Reasoning != "" {
			parts = append(parts, StreamPart{Type: StreamReasoningDelta, ID: streamIndexID(choice.Index), Delta: *choice.Delta.Reasoning})
		}
		if choice.Delta.Content != nil && *choice.Delta.Content != "" {
			parts = append(parts, StreamPart{Type: StreamTextDelta, ID: streamIndexID(choice.Index), Delta: *choice.Delta.Content})
		}
		if choice.Delta.Refusal != nil && *choice.Delta.Refusal != "" {
			parts = append(parts, StreamPart{Type: StreamTextDelta, ID: streamIndexID(choice.Index), Delta: *choice.Delta.Refusal, ProviderMetadata: map[string]any{"refusal": true}})
		}
		for _, tc := range choice.Delta.ToolCalls {
			if tc.ID != "" {
				parts = append(parts, StreamPart{Type: StreamToolInputStart, ID: streamIndexID(tc.Index), ToolCallID: tc.ID, ToolName: tc.Function.Name})
			}
			if tc.Function.Arguments != "" {
				parts = append(parts, StreamPart{Type: StreamToolInputDelta, ID: streamIndexID(tc.Index), ToolCallID: tc.ID, Delta: tc.Function.Arguments})
			}
		}
		if choice.FinishReason != nil && *choice.FinishReason != "" {
			parts = append(parts, StreamPart{Type: StreamFinish, FinishReason: decodeOpenAIFinishReason(*choice.FinishReason)})
		}
	}

	if chunk.Usage != nil && (chunk.Usage.PromptTokens != nil || chunk.Usage.CompletionTokens != nil) {
		parts = append(parts, StreamPart{
			Type: StreamResponseMetadata,
			Usage: decodeOpenAIUsage(openAIUsage{
				PromptTokens:            chunk.Usage.PromptTokens,
				CompletionTokens:        chunk.Usage.CompletionTokens,
				TotalTokens:             chunk.Usage.TotalTokens,
				PromptTokensDetails:     chunk.Usage.PromptTokensDetails,
				CompletionTokensDetails: chunk.Usage.CompletionTokensDetails,
			}),
		})
	}

	return parts, nil
}

func (d *openAIChatStreamDecoder) Close() ([]StreamPart, error) {
	return nil, nil
}

type openAIChatStreamEncoder struct {
	model       string
	started     bool
	finished    bool
	nextIndex   int
	toolIndexes map[string]int
	toolNames   map[string]string
	toolInputs  map[string]string
}

func (e *openAIChatStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	if e.toolNames == nil {
		e.toolNames = make(map[string]string)
		e.toolInputs = make(map[string]string)
	}

	switch part.Type {
	case StreamStart:
		e.started = true
		chunk := openAIChatStreamChunk{ID: part.ID, Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp()}
		if part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
			chunk.Usage = &openAIChatStreamChunkUsage{
				PromptTokens: part.Usage.InputTokens, CompletionTokens: part.Usage.OutputTokens,
				TotalTokens: calculateTotalTokens(part.Usage.InputTokens, part.Usage.OutputTokens),
			}
			if part.Usage.CachedInputTokens != nil {
				chunk.Usage.PromptTokensDetails = &openAIPromptTokensDetails{CachedTokens: part.Usage.CachedInputTokens}
			}
			if part.Usage.ReasoningTokens != nil {
				chunk.Usage.CompletionTokensDetails = &openAICompletionTokensDetails{ReasoningTokens: part.Usage.ReasoningTokens}
			}
		}
		role := "assistant"
		chunk.Choices = []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{Role: role}}}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamTextDelta:
		content := part.Delta
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{Content: &content}}}}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamReasoningDelta:
		reasoning := part.Delta
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{Reasoning: &reasoning}}}}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamToolInputStart:
		idx := e.ensureToolIndex(part.ToolCallID)
		e.toolNames[part.ToolCallID] = part.ToolName
		e.toolInputs[part.ToolCallID] = ""
		tc := openAIChatStreamToolCall{Index: idx, ID: part.ToolCallID, Type: "function", Function: openAIChatStreamToolCallFunction{Name: part.ToolName}}
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{ToolCalls: []openAIChatStreamToolCall{tc}}}}}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamToolInputDelta:
		idx := e.ensureToolIndex(part.ToolCallID)
		e.toolInputs[part.ToolCallID] += part.Delta
		tc := openAIChatStreamToolCall{Index: idx, Function: openAIChatStreamToolCallFunction{Arguments: part.Delta}}
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{ToolCalls: []openAIChatStreamToolCall{tc}}}}}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamToolInputEnd:
		return nil, nil
	case StreamToolCall:
		return e.encodeToolCall(part)
	case StreamFinish:
		e.finished = true
		return e.encodeFinish(part)
	case StreamResponseMetadata:
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{}}}}
		if part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
			chunk.Usage = &openAIChatStreamChunkUsage{
				PromptTokens:     part.Usage.InputTokens,
				CompletionTokens: part.Usage.OutputTokens,
				TotalTokens:      calculateTotalTokens(part.Usage.InputTokens, part.Usage.OutputTokens),
			}
			if part.Usage.CachedInputTokens != nil {
				chunk.Usage.PromptTokensDetails = &openAIPromptTokensDetails{CachedTokens: part.Usage.CachedInputTokens}
			}
			if part.Usage.ReasoningTokens != nil {
				chunk.Usage.CompletionTokensDetails = &openAICompletionTokensDetails{ReasoningTokens: part.Usage.ReasoningTokens}
			}
		}
		return singleOpenAIChatStreamEvent(chunk)
	case StreamError:
		return e.encodeStreamError(part)
	case StreamRaw:
		chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{Content: strPtr(fmt.Sprint(part.RawValue))}}}}
		return singleOpenAIChatStreamEvent(chunk)
	default:
		return nil, nil
	}
}

func (e *openAIChatStreamEncoder) Close() ([]RawStreamEvent, error) {
	if e.finished {
		return []RawStreamEvent{{Data: []byte("[DONE]")}}, nil
	}
	events, err := e.encodeFinish(StreamPart{Type: StreamFinish, FinishReason: FinishStop})
	if err != nil {
		return nil, err
	}
	return append(events, RawStreamEvent{Data: []byte("[DONE]")}), nil
}

func (e *openAIChatStreamEncoder) EncodeError(err error) []RawStreamEvent {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}
	part := StreamPart{Type: StreamError, Error: map[string]any{"message": message, "type": "protocol_bridge_error"}}
	events, _ := e.encodeStreamError(part)
	return events
}

func (e *openAIChatStreamEncoder) encodeToolCall(part StreamPart) ([]RawStreamEvent, error) {
	toolID := part.ToolCallID
	if toolID == "" {
		toolID = part.ID
	}
	idx := e.ensureToolIndex(toolID)
	name := part.ToolName
	if name == "" {
		name = e.toolNames[toolID]
	}
	input, err := encodeOpenAIToolInput(part.Input)
	if err != nil {
		return nil, err
	}
	var events []RawStreamEvent
	startChunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{ToolCalls: []openAIChatStreamToolCall{{Index: idx, ID: toolID, Type: "function", Function: openAIChatStreamToolCallFunction{Name: name}}}}}}}
	start, err := singleOpenAIChatStreamEvent(startChunk)
	if err != nil {
		return nil, err
	}
	events = append(events, start...)
	deltaChunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{ToolCalls: []openAIChatStreamToolCall{{Index: idx, Function: openAIChatStreamToolCallFunction{Arguments: input}}}}}}}
	delta, err := singleOpenAIChatStreamEvent(deltaChunk)
	if err != nil {
		return nil, err
	}
	events = append(events, delta...)
	delete(e.toolIndexes, toolID)
	return events, nil
}

func (e *openAIChatStreamEncoder) encodeFinish(part StreamPart) ([]RawStreamEvent, error) {
	reason := encodeOpenAIFinishReason(part.FinishReason)
	chunk := openAIChatStreamChunk{Object: "chat.completion.chunk", Model: e.model, Created: currentTimestamp(), Choices: []openAIChatStreamChoice{{Index: 0, Delta: &openAIChatStreamDelta{}, FinishReason: &reason}}}
	if part.Usage.InputTokens != nil || part.Usage.OutputTokens != nil {
		chunk.Usage = &openAIChatStreamChunkUsage{
			PromptTokens:     part.Usage.InputTokens,
			CompletionTokens: part.Usage.OutputTokens,
			TotalTokens:      calculateTotalTokens(part.Usage.InputTokens, part.Usage.OutputTokens),
		}
		if part.Usage.CachedInputTokens != nil {
			chunk.Usage.PromptTokensDetails = &openAIPromptTokensDetails{CachedTokens: part.Usage.CachedInputTokens}
		}
		if part.Usage.ReasoningTokens != nil {
			chunk.Usage.CompletionTokensDetails = &openAICompletionTokensDetails{ReasoningTokens: part.Usage.ReasoningTokens}
		}
	}
	return singleOpenAIChatStreamEvent(chunk)
}

func (e *openAIChatStreamEncoder) encodeStreamError(part StreamPart) ([]RawStreamEvent, error) {
	raw, err := json.Marshal(openAIErrorResponse{Error: openAIError{Message: fmt.Sprint(part.Error), Type: "protocol_bridge_error"}})
	if err != nil {
		return nil, err
	}
	return []RawStreamEvent{{Data: raw}}, nil
}

func (e *openAIChatStreamEncoder) ensureToolIndex(id string) int {
	if idx, ok := e.toolIndexes[id]; ok {
		return idx
	}
	idx := e.nextIndex
	e.nextIndex++
	e.toolIndexes[id] = idx
	return idx
}

func singleOpenAIChatStreamEvent(chunk openAIChatStreamChunk) ([]RawStreamEvent, error) {
	raw, err := json.Marshal(chunk)
	if err != nil {
		return nil, err
	}
	return []RawStreamEvent{{Data: raw}}, nil
}

func currentTimestamp() int64 {
	return 0
}

func strPtr(s string) *string {
	return &s
}

func (a OpenAIChatAdapter) EncodeError(err error) ([]byte, int) {
	message := "unknown error"
	if err != nil {
		message = err.Error()
	}

	raw, marshalErr := json.Marshal(openAIErrorResponse{
		Error: openAIError{
			Message: message,
			Type:    "protocol_bridge_error",
		},
	})
	if marshalErr != nil {
		return []byte(`{"error":{"message":"failed to encode error","type":"protocol_bridge_error"}}`), http.StatusInternalServerError
	}
	return raw, http.StatusBadRequest
}

func decodeOpenAIChatMessage(message openAIChatMessage) (Message, error) {
	role := Role(message.Role)
	decoded := Message{Role: role}

	if strings.TrimSpace(message.Reasoning) != "" {
		decoded.Parts = append(decoded.Parts, Part{Type: PartReasoning, Reasoning: &ReasoningPart{Text: message.Reasoning}})
	}
	if message.Content != nil {
		parts, err := decodeOpenAIChatContent(asRawMessage(message.Content))
		if err != nil {
			return Message{}, err
		}
		decoded.Parts = append(decoded.Parts, parts...)
	}
	if strings.TrimSpace(message.Refusal) != "" {
		decoded.Parts = append(decoded.Parts, Part{Type: PartRefusal, Refusal: &RefusalPart{Text: message.Refusal}})
	}

	for _, toolCall := range message.ToolCalls {
		input, err := decodeOpenAIToolInput(toolCall.Function.Arguments)
		if err != nil {
			return Message{}, err
		}
		decoded.Parts = append(decoded.Parts, Part{
			Type: PartToolCall,
			ToolCall: &ToolCallPart{
				ToolCallID: toolCall.ID,
				ToolName:   toolCall.Function.Name,
				Input:      input,
			},
		})
	}

	if role == RoleTool {
		parts, err := decodeOpenAIChatContent(asRawMessage(message.Content))
		if err != nil {
			return Message{}, err
		}
		decoded.Parts = []Part{
			{
				Type: PartToolResult,
				ToolResult: &ToolResultPart{
					ToolCallID: message.ToolCallID,
					Output: ToolResultOutput{
						Type: ToolResultText,
						Text: joinTextParts(parts),
					},
				},
			},
		}
	}

	return decoded, nil
}

func decodeReasoningEffort(effort string) *bool {
	if strings.TrimSpace(effort) == "" {
		return nil
	}
	enabled := true
	return &enabled
}

func encodeReasoningEffort(reasoning *bool) string {
	if reasoning == nil || !*reasoning {
		return ""
	}
	return "medium"
}

func encodeOpenAIStreamOptions(stream bool) any {
	if !stream {
		return nil
	}
	return map[string]any{"include_usage": true}
}

func encodeOpenAIChatMessages(message Message) ([]openAIChatMessage, error) {
	if message.Role == RoleTool {
		return encodeOpenAIToolMessages(message), nil
	}

	encoded := openAIChatMessage{
		Role:    string(message.Role),
		Content: encodeOpenAITextContent(message.Parts),
	}

	for _, part := range message.Parts {
		if part.Type == PartRefusal && part.Refusal != nil {
			encoded.Refusal = part.Refusal.Text
			continue
		}
		if part.Type != PartToolCall || part.ToolCall == nil {
			continue
		}
		arguments, err := encodeOpenAIToolInput(part.ToolCall.Input)
		if err != nil {
			return nil, err
		}
		encoded.ToolCalls = append(encoded.ToolCalls, openAIChatToolCall{
			ID:   part.ToolCall.ToolCallID,
			Type: "function",
			Function: openAIChatToolCallFunction{
				Name:      part.ToolCall.ToolName,
				Arguments: arguments,
			},
		})
	}

	if (len(encoded.ToolCalls) > 0 || encoded.Refusal != "") && encoded.Content == "" {
		encoded.Content = nil
	}

	return []openAIChatMessage{encoded}, nil
}

func encodeOpenAIToolMessages(message Message) []openAIChatMessage {
	encoded := make([]openAIChatMessage, 0)
	for _, part := range message.Parts {
		if part.Type != PartToolResult || part.ToolResult == nil {
			continue
		}
		encoded = append(encoded, openAIChatMessage{
			Role:       string(RoleTool),
			ToolCallID: part.ToolResult.ToolCallID,
			Content:    encodeOpenAIToolOutput(part.ToolResult.Output),
		})
	}
	return encoded
}

func encodeOpenAIAssistantMessage(content []Part) (openAIChatMessage, error) {
	message := Message{Role: RoleAssistant, Parts: content}
	encoded, err := encodeOpenAIChatMessages(message)
	if err != nil {
		return openAIChatMessage{}, err
	}
	if len(encoded) == 0 {
		return openAIChatMessage{Role: string(RoleAssistant)}, nil
	}
	return encoded[0], nil
}

func decodeOpenAIChatContent(raw json.RawMessage) ([]Part, error) {
	if len(raw) == 0 || string(raw) == "null" {
		return nil, nil
	}

	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		if text == "" {
			return nil, nil
		}
		return []Part{{Type: PartText, Text: &TextPart{Text: text}}}, nil
	}

	var parts []openAIChatContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return nil, fmt.Errorf("decode openai chat content: %w", err)
	}

	decoded := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type == "text" {
			decoded = append(decoded, Part{Type: PartText, Text: &TextPart{Text: part.Text}})
			continue
		}
		if part.Type == "image_url" && part.ImageURL != nil {
			file := decodeFileURL(part.ImageURL.URL, FileImage)
			file.Detail = part.ImageURL.Detail
			decoded = append(decoded, Part{Type: PartFile, File: file})
			continue
		}
		if part.Type == "file" && part.File != nil {
			file := &FilePart{Type: FileDocument, Data: part.File.FileData, FileID: part.File.FileID, Filename: part.File.Filename}
			decoded = append(decoded, Part{Type: PartFile, File: file})
		}
	}
	return decoded, nil
}

func encodeOpenAITextContent(parts []Part) any {
	encoded := make([]openAIChatContentPart, 0, len(parts))
	for _, part := range parts {
		if part.Type == PartText && part.Text != nil {
			encoded = append(encoded, openAIChatContentPart{Type: "text", Text: part.Text.Text})
			continue
		}
		if part.Type == PartFile && part.File != nil && part.File.Type == FileImage {
			url := encodeFileURL(part.File)
			if url == "" {
				continue
			}
			encoded = append(encoded, openAIChatContentPart{Type: "image_url", ImageURL: &openAIChatImageURL{URL: url, Detail: part.File.Detail}})
			continue
		}
		if part.Type == PartFile && part.File != nil && part.File.Type == FileDocument {
			file := openAIChatFilePart{FileData: part.File.Data, FileID: part.File.FileID, Filename: part.File.Filename}
			if file.FileData == "" && file.FileID == "" {
				continue
			}
			encoded = append(encoded, openAIChatContentPart{Type: "file", File: &file})
		}
	}
	if len(encoded) == 0 {
		return ""
	}
	if len(encoded) == 1 && encoded[0].Type == "text" {
		return encoded[0].Text
	}
	return encoded
}

func joinTextParts(parts []Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.Type == PartText && part.Text != nil {
			builder.WriteString(part.Text.Text)
		}
	}
	return builder.String()
}

func joinReasoningParts(parts []Part) string {
	var builder strings.Builder
	for _, part := range parts {
		if part.Type == PartReasoning && part.Reasoning != nil {
			builder.WriteString(part.Reasoning.Text)
		}
	}
	return builder.String()
}

func decodeOpenAIToolInput(arguments string) (any, error) {
	if strings.TrimSpace(arguments) == "" {
		return nil, nil
	}

	var input any
	if err := json.Unmarshal([]byte(arguments), &input); err != nil {
		return arguments, nil
	}
	return input, nil
}

func encodeOpenAIToolInput(input any) (string, error) {
	if input == nil {
		return "{}", nil
	}
	raw, err := json.Marshal(input)
	if err != nil {
		return "", fmt.Errorf("encode openai tool input: %w", err)
	}
	return string(raw), nil
}

func encodeOpenAIToolOutput(output ToolResultOutput) any {
	switch output.Type {
	case ToolResultJSON, ToolResultErrorJSON:
		raw, err := json.Marshal(output.JSON)
		if err != nil {
			return "null"
		}
		return string(raw)
	case ToolResultContent:
		return joinTextParts(output.Content)
	default:
		return output.Text
	}
}

func decodeOpenAITools(tools []openAIChatTool) []Tool {
	decoded := make([]Tool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != "function" {
			continue
		}
		decoded = append(decoded, Tool{
			Type:        ToolFunction,
			Name:        tool.Function.Name,
			Description: tool.Function.Description,
			InputSchema: tool.Function.Parameters,
			Strict:      tool.Function.Strict,
		})
	}
	return decoded
}

func encodeOpenAITools(tools []Tool) []openAIChatTool {
	encoded := make([]openAIChatTool, 0, len(tools))
	for _, tool := range tools {
		if tool.Type != ToolFunction {
			continue
		}
		encoded = append(encoded, openAIChatTool{
			Type: "function",
			Function: openAIChatFunctionTool{
				Name:        tool.Name,
				Description: tool.Description,
				Parameters:  tool.InputSchema,
				Strict:      tool.Strict,
			},
		})
	}
	return encoded
}

func decodeOpenAIToolChoice(raw json.RawMessage) *ToolChoice {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		switch value {
		case "auto":
			return &ToolChoice{Type: ToolChoiceAuto}
		case "none":
			return &ToolChoice{Type: ToolChoiceNone}
		case "required":
			return &ToolChoice{Type: ToolChoiceRequired}
		}
	}

	var object struct {
		Type     string `json:"type"`
		Function struct {
			Name string `json:"name"`
		} `json:"function"`
	}
	if err := json.Unmarshal(raw, &object); err == nil && object.Type == "function" {
		return &ToolChoice{Type: ToolChoiceTool, ToolName: object.Function.Name}
	}

	return nil
}

func encodeOpenAIToolChoice(choice *ToolChoice) any {
	if choice == nil {
		return nil
	}
	switch choice.Type {
	case ToolChoiceAuto:
		return "auto"
	case ToolChoiceNone:
		return "none"
	case ToolChoiceRequired:
		return "required"
	case ToolChoiceTool:
		return map[string]any{
			"type": "function",
			"function": map[string]any{
				"name": choice.ToolName,
			},
		}
	default:
		return nil
	}
}

func decodeOpenAIResponseFormat(raw json.RawMessage) *ResponseFormat {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var object struct {
		Type       string `json:"type"`
		JSONSchema struct {
			Name        string         `json:"name"`
			Description string         `json:"description"`
			Schema      map[string]any `json:"schema"`
			Strict      *bool          `json:"strict"`
		} `json:"json_schema"`
	}
	if err := json.Unmarshal(raw, &object); err != nil {
		return nil
	}

	switch object.Type {
	case "json_object":
		return &ResponseFormat{Type: ResponseFormatJSON}
	case "json_schema":
		return &ResponseFormat{
			Type:        ResponseFormatJSON,
			Schema:      object.JSONSchema.Schema,
			Name:        object.JSONSchema.Name,
			Description: object.JSONSchema.Description,
			Strict:      object.JSONSchema.Strict,
		}
	case "text":
		return &ResponseFormat{Type: ResponseFormatText}
	default:
		return nil
	}
}

func encodeOpenAIResponseFormat(format *ResponseFormat) any {
	if format == nil || format.Type == ResponseFormatText {
		return nil
	}
	if format.Type != ResponseFormatJSON {
		return nil
	}
	if format.Schema == nil {
		return map[string]any{"type": "json_object"}
	}
	return map[string]any{
		"type": "json_schema",
		"json_schema": map[string]any{
			"name":        format.Name,
			"description": format.Description,
			"schema":      format.Schema,
			"strict":      format.Strict,
		},
	}
}

func decodeOpenAIStop(raw json.RawMessage) []string {
	if len(raw) == 0 || string(raw) == "null" {
		return nil
	}

	var value string
	if err := json.Unmarshal(raw, &value); err == nil {
		return []string{value}
	}

	var values []string
	if err := json.Unmarshal(raw, &values); err == nil {
		return values
	}
	return nil
}

func encodeOpenAIStop(stop []string) any {
	if len(stop) == 0 {
		return nil
	}
	if len(stop) == 1 {
		return stop[0]
	}
	return stop
}

func decodeOpenAIFinishReason(reason string) FinishReason {
	switch reason {
	case "stop":
		return FinishStop
	case "length":
		return FinishLength
	case "content_filter":
		return FinishContentFilter
	case "tool_calls", "function_call":
		return FinishToolCalls
	case "":
		return FinishUnknown
	default:
		return FinishOther
	}
}

func encodeOpenAIFinishReason(reason FinishReason) string {
	switch reason {
	case FinishStop:
		return "stop"
	case FinishLength:
		return "length"
	case FinishContentFilter:
		return "content_filter"
	case FinishToolCalls:
		return "tool_calls"
	default:
		return "stop"
	}
}

func decodeOpenAIUsage(usage openAIUsage) Usage {
	decoded := Usage{
		InputTokens:  usage.PromptTokens,
		OutputTokens: usage.CompletionTokens,
	}
	if usage.PromptTokensDetails != nil {
		decoded.CachedInputTokens = usage.PromptTokensDetails.CachedTokens
	}
	if usage.CompletionTokensDetails != nil {
		decoded.ReasoningTokens = usage.CompletionTokensDetails.ReasoningTokens
	}
	return decoded
}

func encodeOpenAIUsage(usage Usage, billingUsage BillingUsage) openAIUsage {
	if hasBillingUsage(billingUsage) {
		inputTokens := billingUsage.InputTokens + billingUsage.CachedInputTokens
		outputTokens := billingUsage.OutputTokens
		cachedInputTokens := billingUsage.CachedInputTokens
		encoded := openAIUsage{
			PromptTokens:     &inputTokens,
			CompletionTokens: &outputTokens,
			TotalTokens:      calculateTotalTokens(&inputTokens, &outputTokens),
		}
		if cachedInputTokens > 0 {
			encoded.PromptTokensDetails = &openAIPromptTokensDetails{CachedTokens: &cachedInputTokens}
		}
		if usage.ReasoningTokens != nil {
			encoded.CompletionTokensDetails = &openAICompletionTokensDetails{ReasoningTokens: usage.ReasoningTokens}
		}
		return encoded
	}
	encoded := openAIUsage{
		PromptTokens:     usage.InputTokens,
		CompletionTokens: usage.OutputTokens,
		TotalTokens:      calculateTotalTokens(usage.InputTokens, usage.OutputTokens),
	}
	if usage.CachedInputTokens != nil {
		encoded.PromptTokensDetails = &openAIPromptTokensDetails{CachedTokens: usage.CachedInputTokens}
	}
	if usage.ReasoningTokens != nil {
		encoded.CompletionTokensDetails = &openAICompletionTokensDetails{ReasoningTokens: usage.ReasoningTokens}
	}
	return encoded
}

type openAIChatRequest struct {
	Model               string              `json:"model"`
	Messages            []openAIChatMessage `json:"messages"`
	MaxTokens           *int                `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int                `json:"max_completion_tokens,omitempty"`
	Temperature         *float64            `json:"temperature,omitempty"`
	Stop                any                 `json:"stop,omitempty"`
	TopP                *float64            `json:"top_p,omitempty"`
	PresencePenalty     *float64            `json:"presence_penalty,omitempty"`
	FrequencyPenalty    *float64            `json:"frequency_penalty,omitempty"`
	Seed                *int64              `json:"seed,omitempty"`
	N                   *int                `json:"n,omitempty"`
	ResponseFormat      any                 `json:"response_format,omitempty"`
	ReasoningEffort     string              `json:"reasoning_effort,omitempty"`
	StreamOptions       any                 `json:"stream_options,omitempty"`
	Tools               []openAIChatTool    `json:"tools,omitempty"`
	ToolChoice          any                 `json:"tool_choice,omitempty"`
	ParallelToolCalls   *bool               `json:"parallel_tool_calls,omitempty"`
	Stream              bool                `json:"stream,omitempty"`
}

func (r *openAIChatRequest) UnmarshalJSON(raw []byte) error {
	type alias openAIChatRequest
	var decoded struct {
		alias
		Stop           json.RawMessage `json:"stop"`
		ResponseFormat json.RawMessage `json:"response_format"`
		ToolChoice     json.RawMessage `json:"tool_choice"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*r = openAIChatRequest(decoded.alias)
	r.Stop = decoded.Stop
	r.ResponseFormat = decoded.ResponseFormat
	r.ToolChoice = decoded.ToolChoice
	return nil
}

type openAIChatMessage struct {
	Role       string               `json:"role"`
	Content    any                  `json:"content,omitempty"`
	Reasoning  string               `json:"reasoning_content,omitempty"`
	Refusal    string               `json:"refusal,omitempty"`
	ToolCalls  []openAIChatToolCall `json:"tool_calls,omitempty"`
	ToolCallID string               `json:"tool_call_id,omitempty"`
}

func (m *openAIChatMessage) UnmarshalJSON(raw []byte) error {
	var decoded struct {
		Role       string               `json:"role"`
		Content    json.RawMessage      `json:"content"`
		Reasoning  string               `json:"reasoning_content"`
		Refusal    string               `json:"refusal"`
		ToolCalls  []openAIChatToolCall `json:"tool_calls"`
		ToolCallID string               `json:"tool_call_id"`
	}
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	m.Role = decoded.Role
	m.Content = decoded.Content
	m.Reasoning = decoded.Reasoning
	m.Refusal = decoded.Refusal
	m.ToolCalls = decoded.ToolCalls
	m.ToolCallID = decoded.ToolCallID
	return nil
}

type openAIChatContentPart struct {
	Type     string              `json:"type"`
	Text     string              `json:"text,omitempty"`
	ImageURL *openAIChatImageURL `json:"image_url,omitempty"`
	File     *openAIChatFilePart `json:"file,omitempty"`
}

type openAIChatImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type openAIChatFilePart struct {
	FileData string `json:"file_data,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Filename string `json:"filename,omitempty"`
}

type openAIChatToolCall struct {
	ID       string                     `json:"id"`
	Type     string                     `json:"type"`
	Function openAIChatToolCallFunction `json:"function"`
}

type openAIChatToolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type openAIChatTool struct {
	Type     string                 `json:"type"`
	Function openAIChatFunctionTool `json:"function"`
}

type openAIChatFunctionTool struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Parameters  map[string]any `json:"parameters,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type openAIChatResponse struct {
	ID      string             `json:"id,omitempty"`
	Object  string             `json:"object,omitempty"`
	Created int64              `json:"created,omitempty"`
	Model   string             `json:"model,omitempty"`
	Choices []openAIChatChoice `json:"choices"`
	Usage   openAIUsage        `json:"usage,omitempty"`
}

type openAIChatChoice struct {
	Index        int               `json:"index"`
	Message      openAIChatMessage `json:"message"`
	FinishReason string            `json:"finish_reason"`
}

type openAIUsage struct {
	PromptTokens            *int                           `json:"prompt_tokens,omitempty"`
	CompletionTokens        *int                           `json:"completion_tokens,omitempty"`
	TotalTokens             *int                           `json:"total_tokens,omitempty"`
	PromptTokensDetails     *openAIPromptTokensDetails     `json:"prompt_tokens_details,omitempty"`
	CompletionTokensDetails *openAICompletionTokensDetails `json:"completion_tokens_details,omitempty"`
}

type openAIPromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens,omitempty"`
}

type openAICompletionTokensDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens,omitempty"`
}

type openAIErrorResponse struct {
	Error openAIError `json:"error"`
}

type openAIError struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func asRawMessage(value any) json.RawMessage {
	switch typed := value.(type) {
	case nil:
		return nil
	case json.RawMessage:
		return typed
	case []byte:
		return typed
	default:
		raw, err := json.Marshal(typed)
		if err != nil {
			return nil
		}
		return raw
	}
}
