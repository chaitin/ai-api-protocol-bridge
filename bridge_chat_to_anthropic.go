package protocolbridge

import (
	"encoding/json"
	"fmt"
)

func (b openAIChatToAnthropicBridge) InboundProtocol() Protocol {
	return ProtocolOpenAIChat
}

func (b openAIChatToAnthropicBridge) UpstreamProtocol() Protocol {
	return ProtocolAnthropicMessages
}

func (b openAIChatToAnthropicBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode openai chat to anthropic request: nil request")
	}

	model := req.Model
	if opts.Model != "" {
		model = opts.Model
	}

	request := anthropicRequest{
		Model:         model,
		Messages:      make([]anthropicMessage, 0, len(req.Prompt)),
		MaxTokens:     maxOutputTokensOrDefault(req.MaxOutputTokens),
		Temperature:   req.Temperature,
		StopSequences: append([]string(nil), req.StopSequences...),
		TopP:          req.TopP,
		Tools:         encodeAnthropicTools(req.Tools),
		Stream:        req.Stream,
	}
	request.OutputConfig = encodeAnthropicOutputConfig(req.ResponseFormat)
	request.Thinking = encodeAnthropicThinkingForOpenAIInbound(req, request.MaxTokens)
	request.ToolChoice = encodeAnthropicToolChoice(sanitizeAnthropicToolChoice(req.ToolChoice, request.Thinking), req.ParallelToolCalls)

	previousWasTool := false
	for _, message := range req.Prompt {
		if message.Role == RoleSystem || message.Role == RoleDeveloper {
			request.System = appendSystemText(request.System, joinTextParts(message.Parts))
			previousWasTool = false
			continue
		}
		request.Messages, previousWasTool = appendOpenAIInboundAnthropicMessage(request.Messages, message, previousWasTool)
	}

	applyAnthropicCache(&request, req.Cache)
	return json.Marshal(request)
}

func (b openAIChatToAnthropicBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	return b.adapter.DecodeResponse(raw)
}

func (b openAIChatToAnthropicBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return (&AnthropicMessagesAdapter{}).NewStreamDecoder(opts)
}

func (b openAIChatToAnthropicBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return &anthropicToOpenAIChatStreamEncoder{base: openAIChatStreamEncoder{model: opts.Model, toolIndexes: make(map[string]int)}}, nil
}

type anthropicToOpenAIChatStreamEncoder struct {
	base openAIChatStreamEncoder
}

func (e *anthropicToOpenAIChatStreamEncoder) Encode(part StreamPart) ([]RawStreamEvent, error) {
	if part.Type == StreamStart || part.Type == StreamFinish || part.Type == StreamResponseMetadata {
		part.Usage = anthropicUsageToResponsesUsage(part.Usage)
	}
	if part.Type == StreamReasoningDelta {
		if signature, ok := part.ProviderMetadata["signature"].(string); ok && signature != "" && part.Delta == "" {
			return nil, nil
		}
	}
	return e.base.Encode(part)
}

func (e *anthropicToOpenAIChatStreamEncoder) Close() ([]RawStreamEvent, error) {
	return e.base.Close()
}

func (e *anthropicToOpenAIChatStreamEncoder) EncodeError(err error) []RawStreamEvent {
	return e.base.EncodeError(err)
}
