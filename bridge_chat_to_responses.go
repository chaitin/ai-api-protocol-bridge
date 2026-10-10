package protocolbridge

import "fmt"

// openAIChatToOpenAIResponsesBridge serves a Chat Completions client against an
// OpenAI Responses upstream.
//
// It is the last missing cell, and the quietest: the two protocols agree on
// enough that the upstream half is the responses adapter and the client half is
// the chat adapter, with nothing in between.
//
// The stream encoder is where that would normally stop being true. Responses
// reports usage on response.completed, the event that also ends the stream,
// while chat reports finish_reason and usage in two separate chunks. It would
// be easy to conclude that a finish arriving with usage has to be split by
// hand. It does not: the chat encoder accumulates usage from every part it
// sees, including the finish, and Close emits the usage-only chunk after the
// finish chunk before the [DONE] sentinel — which is exactly the shape the
// protocol asks for. So unlike the two Anthropic bridges, this one needs no
// wrapper, and adding one would only duplicate the encoder.
//
// Usage needs no conversion either. Responses and chat both count prompt tokens
// the OpenAI way, the total with the cached portion broken out separately, and
// both decoders map onto the same neutral fields.
type openAIChatToOpenAIResponsesBridge struct {
	adapter OpenAIResponsesAdapter
}

func (b openAIChatToOpenAIResponsesBridge) InboundProtocol() Protocol {
	return ProtocolOpenAIChat
}

func (b openAIChatToOpenAIResponsesBridge) UpstreamProtocol() Protocol {
	return ProtocolOpenAIResponses
}

func (b openAIChatToOpenAIResponsesBridge) EncodeUpstreamRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error) {
	if req == nil {
		return nil, fmt.Errorf("encode openai chat to openai responses request: nil request")
	}
	return b.adapter.EncodeRequest(req, opts)
}

func (b openAIChatToOpenAIResponsesBridge) DecodeUpstreamResponse(raw []byte) (*LLMResponse, error) {
	resp, err := b.adapter.DecodeResponse(raw)
	if err != nil {
		return nil, err
	}
	resp.Protocol = ProtocolOpenAIResponses
	return resp, nil
}

func (b openAIChatToOpenAIResponsesBridge) NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error) {
	return b.adapter.NewStreamDecoder(opts)
}

func (b openAIChatToOpenAIResponsesBridge) NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error) {
	return NewOpenAIChatAdapter().NewStreamEncoder(opts)
}
