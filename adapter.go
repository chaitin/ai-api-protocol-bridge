package protocolbridge

type EncodeRequestOptions struct {
	Model string

	// CacheControl decides whether the Anthropic encoder may add a
	// cache_control breakpoint of its own when the request does not express a
	// caching preference. Anthropic only caches a prefix when a breakpoint asks
	// it to, so Auto — the zero value — adds one; Disabled never does.
	//
	// The request's own preference always wins: a non-nil LLMRequest.Cache is
	// honoured whatever this says.
	CacheControl CacheControlPolicy

	// LossPolicy decides whether a loss the conversion had to incur fails the
	// call. The zero value, LossPolicyAllow, records the losses on
	// LLMRequest.Warnings and completes the request.
	LossPolicy LossPolicy
}

// CacheControlPolicy controls the cache_control breakpoint an Anthropic
// request is encoded with.
type CacheControlPolicy string

const (
	// CacheControlAuto adds one breakpoint when the request expresses no
	// preference of its own. This is the default, and matches Anthropic's
	// pricing: without a breakpoint nothing is cached, and an agent's long
	// system prompt is paid for in full on every turn.
	CacheControlAuto CacheControlPolicy = ""

	// CacheControlDisabled never adds a breakpoint.
	CacheControlDisabled CacheControlPolicy = "disabled"

	// CacheControlEnabled adds a breakpoint even if the request would not.
	CacheControlEnabled CacheControlPolicy = "enabled"
)

type EncodeResponseOptions struct {
	Model string

	// Created is the Unix timestamp in seconds to stamp on the response. Zero
	// means "use the current time"; hosts that need deterministic output for
	// snapshots or caching should set it.
	Created int64

	// LossPolicy decides whether a loss the conversion had to incur fails the
	// call. The zero value, LossPolicyAllow, records the losses on
	// LLMResponse.Warnings and completes the response.
	LossPolicy LossPolicy
}

type StreamDecodeOptions struct{}

type StreamEncodeOptions struct {
	Model string

	// Created is the Unix timestamp in seconds to stamp on every event of the
	// stream. Zero means "stamp the first event and reuse that value", which is
	// what a real provider does: one timestamp for the whole completion.
	Created int64
}

type Adapter interface {
	Protocol() Protocol

	DecodeRequest(raw []byte) (*LLMRequest, error)
	EncodeRequest(req *LLMRequest, opts EncodeRequestOptions) ([]byte, error)

	DecodeResponse(raw []byte) (*LLMResponse, error)
	EncodeResponse(resp *LLMResponse, opts EncodeResponseOptions) ([]byte, error)

	NewStreamDecoder(opts StreamDecodeOptions) (StreamDecoder, error)
	NewStreamEncoder(opts StreamEncodeOptions) (StreamEncoder, error)

	EncodeError(err error) ([]byte, int)
}

type StreamDecoder interface {
	Decode(event RawStreamEvent) ([]StreamPart, error)
	Close() ([]StreamPart, error)
}

type StreamEncoder interface {
	Encode(part StreamPart) ([]RawStreamEvent, error)
	Close() ([]RawStreamEvent, error)
	EncodeError(err error) []RawStreamEvent
}
