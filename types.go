package protocolbridge

import "strings"

const (
	defaultMaxOutputTokens           = 4096
	defaultThinkingBudgetTokens      = 1024
	minAnthropicThinkingBudgetTokens = 1024
)

type Protocol string

const (
	ProtocolOpenAIChat        Protocol = "openai_chat"
	ProtocolOpenAIResponses   Protocol = "openai_responses"
	ProtocolAnthropicMessages Protocol = "anthropic_messages"
)

type LLMRequest struct {
	Protocol Protocol `json:"protocol,omitempty"`

	Model string `json:"model"`

	Prompt []Message `json:"prompt"`

	MaxOutputTokens *int     `json:"max_output_tokens,omitempty"`
	Temperature     *float64 `json:"temperature,omitempty"`
	StopSequences   []string `json:"stop_sequences,omitempty"`
	TopP            *float64 `json:"top_p,omitempty"`
	TopK            *int     `json:"top_k,omitempty"`

	PresencePenalty  *float64 `json:"presence_penalty,omitempty"`
	FrequencyPenalty *float64 `json:"frequency_penalty,omitempty"`
	Seed             *int64   `json:"seed,omitempty"`
	CandidateCount   *int     `json:"candidate_count,omitempty"`

	ResponseFormat *ResponseFormat `json:"response_format,omitempty"`

	State   *RequestState `json:"state,omitempty"`
	Include []string      `json:"include,omitempty"`
	Cache   *bool         `json:"cache,omitempty"`

	Reasoning             *bool  `json:"reasoning,omitempty"`
	ReasoningBudgetTokens *int   `json:"reasoning_budget_tokens,omitempty"`
	ReasoningEffort       string `json:"reasoning_effort,omitempty"`
	ReasoningSummary      string `json:"reasoning_summary,omitempty"`

	Tools      []Tool      `json:"tools,omitempty"`
	ToolChoice *ToolChoice `json:"tool_choice,omitempty"`

	ParallelToolCalls *bool `json:"parallel_tool_calls,omitempty"`

	Stream bool `json:"stream,omitempty"`

	Metadata map[string]string `json:"metadata,omitempty"`

	ProviderOptions map[string]any `json:"provider_options,omitempty"`
}

type Message struct {
	Role Role `json:"role"`

	Parts []Part `json:"parts,omitempty"`

	ProviderOptions map[string]any `json:"provider_options,omitempty"`
}

type Role string

const (
	RoleSystem    Role = "system"
	RoleDeveloper Role = "developer"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

type Part struct {
	Type PartType `json:"type"`

	Text       *TextPart       `json:"text,omitempty"`
	File       *FilePart       `json:"file,omitempty"`
	Reasoning  *ReasoningPart  `json:"reasoning,omitempty"`
	Refusal    *RefusalPart    `json:"refusal,omitempty"`
	ToolCall   *ToolCallPart   `json:"tool_call,omitempty"`
	ToolResult *ToolResultPart `json:"tool_result,omitempty"`

	ProviderOptions map[string]any `json:"provider_options,omitempty"`
}

type PartType string

const (
	PartText       PartType = "text"
	PartFile       PartType = "file"
	PartReasoning  PartType = "reasoning"
	PartRefusal    PartType = "refusal"
	PartToolCall   PartType = "tool-call"
	PartToolResult PartType = "tool-result"
)

type TextPart struct {
	Text string `json:"text"`
}

type FilePart struct {
	Type      FilePartType `json:"type,omitempty"`
	Filename  string       `json:"filename,omitempty"`
	MediaType string       `json:"media_type"`

	Data   string `json:"data,omitempty"`
	URL    string `json:"url,omitempty"`
	FileID string `json:"file_id,omitempty"`
	Detail string `json:"detail,omitempty"`
}

type FilePartType string

const (
	FileImage    FilePartType = "image"
	FileDocument FilePartType = "document"
)

type ReasoningPart struct {
	Text      string `json:"text"`
	Redacted  string `json:"redacted,omitempty"`
	Signature string `json:"signature,omitempty"`
}

type RefusalPart struct {
	Text string `json:"text"`
}

type ToolCallPart struct {
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name"`

	Input any `json:"input,omitempty"`

	ProviderExecuted bool `json:"provider_executed,omitempty"`
}

type ToolResultPart struct {
	ToolCallID string `json:"tool_call_id"`
	ToolName   string `json:"tool_name"`

	Output ToolResultOutput `json:"output"`
}

type ToolResultOutput struct {
	Type ToolResultOutputType `json:"type"`

	Text    string `json:"text,omitempty"`
	JSON    any    `json:"json,omitempty"`
	Content []Part `json:"content,omitempty"`
}

type ToolResultOutputType string

const (
	ToolResultText      ToolResultOutputType = "text"
	ToolResultJSON      ToolResultOutputType = "json"
	ToolResultErrorText ToolResultOutputType = "error-text"
	ToolResultErrorJSON ToolResultOutputType = "error-json"
	ToolResultContent   ToolResultOutputType = "content"
)

type Tool struct {
	Type        ToolType       `json:"type"`
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
	Config      map[string]any `json:"config,omitempty"`

	ProviderOptions map[string]any `json:"provider_options,omitempty"`
}

type ToolType string

const (
	ToolFunction        ToolType = "function"
	ToolProviderDefined ToolType = "provider-defined"
)

type ToolChoice struct {
	Type ToolChoiceType `json:"type"`

	ToolName string `json:"tool_name,omitempty"`
}

type ToolChoiceType string

const (
	ToolChoiceAuto     ToolChoiceType = "auto"
	ToolChoiceNone     ToolChoiceType = "none"
	ToolChoiceRequired ToolChoiceType = "required"
	ToolChoiceTool     ToolChoiceType = "tool"
)

type ResponseFormat struct {
	Type ResponseFormatType `json:"type"`

	Schema      map[string]any `json:"schema,omitempty"`
	Name        string         `json:"name,omitempty"`
	Description string         `json:"description,omitempty"`
	Strict      *bool          `json:"strict,omitempty"`
}

type RequestState struct {
	PreviousResponseID string `json:"previous_response_id,omitempty"`
	ConversationID     string `json:"conversation_id,omitempty"`
}

type ResponseFormatType string

const (
	ResponseFormatText ResponseFormatType = "text"
	ResponseFormatJSON ResponseFormatType = "json"
)

type LLMResponse struct {
	Protocol Protocol `json:"protocol,omitempty"`

	ID    string `json:"id,omitempty"`
	Model string `json:"model,omitempty"`

	Role    Role        `json:"role,omitempty"`
	Content []Part      `json:"content"`
	Choices []LLMChoice `json:"choices,omitempty"`

	FinishReason FinishReason `json:"finish_reason"`

	Usage Usage `json:"usage"`

	ProviderMetadata map[string]any `json:"provider_metadata,omitempty"`

	Warnings []Warning `json:"warnings,omitempty"`
}

type LLMChoice struct {
	Index int `json:"index"`

	Role    Role   `json:"role,omitempty"`
	Content []Part `json:"content"`

	FinishReason FinishReason `json:"finish_reason"`
}

type BillingUsage struct {
	InputTokens       int `json:"input_tokens,omitempty"`
	CachedInputTokens int `json:"cached_input_tokens,omitempty"`
	OutputTokens      int `json:"output_tokens,omitempty"`
}

func (r LLMResponse) BillingUsage() BillingUsage {
	return billingUsageForProtocol(r.Protocol, r.Usage)
}

func firstResponseContent(resp *LLMResponse) ([]Part, FinishReason) {
	if resp == nil || len(resp.Choices) == 0 {
		if resp == nil {
			return nil, FinishUnknown
		}
		return resp.Content, resp.FinishReason
	}
	choice := resp.Choices[0]
	return choice.Content, choice.FinishReason
}

type FinishReason string

const (
	FinishStop          FinishReason = "stop"
	FinishLength        FinishReason = "length"
	FinishContentFilter FinishReason = "content-filter"
	FinishToolCalls     FinishReason = "tool-calls"
	FinishError         FinishReason = "error"
	FinishOther         FinishReason = "other"
	FinishUnknown       FinishReason = "unknown"
)

type Usage struct {
	InputTokens              *int `json:"input_tokens,omitempty"`
	OutputTokens             *int `json:"output_tokens,omitempty"`
	ReasoningTokens          *int `json:"reasoning_tokens,omitempty"`
	CachedInputTokens        *int `json:"cached_input_tokens,omitempty"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens,omitempty"`
}

func mergeUsage(base *Usage, update Usage) {
	if base == nil {
		return
	}
	if update.InputTokens != nil {
		base.InputTokens = update.InputTokens
	}
	if update.OutputTokens != nil {
		base.OutputTokens = update.OutputTokens
	}
	if update.ReasoningTokens != nil {
		base.ReasoningTokens = update.ReasoningTokens
	}
	if update.CachedInputTokens != nil {
		base.CachedInputTokens = update.CachedInputTokens
	}
	if update.CacheCreationInputTokens != nil {
		base.CacheCreationInputTokens = update.CacheCreationInputTokens
	}
	if update.CacheReadInputTokens != nil {
		base.CacheReadInputTokens = update.CacheReadInputTokens
	}
}

func hasUsage(usage Usage) bool {
	return usage.InputTokens != nil || usage.OutputTokens != nil || usage.ReasoningTokens != nil || usage.CachedInputTokens != nil || usage.CacheCreationInputTokens != nil || usage.CacheReadInputTokens != nil
}

func calculateTotalTokens(inputTokens, outputTokens *int) *int {
	if inputTokens == nil || outputTokens == nil {
		return nil
	}
	totalTokens := *inputTokens + *outputTokens
	return &totalTokens
}

func maxOutputTokensOrDefault(value *int) *int {
	if value != nil && *value >= 0 {
		return value
	}
	defaultValue := defaultMaxOutputTokens
	return &defaultValue
}

func intValue(value *int) int {
	if value == nil || *value < 0 {
		return 0
	}
	return *value
}

func clampNonNegative(value int) int {
	if value < 0 {
		return 0
	}
	return value
}

func encodeFileURL(file *FilePart) string {
	if file == nil {
		return ""
	}
	if file.URL != "" {
		return file.URL
	}
	if file.Data == "" {
		return ""
	}
	mediaType := file.MediaType
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return "data:" + mediaType + ";base64," + file.Data
}

func decodeFileURL(url string, fileType FilePartType) *FilePart {
	file := &FilePart{Type: fileType}
	if mediaType, data, ok := splitDataURL(url); ok {
		file.MediaType = mediaType
		file.Data = data
		return file
	}
	file.URL = url
	return file
}

func splitDataURL(url string) (string, string, bool) {
	if !strings.HasPrefix(url, "data:") {
		return "", "", false
	}
	payload := strings.TrimPrefix(url, "data:")
	parts := strings.SplitN(payload, ",", 2)
	if len(parts) != 2 {
		return "", "", false
	}
	metadata := parts[0]
	if !strings.HasSuffix(metadata, ";base64") {
		return "", "", false
	}
	mediaType := strings.TrimSuffix(metadata, ";base64")
	if mediaType == "" {
		mediaType = "application/octet-stream"
	}
	return mediaType, parts[1], true
}

func hasBillingUsage(usage BillingUsage) bool {
	return usage.InputTokens != 0 || usage.CachedInputTokens != 0 || usage.OutputTokens != 0
}

func billingUsageForProtocol(protocol Protocol, usage Usage) BillingUsage {
	switch protocol {
	case ProtocolOpenAIChat, ProtocolOpenAIResponses:
		cachedInputTokens := intValue(usage.CachedInputTokens)
		return BillingUsage{InputTokens: clampNonNegative(intValue(usage.InputTokens) - cachedInputTokens), CachedInputTokens: cachedInputTokens, OutputTokens: intValue(usage.OutputTokens)}
	case ProtocolAnthropicMessages:
		cacheReadTokens := usage.CacheReadInputTokens
		if cacheReadTokens == nil {
			cacheReadTokens = usage.CachedInputTokens
		}
		return BillingUsage{InputTokens: intValue(usage.InputTokens) + intValue(usage.CacheCreationInputTokens), CachedInputTokens: intValue(cacheReadTokens), OutputTokens: intValue(usage.OutputTokens)}
	default:
		return BillingUsage{InputTokens: intValue(usage.InputTokens), CachedInputTokens: intValue(usage.CachedInputTokens), OutputTokens: intValue(usage.OutputTokens)}
	}
}

type Warning struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}
