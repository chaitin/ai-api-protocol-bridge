package protocolbridge

type RawStreamEvent struct {
	Event string
	Data  []byte
	ID    string
	Retry *int
}

type StreamPart struct {
	Type StreamPartType `json:"type"`

	ID string `json:"id,omitempty"`

	Delta string `json:"delta,omitempty"`

	ToolName   string `json:"tool_name,omitempty"`
	ToolCallID string `json:"tool_call_id,omitempty"`

	Input any `json:"input,omitempty"`

	Output *ToolResultOutput `json:"output,omitempty"`

	File *FilePart `json:"file,omitempty"`

	FinishReason FinishReason `json:"finish_reason,omitempty"`

	Usage Usage `json:"usage,omitempty"`

	Warnings []Warning `json:"warnings,omitempty"`

	Error any `json:"error,omitempty"`

	ProviderMetadata map[string]any `json:"provider_metadata,omitempty"`

	RawValue any `json:"raw_value,omitempty"`
}

type StreamPartType string

const (
	StreamStart StreamPartType = "stream-start"

	StreamTextStart StreamPartType = "text-start"
	StreamTextDelta StreamPartType = "text-delta"
	StreamTextEnd   StreamPartType = "text-end"

	StreamReasoningStart StreamPartType = "reasoning-start"
	StreamReasoningDelta StreamPartType = "reasoning-delta"
	StreamReasoningEnd   StreamPartType = "reasoning-end"

	StreamToolInputStart StreamPartType = "tool-input-start"
	StreamToolInputDelta StreamPartType = "tool-input-delta"
	StreamToolInputEnd   StreamPartType = "tool-input-end"

	StreamToolCall   StreamPartType = "tool-call"
	StreamToolResult StreamPartType = "tool-result"

	StreamFile StreamPartType = "file"

	StreamResponseMetadata StreamPartType = "response-metadata"

	StreamFinish StreamPartType = "finish"

	StreamRaw StreamPartType = "raw"

	StreamError StreamPartType = "error"
)
