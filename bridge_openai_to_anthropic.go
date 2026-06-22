package protocolbridge

import "strings"

func encodeAnthropicThinkingForOpenAIInbound(req *LLMRequest, maxTokens *int) any {
	if req == nil {
		return nil
	}
	thinking := encodeAnthropicThinking(req.Reasoning, req.ReasoningBudgetTokens, maxTokens)
	if thinking == nil {
		return nil
	}
	if hasUnsignedOpenAIToolContinuation(req.Prompt) {
		return nil
	}
	return thinking
}

func appendOpenAIInboundAnthropicMessage(messages []anthropicMessage, message Message, previousWasTool bool) ([]anthropicMessage, bool) {
	encoded, ok := encodeOpenAIInboundAnthropicMessage(message)
	if !ok {
		return messages, previousWasTool && message.Role == RoleTool
	}

	if message.Role != RoleTool {
		if message.Role == RoleAssistant && len(messages) > 0 && messages[len(messages)-1].Role == string(RoleAssistant) && shouldMergeAssistantContent(messages[len(messages)-1].Content, encoded.Content) {
			if appendAnthropicMessageContent(&messages[len(messages)-1], encoded.Content) {
				return messages, false
			}
		}
		return append(messages, encoded), false
	}

	if previousWasTool && len(messages) > 0 {
		if appendAnthropicMessageContent(&messages[len(messages)-1], encoded.Content) {
			return messages, true
		}
	}
	return append(messages, encoded), true
}

func encodeOpenAIInboundAnthropicMessage(message Message) (anthropicMessage, bool) {
	parts := openAIInboundAnthropicParts(message.Parts)
	if len(parts) == 0 {
		return anthropicMessage{}, false
	}

	role := message.Role
	if role == RoleTool {
		role = RoleUser
	}
	return anthropicMessage{
		Role:    string(role),
		Content: encodeAnthropicContent(parts),
	}, true
}

func openAIInboundAnthropicParts(parts []Part) []Part {
	filtered := make([]Part, 0, len(parts))
	for _, part := range parts {
		if part.Type == PartReasoning && !isSignedOrRedactedReasoning(part.Reasoning) {
			continue
		}
		filtered = append(filtered, part)
	}
	return filtered
}

func appendAnthropicMessageContent(message *anthropicMessage, content any) bool {
	existing, ok := message.Content.([]anthropicContentBlock)
	if !ok {
		return false
	}
	added, ok := content.([]anthropicContentBlock)
	if !ok {
		return false
	}
	message.Content = append(existing, added...)
	return true
}

func shouldMergeAssistantContent(existing any, added any) bool {
	return anthropicContentHasBlockType(existing, "tool_use") || anthropicContentHasBlockType(added, "tool_use")
}

func anthropicContentHasBlockType(content any, blockType string) bool {
	blocks, ok := content.([]anthropicContentBlock)
	if !ok {
		return false
	}
	for _, block := range blocks {
		if block.Type == blockType {
			return true
		}
	}
	return false
}

func hasUnsignedOpenAIToolContinuation(prompt []Message) bool {
	lastAssistantHadToolCall := false
	lastAssistantHadSignedReasoning := false
	for _, message := range prompt {
		switch message.Role {
		case RoleSystem, RoleDeveloper:
			continue
		case RoleAssistant:
			lastAssistantHadToolCall = messageHasToolCall(message)
			lastAssistantHadSignedReasoning = messageHasSignedReasoning(message)
		case RoleTool:
			if lastAssistantHadToolCall && !lastAssistantHadSignedReasoning {
				return true
			}
		default:
			lastAssistantHadToolCall = false
			lastAssistantHadSignedReasoning = false
		}
	}
	return false
}

func messageHasToolCall(message Message) bool {
	for _, part := range message.Parts {
		if part.Type == PartToolCall && part.ToolCall != nil {
			return true
		}
	}
	return false
}

func messageHasSignedReasoning(message Message) bool {
	for _, part := range message.Parts {
		if part.Type == PartReasoning && isSignedOrRedactedReasoning(part.Reasoning) {
			return true
		}
	}
	return false
}

func isSignedOrRedactedReasoning(reasoning *ReasoningPart) bool {
	if reasoning == nil {
		return false
	}
	return strings.TrimSpace(reasoning.Signature) != "" || strings.TrimSpace(reasoning.Redacted) != ""
}
