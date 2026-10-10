package protocolbridge

// Provider errors cross protocol boundaries in this package as the decoded
// JSON value each decoder happened to receive: a bare string for some gateways,
// an object for the three protocols we speak, and sometimes an object wrapped
// one level down. The readers below tolerate every shape so that an upstream
// failure survives the conversion instead of turning into "<nil>" or a Go map's
// fmt.Sprint rendering.

// errorMessage returns a human-readable message for a decoded provider error.
//
// A bare string is the message itself. An object may nest the real detail under
// "error" (Anthropic's {"type":"error","error":{...}}), so that level is tried
// first before the flat message keys.
func errorMessage(value any) string {
	switch typed := value.(type) {
	case nil:
		return ""
	case string:
		return typed
	case map[string]any:
		if nested, ok := typed["error"]; ok {
			if message := errorMessage(nested); message != "" {
				return message
			}
		}
		for _, key := range []string{"message", "error_description", "detail"} {
			if text, ok := typed[key].(string); ok && text != "" {
				return text
			}
		}
	}
	return ""
}

// errorCode returns the provider's error code, falling back to its error type.
// Unlike errorMessage it never treats a bare string as a code.
func errorCode(value any) string {
	switch typed := value.(type) {
	case map[string]any:
		if nested, ok := typed["error"]; ok {
			if code := errorCode(nested); code != "" {
				return code
			}
		}
		for _, key := range []string{"code", "type"} {
			if text, ok := typed[key].(string); ok && text != "" {
				return text
			}
		}
	}
	return ""
}

// errorParam returns the offending request field when the provider names one.
func errorParam(value any) string {
	switch typed := value.(type) {
	case map[string]any:
		if nested, ok := typed["error"]; ok {
			if param := errorParam(nested); param != "" {
				return param
			}
		}
		if text, ok := typed["param"].(string); ok {
			return text
		}
	}
	return ""
}

// anthropicErrorType maps a provider error code onto Anthropic's closed set of
// error types. Claude Code validates error.type, so an OpenAI code such as
// "server_error" must not be passed through verbatim.
func anthropicErrorType(value any) string {
	switch errorCode(value) {
	case "invalid_request_error", "authentication_error", "permission_error",
		"not_found_error", "rate_limit_error", "api_error", "overloaded_error":
		return errorCode(value)
	}
	return "api_error"
}
