package protocolbridge

import (
	"fmt"
	"strings"
)

// A conversion between two protocols cannot always express everything the
// source said. The pieces below make those losses visible: every converter
// reports what it dropped onto the object it was building, and a host can read
// the report or ask for the conversion to fail instead.
//
// This replaces two older behaviours. One was silence: a dropped image or tool
// simply disappeared. The other was worse — writing a bracketed
// proxy-compatibility sentence into the request, which changed the bytes the
// model was asked about, invalidated the cached prompt prefix, and looked to
// the model like something the user had typed. The guard test in
// diagnostics_test.go keeps that from coming back.

// Severity grades what a loss costs the caller.
type Severity string

const (
	// SeverityInfo records something that was rewritten into an equivalent
	// form, or dropped without changing the answer. It is never fatal: a host
	// asking for strictness does not want a request refused over scratch data
	// the model did not need.
	SeverityInfo Severity = "info"

	// SeverityWarning records content the target protocol cannot express, so
	// the model no longer sees it.
	SeverityWarning Severity = "warning"

	// SeverityError records a loss that changes what the model is asked to do
	// or answer with, such as a tool disappearing from the request.
	SeverityError Severity = "error"
)

// Loss codes. Hosts can switch on these to alert on particular losses; the
// message carries the detail.
const (
	// LossUnsupportedFileInput is a file whose media type or encoding the
	// target protocol has no representation for.
	LossUnsupportedFileInput = "unsupported_file_input"

	// LossUnsupportedFileReference is a file identified only by an opaque
	// provider file id, which the target protocol cannot resolve.
	LossUnsupportedFileReference = "unsupported_file_reference"

	// LossUnsupportedTool is a tool the target protocol cannot express, so the
	// model cannot call it.
	LossUnsupportedTool = "unsupported_tool"

	// LossDroppedStopSequences is a stop sequence the target protocol has no
	// field for.
	LossDroppedStopSequences = "dropped_stop_sequences"

	// LossDroppedReasoning is reasoning content the target protocol has no
	// field for. Losing an earlier turn's reasoning is usually harmless, so the
	// chat encoder reports it at SeverityInfo.
	LossDroppedReasoning = "dropped_reasoning"

	// LossDroppedChoice is an additional completion the target protocol can
	// only represent one of.
	LossDroppedChoice = "dropped_choice"
)

// LossPolicy decides whether reported losses fail the conversion.
type LossPolicy string

const (
	// LossPolicyAllow records losses and completes the conversion. It is the
	// zero value, so a caller that says nothing gets the conversion.
	LossPolicyAllow LossPolicy = ""

	// LossPolicySafe refuses a conversion that would drop something the model
	// needs, which is the error-severity losses, and allows the rest.
	LossPolicySafe LossPolicy = "safe"

	// LossPolicyStrict refuses a conversion that dropped any content, which is
	// every loss above SeverityInfo.
	LossPolicyStrict LossPolicy = "strict"
)

// ConversionError reports the losses a LossPolicy refused to accept. It is
// returned instead of the encoded request, and the losses are also attached to
// the object so a caller that wants to log them can.
type ConversionError struct {
	Warnings []Warning
}

func (e *ConversionError) Error() string {
	if e == nil || len(e.Warnings) == 0 {
		return "protocolbridge: conversion loss"
	}
	parts := make([]string, 0, len(e.Warnings))
	for _, warning := range e.Warnings {
		code := warning.Code
		if code == "" {
			code = "loss"
		}
		if warning.Path != "" {
			parts = append(parts, fmt.Sprintf("%s at %s: %s", code, warning.Path, warning.Message))
			continue
		}
		parts = append(parts, fmt.Sprintf("%s: %s", code, warning.Message))
	}
	return "protocolbridge: conversion loss: " + strings.Join(parts, "; ")
}

// lossRecorder collects the losses one conversion incurred.
//
// A nil recorder is valid and discards, so a helper can be called from a path
// that does not report without every caller having to check.
type lossRecorder struct {
	from     Protocol
	to       Protocol
	warnings []Warning
}

func newLossRecorder(from, to Protocol) *lossRecorder {
	return &lossRecorder{from: from, to: to}
}

// report records one loss. An empty severity means warning, because the losses
// worth reporting by default are the ones that dropped something.
func (r *lossRecorder) report(code, path, message string, severity Severity) {
	if r == nil {
		return
	}
	if severity == "" {
		severity = SeverityWarning
	}
	r.warnings = append(r.warnings, Warning{
		Code:     code,
		Message:  message,
		Severity: severity,
		Path:     path,
		From:     r.from,
		To:       r.to,
	})
}

// reportUnsupportedFile records a file the target protocol cannot carry.
func (r *lossRecorder) reportUnsupportedFile(path string, file *FilePart) {
	if r == nil {
		return
	}
	if file != nil && file.URL == "" && file.Data == "" && file.FileID != "" {
		r.report(LossUnsupportedFileReference, path,
			fmt.Sprintf("file %s is referenced by provider file id %q, which this protocol cannot resolve", describeFile(file), file.FileID),
			SeverityWarning)
		return
	}
	r.report(LossUnsupportedFileInput, path,
		fmt.Sprintf("file input %s has no representation in this protocol and was omitted", describeFile(file)),
		SeverityWarning)
}

// attachTo publishes the collected losses onto the object being built and
// applies the policy, so the caller sees both the report and the refusal.
func (r *lossRecorder) attachTo(req *LLMRequest, policy LossPolicy) error {
	if r == nil || len(r.warnings) == 0 {
		return nil
	}
	if req != nil {
		req.Warnings = append(req.Warnings, r.warnings...)
	}
	return rejectLosses(policy, r.warnings)
}

// attachToResponse publishes the collected losses onto the response being
// built and applies the policy.
func (r *lossRecorder) attachToResponse(resp *LLMResponse, policy LossPolicy) error {
	if r == nil || len(r.warnings) == 0 {
		return nil
	}
	if resp != nil {
		resp.Warnings = append(resp.Warnings, r.warnings...)
	}
	return rejectLosses(policy, r.warnings)
}

// rejectLosses returns a *ConversionError when the policy refuses any of the
// reported losses.
func rejectLosses(policy LossPolicy, warnings []Warning) error {
	switch policy {
	case LossPolicySafe:
		rejected := make([]Warning, 0, len(warnings))
		for _, warning := range warnings {
			if warning.Severity == SeverityError {
				rejected = append(rejected, warning)
			}
		}
		if len(rejected) == 0 {
			return nil
		}
		return &ConversionError{Warnings: rejected}
	case LossPolicyStrict:
		rejected := make([]Warning, 0, len(warnings))
		for _, warning := range warnings {
			if warning.Severity != SeverityInfo {
				rejected = append(rejected, warning)
			}
		}
		if len(rejected) == 0 {
			return nil
		}
		return &ConversionError{Warnings: rejected}
	default:
		return nil
	}
}

// describeFile names a file for a loss message, preferring whatever would let
// the caller find it again.
func describeFile(file *FilePart) string {
	if file == nil {
		return "an unknown file"
	}
	identifier := strings.TrimSpace(file.Filename)
	if identifier == "" {
		identifier = strings.TrimSpace(file.FileID)
	}
	if identifier == "" {
		identifier = strings.TrimSpace(file.MediaType)
	}
	if identifier == "" {
		identifier = "unknown"
	}
	if file.MediaType != "" && file.MediaType != identifier {
		return fmt.Sprintf("%q (%s)", identifier, file.MediaType)
	}
	return fmt.Sprintf("%q", identifier)
}
