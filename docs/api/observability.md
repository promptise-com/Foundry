# Observability API Reference

Plug-and-play observability system for Promptise agents. Controls what gets observed, how much detail is captured, and where events are sent. Includes configurable transporters and a callback handler that bridges LangChain events into the observability collector.

## Core Collector

### ObservabilityCollector

::: promptise.observability.ObservabilityCollector
    options:
      show_source: false
      heading_level: 4

### TimelineEntry

::: promptise.observability.TimelineEntry
    options:
      show_source: false
      heading_level: 4

### TimelineEventType

::: promptise.observability.TimelineEventType
    options:
      show_source: false
      heading_level: 4

### AgentRun

::: promptise.observability.AgentRun
    options:
      show_source: false
      heading_level: 4

### get_current_run

::: promptise.observability.get_current_run
    options:
      show_source: false
      heading_level: 4

### redact_sensitive

::: promptise.observability.redact_sensitive
    options:
      show_source: false
      heading_level: 4

### TimelineEventCategory

::: promptise.observability.TimelineEventCategory
    options:
      show_source: false
      heading_level: 4

---

## Configuration

## ObservabilityConfig

::: promptise.observability_config.ObservabilityConfig
    options:
      show_source: false
      heading_level: 3

## ObserveLevel

::: promptise.observability_config.ObserveLevel
    options:
      show_source: false
      heading_level: 3

## TransporterType

::: promptise.observability_config.TransporterType
    options:
      show_source: false
      heading_level: 3

## ExportFormat

Backward-compatible alias for `TransporterType`.

## Transporters

### create_transporters

::: promptise.observability_transporters.create_transporters
    options:
      show_source: false
      heading_level: 4

### BaseTransporter

::: promptise.observability_transporters.BaseTransporter
    options:
      show_source: false
      heading_level: 4

### HTMLReportTransporter

::: promptise.observability_transporters.HTMLReportTransporter
    options:
      show_source: false
      heading_level: 4

### JSONFileTransporter

::: promptise.observability_transporters.JSONFileTransporter
    options:
      show_source: false
      heading_level: 4

### StructuredLogTransporter

::: promptise.observability_transporters.StructuredLogTransporter
    options:
      show_source: false
      heading_level: 4

### ConsoleTransporter

::: promptise.observability_transporters.ConsoleTransporter
    options:
      show_source: false
      heading_level: 4

### PrometheusTransporter

::: promptise.observability_transporters.PrometheusTransporter
    options:
      show_source: false
      heading_level: 4

### OTLPTransporter

::: promptise.observability_transporters.OTLPTransporter
    options:
      show_source: false
      heading_level: 4

### WebhookTransporter

::: promptise.observability_transporters.WebhookTransporter
    options:
      show_source: false
      heading_level: 4

### CallbackTransporter

::: promptise.observability_transporters.CallbackTransporter
    options:
      show_source: false
      heading_level: 4

## Callback Handler

### PromptiseCallbackHandler

::: promptise.callback_handler.PromptiseCallbackHandler
    options:
      show_source: false
      heading_level: 4
