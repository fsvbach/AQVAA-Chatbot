"""OpenTelemetry tracing setup for Arize Phoenix.

Call setup_tracing() once at startup. When tracing is disabled (or this
function is never called), all trace.get_tracer() calls return a no-op
tracer with virtually zero overhead — no if-guards needed at call sites.
"""
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

from logger.llm_logger import get_logger

logger = get_logger(__name__)


def setup_tracing(endpoint: str, project_name: str) -> None:
    """Configure the global OTel tracer provider to export to Phoenix.

    Safe to call multiple times — subsequent calls are ignored because a
    provider is already set after the first call.
    """
    resource = Resource.create({"service.name": project_name})
    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter(endpoint=endpoint)
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    logger.info("Tracing enabled → Phoenix at %s (project=%s)", endpoint, project_name)


def get_tracer(name: str) -> trace.Tracer:
    """Return a tracer for the given module name.

    Returns a no-op tracer when tracing has not been configured.
    """
    return trace.get_tracer(name)
