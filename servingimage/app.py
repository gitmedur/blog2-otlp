from __future__ import annotations

import argparse
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from pydantic import BaseModel, Field, model_validator


logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("iris-otel-runtime")


def configure_tracing(service_name: str) -> trace.Tracer:
    """Configure OTLP tracing.

    OTLPSpanExporter reads standard OTEL_EXPORTER_OTLP_* environment variables.
    """
    resource = Resource.create(
        {
            "service.name": service_name,
            "service.namespace": os.getenv("OTEL_SERVICE_NAMESPACE", "iris"),
            "service.version": os.getenv("SERVICE_VERSION", "1.0.0"),
            "deployment.environment.name": os.getenv(
                "DEPLOYMENT_ENVIRONMENT", "openshift-ai"
            ),
            "k8s.namespace.name": os.getenv("POD_NAMESPACE", "iris"),
            "k8s.pod.name": os.getenv("POD_NAME", "unknown"),
            "k8s.container.name": "kserve-container",
            "kserve.runtime": "iris-otel-runtime",
        }
    )

    provider = TracerProvider(resource=resource)
    exporter = OTLPSpanExporter()
    provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(provider)
    return trace.get_tracer("iris-otel-runtime")


class Parameters(BaseModel):
    content_type: str | None = None


class InferInput(BaseModel):
    name: str
    shape: list[int]
    datatype: Literal["FP32", "FP64"]
    data: list[Any]
    parameters: Parameters | None = None


class InferRequest(BaseModel):
    id: str | None = None
    inputs: list[InferInput] = Field(min_length=1)
    parameters: dict[str, Any] | None = None

    @model_validator(mode="after")
    def validate_single_input(self) -> "InferRequest":
        if len(self.inputs) != 1:
            raise ValueError("This demo runtime accepts exactly one input tensor.")
        return self


class ModelRuntime:
    def __init__(self, model_name: str, model_dir: Path, tracer: trace.Tracer):
        self.model_name = model_name
        self.model_dir = model_dir
        self.tracer = tracer
        self.model: Any | None = None
        self.model_path: Path | None = None
        self.loaded_at: float | None = None

    def find_model(self) -> Path:
        explicit_path = os.getenv("MODEL_PATH")
        if explicit_path:
            path = Path(explicit_path)
            if not path.is_file():
                raise FileNotFoundError(f"MODEL_PATH does not exist: {path}")
            return path

        preferred_names = (
            "iris_model.pkl",
            "model.pkl",
            "model.joblib",
            "model.pickle",
        )
        for name in preferred_names:
            path = self.model_dir / name
            if path.is_file():
                return path

        candidates = sorted(
            p
            for pattern in ("*.pkl", "*.pickle", "*.joblib")
            for p in self.model_dir.rglob(pattern)
            if p.is_file()
        )
        if not candidates:
            raise FileNotFoundError(
                f"No .pkl, .pickle, or .joblib model found below {self.model_dir}"
            )
        if len(candidates) > 1:
            LOGGER.warning("Multiple model files found; selecting %s", candidates[0])
        return candidates[0]

    def load(self) -> None:
        with self.tracer.start_as_current_span("load-model") as span:
            model_path = self.find_model()
            span.set_attribute("model.name", self.model_name)
            span.set_attribute("model.path", str(model_path))
            span.set_attribute("model.framework", "scikit-learn")

            started = time.perf_counter()
            self.model = joblib.load(model_path)
            elapsed_ms = (time.perf_counter() - started) * 1000

            self.model_path = model_path
            self.loaded_at = time.time()
            span.set_attribute("model.load.duration_ms", elapsed_ms)
            span.set_attribute(
                "model.python_type",
                f"{type(self.model).__module__}.{type(self.model).__name__}",
            )
            LOGGER.info("Loaded model %s from %s in %.2f ms", self.model_name, model_path, elapsed_ms)

    @property
    def ready(self) -> bool:
        return self.model is not None

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model is not loaded")
        return np.asarray(self.model.predict(features))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name",
        default=os.getenv("MODEL_NAME", "iris-otel"),
    )
    parser.add_argument(
        "--model-dir",
        default=os.getenv("MODEL_DIR", "/mnt/models"),
    )
    parser.add_argument(
        "--http-port",
        type=int,
        default=int(os.getenv("HTTP_PORT", "8080")),
    )
    return parser.parse_args()


ARGS = parse_args()
TRACER = configure_tracing(
    os.getenv("OTEL_SERVICE_NAME", f"{ARGS.model_name}-runtime")
)
RUNTIME = ModelRuntime(ARGS.model_name, Path(ARGS.model_dir), TRACER)


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        RUNTIME.load()
    except Exception:
        LOGGER.exception("Model loading failed")
        raise
    yield
    provider = trace.get_tracer_provider()
    shutdown = getattr(provider, "shutdown", None)
    if callable(shutdown):
        shutdown()


app = FastAPI(
    title="OpenTelemetry-instrumented KServe V2 runtime",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/v2/health/live")
async def live() -> dict[str, bool]:
    return {"live": True}


@app.get("/v2/health/ready")
async def ready() -> dict[str, bool]:
    if not RUNTIME.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"ready": True}


@app.get("/v2/models/{model_name}")
async def model_metadata(model_name: str) -> dict[str, Any]:
    if model_name != RUNTIME.model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    return {
        "name": RUNTIME.model_name,
        "versions": ["1"],
        "platform": "sklearn",
        "inputs": [
            {
                "name": "predict",
                "datatype": "FP32",
                "shape": [-1, 4],
            }
        ],
        "outputs": [
            {
                "name": "predict",
                "datatype": "INT64",
                "shape": [-1, 1],
            }
        ],
    }


@app.get("/v2/models/{model_name}/ready")
async def model_ready(model_name: str) -> dict[str, bool]:
    if model_name != RUNTIME.model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    if not RUNTIME.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")
    return {"ready": True}


@app.post("/v2/models/{model_name}/infer")
async def infer(
    model_name: str,
    payload: InferRequest,
    request: Request,
) -> dict[str, Any]:
    if model_name != RUNTIME.model_name:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    if not RUNTIME.ready:
        raise HTTPException(status_code=503, detail="Model is not ready")

    input_tensor = payload.inputs[0]
    current_span = trace.get_current_span()
    current_span.set_attribute("model.name", RUNTIME.model_name)
    current_span.set_attribute("model.framework", "scikit-learn")
    current_span.set_attribute("inference.protocol", "kserve-v2")
    current_span.set_attribute("inference.input.name", input_tensor.name)
    current_span.set_attribute("inference.input.datatype", input_tensor.datatype)
    current_span.set_attribute("inference.request.id_present", payload.id is not None)
    current_span.set_attribute(
        "http.request.header.user_agent",
        request.headers.get("user-agent", "unknown"),
    )

    try:
        with TRACER.start_as_current_span("parse-kserve-v2-request") as span:
            features = np.asarray(input_tensor.data, dtype=np.float32)
            span.set_attribute("inference.input.rank", features.ndim)
            span.set_attribute("inference.input.element_count", int(features.size))

        with TRACER.start_as_current_span("validate-input-shape") as span:
            if features.ndim == 1:
                features = features.reshape(1, -1)
            if features.ndim != 2 or features.shape[1] != 4:
                span.set_attribute("validation.success", False)
                raise ValueError(
                    f"Expected shape [batch, 4], received {list(features.shape)}"
                )
            span.set_attribute("validation.success", True)
            span.set_attribute("inference.batch_size", int(features.shape[0]))
            span.set_attribute("inference.feature_count", int(features.shape[1]))

        with TRACER.start_as_current_span("sklearn.predict") as span:
            started = time.perf_counter()
            predictions = RUNTIME.predict(features)
            elapsed_ms = (time.perf_counter() - started) * 1000
            span.set_attribute("model.predict.duration_ms", elapsed_ms)
            span.set_attribute("inference.batch_size", int(features.shape[0]))
            span.set_attribute("inference.output.count", int(predictions.size))

        with TRACER.start_as_current_span("build-kserve-v2-response"):
            output_data = predictions.astype(np.int64).reshape(-1, 1).tolist()
            return {
                "model_name": RUNTIME.model_name,
                "model_version": "1",
                "id": payload.id,
                "parameters": {},
                "outputs": [
                    {
                        "name": "predict",
                        "shape": [len(output_data), 1],
                        "datatype": "INT64",
                        "parameters": {"content_type": "np"},
                        "data": output_data,
                    }
                ],
            }
    except ValueError as exc:
        current_span.record_exception(exc)
        current_span.set_status(
            trace.Status(trace.StatusCode.ERROR, str(exc))
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Inference failed")
        current_span.record_exception(exc)
        current_span.set_status(
            trace.Status(trace.StatusCode.ERROR, "Inference failed")
        )
        raise HTTPException(status_code=500, detail="Inference failed") from exc


# Exclude health probes so they do not flood Tempo with low-value traces.
FastAPIInstrumentor.instrument_app(
    app,
    excluded_urls=r"/v2/health/live,/v2/health/ready,/v2/models/.*/ready",
)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=ARGS.http_port,
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
        proxy_headers=True,
    )
