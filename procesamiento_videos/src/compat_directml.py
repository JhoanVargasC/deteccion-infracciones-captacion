"""Parche para ejecutar Ultralytics sobre torch-directml (GPU AMD en Windows).

Corrige tres fallas del backend DirectML:
1. torch.inference_mode() falla. Se reemplaza por torch.no_grad(), equivalente en inferencia.
2. AutoBackend.warmup() ejecuta un NMS clásico con indexado booleano que DirectML no admite.
   Es solo un precalentamiento, así que se anula.
3. El indexado booleano sobre tensores de DirectML también falla en el postproceso. El tensor
   de predicciones, que es pequeño, se mueve a CPU justo antes del NMS. La red sigue en la GPU.
Por la falla 3, solo los modelos end2end (YOLO26) funcionan en esta GPU.

Uso: import compat_directml, antes de importar ultralytics.
"""
import torch


class _SinGradienteComoInferencia:
    def __init__(self, mode=True):
        self._cm = torch.no_grad()

    def __enter__(self):
        return self._cm.__enter__()

    def __exit__(self, *args):
        return self._cm.__exit__(*args)

    def __call__(self, fn):
        return torch.no_grad()(fn)


torch.inference_mode = _SinGradienteComoInferencia

from ultralytics.nn.autobackend import AutoBackend  # noqa: E402
from ultralytics.utils import nms as _nms  # noqa: E402

AutoBackend.warmup = lambda self, *a, **k: None

_nms_original = _nms.non_max_suppression


def _a_cpu(x):
    if torch.is_tensor(x):
        return x.cpu() if x.device.type != "cpu" else x
    if isinstance(x, (list, tuple)):
        return type(x)(_a_cpu(v) for v in x)
    return x


def _nms_en_cpu(prediccion, *args, **kwargs):
    return _nms_original(_a_cpu(prediccion), *args, **kwargs)


_nms.non_max_suppression = _nms_en_cpu


def dispositivo(nombre):
    """Dispositivo de torch según parametros.yaml: 'directml' o el nombre que acepte torch."""
    if nombre == "directml":
        import torch_directml
        return torch_directml.device()
    return nombre
