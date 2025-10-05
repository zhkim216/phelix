from typing import Any, Dict

import lightning as L
import torch


class _DummyInnerModel(torch.nn.Module):
    """Logger.watch 및 외부 호출과의 호환을 위한 최소 스텁 모델."""

    def __init__(self) -> None:
        super().__init__()
        # 파라미터가 아예 없으면 일부 훅이 불편해할 수 있어 아주 작은 더미 파라미터를 둠
        self._stub = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self._scale_factors: Dict[str, Any] | None = None
        self._sigma_data: Any = None

    def set_scale_factors(self, scale_factors: Dict[str, Any]) -> None:
        self._scale_factors = scale_factors

    def set_sigma_data(self, sigma_data: Any) -> None:
        self._sigma_data = sigma_data

    def forward(self, *_args, **_kwargs):  # pragma: no cover - 사용되지 않음
        return {}


class LitDummySeqDenoiser(L.LightningModule):
    """데이터 로딩 경로만 검증하기 위한 더미 Lightning 모듈.

    - 입력 배치 텐서들이 유한한지(NaN/Inf)만 검사하고, 작은 상수 로스를 반환합니다.
    - 실제 모델 연산, 옵티마이저 스텝, 스케줄러 스텝은 수행하지 않습니다.
    - AMP/분산/핀메모리/디바이스 전송 등 Trainer 경로는 그대로 재현됩니다.
    """

    def __init__(self, cfg: Any | None = None, log_non_finite_examples: bool = True, constant_loss: float = 0.0):
        super().__init__()
        self.save_hyperparameters(ignore=["cfg"])  # cfg는 크고 직렬화 불필요
        self.cfg = cfg
        self.log_non_finite_examples = log_non_finite_examples
        # 아주 작은 상수 로스. 0이면 그래프가 비어 있을 수 있으니 미세한 연산 포함.
        self.constant_loss = float(constant_loss)
        # train_seq_denoiser에서 참조하는 `.model` 속성 스텁
        self.model = _DummyInnerModel()
        # DDP가 요구하는 학습 가능한 파라미터를 제공 (forward에서 0 계수로 참조)
        self._ddp_stub = torch.nn.Parameter(torch.zeros(1), requires_grad=True)

    def _assert_batch_finite(self, batch: Dict[str, Any]) -> list[str]:
        bad_keys: list[str] = []
        for k, v in batch.items():
            if torch.is_tensor(v):
                # dtype이 bool/int여도 isfinite는 True를 반환함. float 계열만 엄밀 체크.
                needs_check = torch.is_floating_point(v) or torch.is_complex(v)
                if needs_check and not torch.isfinite(v).all():
                    bad_keys.append(k)
        return bad_keys

    def forward(self, batch: Dict[str, Any]) -> torch.Tensor:
        # 최소 연산으로 작은 스칼라를 만들어 그래프를 유지
        device = self.device
        loss_tensor = torch.tensor(self.constant_loss, dtype=torch.float32, device=device)
        # 미세한 연산을 추가하여 AMP/그래프 경로를 유지
        # DDP unused parameter 검사를 통과하도록 더미 파라미터를 0 계수로 참조
        loss_tensor = loss_tensor + torch.zeros((), dtype=torch.float32, device=device) + (self._ddp_stub.sum() * 0.0)
        return loss_tensor

    def training_step(self, batch: Dict[str, Any], _: int) -> torch.Tensor:
        bad_keys = self._assert_batch_finite(batch)
        if bad_keys and self.log_non_finite_examples:
            ex_ids = batch.get("example_id", [])
            self.log_dict({"non_finite_keys": float(len(bad_keys) > 0)}, prog_bar=True)
            self.print(f"[LitDummySeqDenoiser] Non-finite tensors in keys={bad_keys}; example_ids={ex_ids}")

        loss = self.forward(batch)
        # 로그를 약간 남겨 학습 루프가 도는지 확인
        self.log("train_loss", loss.detach(), prog_bar=True, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: Dict[str, Any], _: int) -> torch.Tensor:
        bad_keys = self._assert_batch_finite(batch)
        if bad_keys and self.log_non_finite_examples:
            ex_ids = batch.get("example_id", [])
            self.print(f"[LitDummySeqDenoiser] (val) Non-finite tensors in keys={bad_keys}; example_ids={ex_ids}")

        loss = self.forward(batch)
        self.log("val_loss", loss.detach(), prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        # 옵티마이저가 필요 없지만, Lightning은 반환을 기대할 수 있으므로 더미 옵티마이저를 반환
        # 파라미터가 없으면 에러이므로, register_buffer와 달리 더미 파라미터를 하나 둠
        if not any(p.requires_grad for p in self.parameters()):
            # 더미 파라미터 추가 (그래디언트 계산은 안 함)
            self._dummy = torch.nn.Parameter(torch.zeros(1), requires_grad=True)
        optimizer = torch.optim.SGD([p for p in self.parameters() if p.requires_grad], lr=1e-6)
        return optimizer


