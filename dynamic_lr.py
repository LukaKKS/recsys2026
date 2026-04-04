"""
CLS 이론에 기반한 동적 학습률 계산기.

CLS 매핑:
    - 저빈도(롱테일/해마): 빠른 학습률 → 새로운 개별 패턴을 빠르게 습득
    - 고빈도(숏헤드/신피질): 낮은 학습률 → 안정적인 공통 패턴 유지

사용 예시:
    dynamic_lr = DynamicLR(base_lr=0.1)

    # 저빈도 특성 (access_count=5)
    lr = dynamic_lr.get_lr(is_frequent=False, access_count=5)

    # 고빈도 특성 (access_count=1000)
    lr = dynamic_lr.get_lr(is_frequent=True, access_count=1000)
"""

from __future__ import annotations

import math


class DynamicLR:
    """
    빈도 기반 동적 학습률 계산기.

    Parameters
    ----------
    base_lr : float
        기본 학습률 (DLRM의 --learning-rate와 동일한 값 권장).
    long_tail_multiplier : float
        저빈도 특성의 학습률 배율 (> 1 → 빠른 학습).
    short_head_multiplier : float
        고빈도 특성의 학습률 배율 (< 1 → 안정적 학습).
    decay_factor : float
        접근 횟수에 따른 지수 감쇠 계수.
        lr *= exp(-decay_factor * (access_count - 1))
    min_lr : float
        학습률 하한.
    max_lr : float
        학습률 상한.
    """

    def __init__(
        self,
        base_lr: float = 0.1,
        long_tail_multiplier: float = 2.0,
        short_head_multiplier: float = 0.5,
        decay_factor: float = 0.01,
        min_lr: float = 1e-5,
        max_lr: float = 1.0,
    ):
        self.base_lr = base_lr
        self.long_tail_multiplier = long_tail_multiplier
        self.short_head_multiplier = short_head_multiplier
        self.decay_factor = decay_factor
        self.min_lr = min_lr
        self.max_lr = max_lr

    def get_lr(self, is_frequent: bool, access_count: int = 1) -> float:
        """
        학습률 계산.

        Parameters
        ----------
        is_frequent : bool
            True  → 고빈도(숏헤드)  → short_head_multiplier 적용
            False → 저빈도(롱테일)  → long_tail_multiplier  적용
        access_count : int
            해당 특성의 누적 접근 횟수.
            횟수가 많을수록 지수 감쇠로 학습률이 점진적으로 낮아집니다.

        Returns
        -------
        float
            [min_lr, max_lr] 범위 내에 클램핑된 최종 학습률.
        """
        multiplier = (
            self.short_head_multiplier if is_frequent else self.long_tail_multiplier
        )
        # 접근 횟수가 늘수록 완만하게 감소 (첫 접근은 감쇠 없음)
        decay = math.exp(-self.decay_factor * max(access_count - 1, 0))
        lr = self.base_lr * multiplier * decay
        return float(max(self.min_lr, min(self.max_lr, lr)))

    def get_lr_ratio(self, is_frequent: bool) -> float:
        """
        접근 횟수를 무시하고 배율만 반환 (빠른 비교용).

        Returns
        -------
        float
            short_head_multiplier 또는 long_tail_multiplier.
        """
        return self.short_head_multiplier if is_frequent else self.long_tail_multiplier

    def __repr__(self) -> str:
        return (
            f"DynamicLR(base_lr={self.base_lr}, "
            f"lt×{self.long_tail_multiplier}, "
            f"sh×{self.short_head_multiplier}, "
            f"decay={self.decay_factor})"
        )
