# SPDX-FileCopyrightText: Copyright (c) 2026 SpaceAI
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import abc
from typing import Any


class SensorLayer(abc.ABC):
    name = "base"

    @abc.abstractmethod
    def capture(self, sim_state: Any) -> Any:
        raise NotImplementedError
