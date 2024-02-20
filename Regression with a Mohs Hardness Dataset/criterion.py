#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

from torch import nn


class Criterion:
    criterion = nn.MSELoss()

    def __call__(self, predictions, targets):
        loss = self.criterion(predictions, targets)
        ...
        return loss
