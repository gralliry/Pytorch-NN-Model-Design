#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Description :

class Criterion:

    def __call__(self, predictions, targets):
        return self.forward(predictions, targets)

    def forward(self, predictions, targets):
        ...
        return
