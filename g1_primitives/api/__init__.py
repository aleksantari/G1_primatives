"""The caller-facing API layer: the Robot facade, the action/perception primitives, and
their result/option types. Everything an agent host needs imports from here (re-exported
at the package top level); the layers below (motion/perception/grasp/hardware) are
implementation detail.
"""
