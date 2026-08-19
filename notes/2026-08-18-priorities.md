# Project priorities and decisions

Recorded: 2026-08-18

This note summarizes the project owner's answers to the first adversarial interview. It is a decision record, not an empirical result.

## Priority order

1. Energy efficiency and capability per parameter
2. Continual learning and ecologies of models
3. Theory of mind as experimental target and philosophical motivation

Theory of mind is currently useful for building intuitions about model design. Available local hardware makes machine consciousness an inappropriate near-term empirical or ethical claim for this project.

## Learning

The implementation does not have to update weights. There is nevertheless a strong prior in favor of investigating weight change because persistent plasticity is a conspicuous difference between deployed LLMs and familiar biological learners.

Facts, skills, values, behavior, and other model properties are all potential learning material. Whether they should share one update mechanism remains an open safety and systems question rather than a settled design choice.

## Activation trajectories

The motivating intuition is that differences among many activation trajectories contain information beyond a coordinate chart. The project should test whether this information is:

- predictive of teacher behavior on unseen inputs;
- transferable to a smaller student;
- invariant to harmless changes of basis;
- causally implicated rather than merely decodable; and
- more useful than teacher logits, labels, or additional examples at matched cost.

The activation-geometry framing should be abandoned or weakened if trajectories carry no transferable information without the teacher weights, or if the proposed geometry is only a lossless restatement of the existing computation with no compression or predictive gain.

## Hardware envelope

Available machines include:

- Apple M2 with 16 GB unified memory;
- AMD Ryzen 7 7700 with 64 GB RAM and an NVIDIA RTX 4060 with 8 GB VRAM.

The initial mechanism experiment should fit comfortably on the latter and retain a CPU-compatible reduced configuration. Larger-model replication is optional and follows only after the small test produces a real effect.
