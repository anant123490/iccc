# ICDAS production candidates

No ICDAS model weights are currently stored in this tree. Previous run checkpoints and evaluations were removed for a fresh development cycle.

The model architecture, training code, and original configuration remain under ml/. Future deployment requires an evaluated five-class softmax checkpoint at models/icdas/current/deploy.keras. Never load a four-output ordinal checkpoint.
