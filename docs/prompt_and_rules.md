# Prompt and Rules for Server Agent (CSIG Track 1)

## Environment Rules & Inventory (Summarized from prompt.md and server_agent_prompt.md)
- **Competition**: CSIG2026 JinSight Cup Track 1 (IR Video Satellite Moving Target Detection).
- **Goal**: Implement a CPU-runnable, offline-testable engineering skeleton that does not require weights or real data initially.
- **Strict Boundaries**:
  - No model training or fine-tuning in the first round.
  - No large scale evaluation on training/val sets.
  - No upload to Codabench.
  - Do not use unofficial data mirrors.
- **Hardware constraints & checks**:
  - Requires > 250 GB free disk space to download data.
  - The agent should only proceed with data download if the space allows.
- **Coordinates Warning**:
  - Internal processing must use (x=column, y=row). The default submission order is xy, but there is a risk the online scorer might expect yx (based on legacy DeepPro row, col scripts). The pipeline must support a toggle for this.

## Server Environment Configuration
- Based on scripts/gpu_session.sh, GPU tools (nvidia-smi, PyTorch with CUDA) should be run in the managed session to ensure character devices are recreated.
- Python 3.9+ is expected.
