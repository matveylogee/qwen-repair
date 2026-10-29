# Project rules

- Current scope is M1: bounded Colab feasibility with Qwen, LoRA SFT/GRPO smoke
  tests, a small KodCode pilot and an isolated executor. Preserve the M0 demo.
- GPU training, isolation and resume are verified only by actual runs. Never
  label mocks, synthetic rewards or CPU contract tests as GPU validation.
- Use pinned model/data revisions and a separate GPU dependency profile. Keep
  smoke runs short; do not publish datasets or start paid/long training jobs.
- Keep code in `src/qwen_repair`; notebooks only install and call the CLI.
- Never execute candidate code on the host. The fake executor only looks up
  prepared results. AST parsing is validation, not a sandbox.
- Construct prompts through `PromptView`. Never serialize a task, reference,
  private tests or private evaluations into policy history.
- Keep fixed attempt budgets, select the last attempt even if invalid, and
  exclude infrastructure failures from ordinary success denominators.
- Mark every demo artifact `synthetic_demo=true`; never describe fixture results
  as Qwen quality or a real test execution.
- Preserve user changes. Keep dependencies small and GPU imports out of CPU paths.
- Run `python -m pytest`, `qrepair doctor`, demo and summarize after relevant changes.
- Write identifiers in English and user documentation in Russian. Report only
  checks actually run, including their limitations.
