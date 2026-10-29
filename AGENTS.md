# Project rules

- Current scope is M0: a CPU-only synthetic repair demo. Training, model downloads,
  dataset ingestion and real executors require a subsequent task.
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
