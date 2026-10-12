# Lessons

## Always surface what's being run / its progress to the user
When running or monitoring any command/background job, proactively show the user
what's executing and its status — don't just act silently in the background and
only report at the end.
- Trigger: user said "sau này chạy gì thì cũng hiện lên cho tôi xem nha" (from now
  on, show me whatever runs).
- Applies to: shell commands, background tasks, long-running jobs (training,
  inference, batch scripts), monitors.

## Silent log ≠ dead process
A backgrounded Python script with no new log output is not necessarily dead —
Python block-buffers stdout by default when redirected to a file (non-tty).
- Always cross-check via `ps aux` for the actual PID before concluding a process
  died, rather than trusting an empty/silent log alone.
- Use `python3 -u` (unbuffered) for any script whose progress will be monitored
  via log-tailing, to avoid this ambiguity.
- Never let two instances of a script that does a truncating `"w"`-mode write to
  the same output file run concurrently — this causes silent data corruption
  (confirmed: interleaved writes produced invalid JSON lines in `poses.jsonl`).
  Before relaunching a "stuck" job, verify via `ps aux` that no prior instance is
  still alive, and kill it first if so.
