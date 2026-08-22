# tasks -- the active silo

Dev-facing, mirroring `dopli/tasks/`. Everything live lives here and
nowhere else: plans, handoffs, assessments awaiting a ruling, open
questions. If work is planned but not on [board.md](board.md), it does
not exist.

Nothing leaves this silo while a gate inside it is still open. A doc
carrying "AWAITING USER REVIEW", an unanswered question, or a decision
the user has not made stays in tasks/ until that gate closes.

## Files

- [board.md](board.md) -- the active queue. One entry per open task. A
  chat or human should be able to pick a task by reading this file
  alone.
- [history.md](history.md) -- rolling log of recently closed tasks
  (roughly the last 15). Older closures live only in archive/.
- active/ -- the doc for a task being worked. Named
  `YYYY-MM-DD_kebab-slug.md` by the date the task was opened.
- on_hold/ -- offshoots and inquiries spawned off a task that nobody
  is on: gated on a user answer, parked for capacity, waiting on a
  paper. Same naming. Every on_hold doc still has a board line, so
  on_hold/ never becomes an off-book backlog.

A task doc is a flat `.md`. It is promoted to a folder of the same
name only once it accumulates extra artifacts -- figures, sub-docs, a
decision annex -- and then the main doc keeps the folder's name plus
`.md` inside it.

## Decisions live in the task doc

There is no decisions inbox. When a task needs a ruling, the question
goes into the task doc as an inline slot the user edits in place:

    DECIDE: <the question, with the options and their costs>
    (default: <what a landing session assumes if left blank>)
    A:

Splitting an open question into its own task is exactly the fracturing
this structure exists to prevent. The board line says what the task is
waiting on; the doc holds the question.

## Board entry format

    - [PENDING] <one-line task statement> -- GAIN <estimate>,
      EFFORT S|M|L, RISK low|med|high. Source: <link to archive doc or
      library note>. Detail: <link to active/ or on_hold/ file, if any>.

Status markers: [PENDING] (open), [ACTIVE] (a chat/human is on it --
add the date), [BLOCKED] (say on what). Group entries under `##`
section headers by repo area (utespac-core, migration, validation,
ec-coherent, raw-processing, meta).

## Lifecycle

1. Task is born: add a board line (plus an active/ file if
   non-trivial).
2. Task is picked up: marker to [ACTIVE] with date, so parallel chats
   do not collide.
3. Task stalls on something outside the session: doc moves to
   on_hold/, board line says what it is waiting on, links updated.
4. Task is closed: move the line to history.md with the completion
   date and a link to the record. The task doc moves to
   `archive/<section>/` (created on first closure) if the work stands,
   or to a dead_ends record if the idea itself was the mistake. Delete
   the task doc only when the archive record fully supersedes it.
5. Tasks that chain: the new task links to the closed one in
   history.md or archive -- no prose recaps.

Dropped tasks move to history.md as [DROPPED] with one line on why.

Session/validation notes written into gitignored locations are NOT a
task store: any [PENDING] or open caveat recorded there must get a
board line (or a tasks/ file) before the session ends, or it silently
falls out of the queue.
