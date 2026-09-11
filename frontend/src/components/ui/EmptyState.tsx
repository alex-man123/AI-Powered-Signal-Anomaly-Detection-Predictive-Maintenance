import { Inbox } from "lucide-react"
import { cn } from "cn"

// TASK 11.9 — the one shared "request succeeded, there's just nothing to
// show" shell. Deliberately distinct from ErrorState (this is not a failure)
// and from LoadingState (data has actually arrived) — never used to paper
// over a request that hasn't resolved yet or one that failed.
//
// `size="md"` (default) is a page-level empty result (e.g. zero models, zero
// experiments) and gets a bit more visual weight (icon + title); `size="sm"`
// is a compact inline empty note inside an already-rendered section (e.g. one
// tab with nothing to show) and stays a single muted line, matching this
// app's existing inline-note convention.
export interface EmptyStateProps {
  title?: string
  message: string
  size?: "sm" | "md"
  className?: string
}

function EmptyState({ title, message, size = "md", className }: EmptyStateProps) {
  if (size === "sm") {
    return <p className={cn("text-sm text-muted-foreground", className)}>{message}</p>
  }

  return (
    <div
      className={cn(
        "flex flex-col items-center gap-2 rounded-lg border border-dashed border-border px-6 py-10 text-center",
        className,
      )}
    >
      <Inbox className="size-6 text-muted-foreground" />
      {title && <p className="text-sm font-medium">{title}</p>}
      <p className="text-sm text-muted-foreground">{message}</p>
    </div>
  )
}

export { EmptyState }
