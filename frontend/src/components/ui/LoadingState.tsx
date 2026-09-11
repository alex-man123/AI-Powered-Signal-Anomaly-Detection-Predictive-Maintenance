import { Loader } from "lucide-react"
import { cn } from "cn"

// TASK 11.9 — the one shared "request is pending" shell every page/section
// that fetches data uses, instead of each re-implementing the same
// icon+text row. `size="sm"` matches the compact inline usage (a section
// inside a Card, e.g. a single tab or table cell); `size="md"` (default)
// matches a page-level loading line.
export interface LoadingStateProps {
  message?: string
  size?: "sm" | "md"
  className?: string
}

const SIZE_CLASSES: Record<NonNullable<LoadingStateProps["size"]>, { icon: string; text: string }> = {
  sm: { icon: "size-3.5", text: "text-xs" },
  md: { icon: "size-4", text: "text-sm" },
}

function LoadingState({ message = "Loading…", size = "md", className }: LoadingStateProps) {
  const sizes = SIZE_CLASSES[size]

  return (
    <p className={cn("flex items-center gap-2 text-muted-foreground", sizes.text, className)}>
      <Loader className={cn("animate-spin", sizes.icon)} />
      {message}
    </p>
  )
}

export { LoadingState }
