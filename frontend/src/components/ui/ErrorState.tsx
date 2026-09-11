import { CircleAlert } from "lucide-react"
import { cn } from "cn"

import { Button } from "@/components/ui/button"

// TASK 11.9 — the one shared "request failed" shell. Never shows a stack
// trace or internal detail — `message` is always the already-user-safe
// string the caller's API layer produced (see `errorMessageFor` in
// services/api.ts). `onRetry` is optional: a sub-section error (one tab, one
// table cell) usually isn't a good place for a retry action, so callers only
// pass it where re-fetching the whole page/section actually makes sense.
export interface ErrorStateProps {
  message: string
  onRetry?: () => void
  retryLabel?: string
  size?: "sm" | "md"
  className?: string
}

const SIZE_CLASSES: Record<NonNullable<ErrorStateProps["size"]>, { icon: string; text: string }> = {
  sm: { icon: "size-3.5", text: "text-xs" },
  md: { icon: "size-4", text: "text-sm" },
}

function ErrorState({ message, onRetry, retryLabel = "Retry", size = "md", className }: ErrorStateProps) {
  const sizes = SIZE_CLASSES[size]

  return (
    <div className={cn("flex animate-in flex-col items-start gap-3 fade-in duration-200", className)}>
      <p className={cn("flex items-center gap-2 text-anomaly", sizes.text)}>
        <CircleAlert className={cn("shrink-0", sizes.icon)} />
        {message}
      </p>
      {onRetry && (
        <Button type="button" variant="outline" size="sm" onClick={onRetry}>
          {retryLabel}
        </Button>
      )}
    </div>
  )
}

export { ErrorState }
