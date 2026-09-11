import { Button } from '@/components/ui/button'
import { useActiveSampleClass } from '@/context/ActiveSampleClassContext'
import { SIGNAL_LABEL_LABELS } from '@/lib/model-display'
import { cn } from '@/lib/utils'
import type { SignalLabel } from '@/services/api'

const REAL_SIGNAL_LABELS: SignalLabel[] = ['normal', 'imbalance', 'horizontal-misalignment', 'vertical-misalignment']

// Shared control rendered identically on Dashboard, Signal Analysis, DSP Lab,
// and Anomaly Detection — all four previously always used the same
// hardcoded real `normal` sample signal (`GET /api/models/sample-signal`
// with no `label`). Picking a class here updates `ActiveSampleClassContext`
// (sessionStorage-backed, same convention as the Models page's active-model
// selection), so the other three pages show that same real class's sample
// the next time they fetch it.
export function SampleClassSelector({ className }: { className?: string }) {
  const { activeSampleClass, setActiveSampleClass } = useActiveSampleClass()
  const current = activeSampleClass ?? 'normal'

  return (
    <div className={cn('flex flex-wrap items-center gap-1.5', className)}>
      <span className="text-xs text-muted-foreground">Sample signal:</span>
      {REAL_SIGNAL_LABELS.map((label) => (
        <Button
          key={label}
          type="button"
          size="sm"
          variant={current === label ? 'default' : 'outline'}
          onClick={() => setActiveSampleClass(label)}
        >
          {SIGNAL_LABEL_LABELS[label]}
        </Button>
      ))}
    </div>
  )
}
