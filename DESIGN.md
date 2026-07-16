# Design

## Direction

The product is a light civil-operations atlas, not a consumer dashboard or landing page. The first screen is the recovery desk: bounded inputs at left and computed evidence at right. Its signature interaction is the daily allocation ledger, where a day scrubber aligns the shock, five service end states, both allocations, and projection distance.

## Tokens And Type

- Canvas `#f3f6f4`, paper `#ffffff`, operational ink `#162521`
- Recovery teal `#14706b`, shock coral `#ca5b43`, infrastructure ochre `#b98226`, comparison blue `#4d7197`
- Bahnschrift/Arial Narrow for compact operational headings; Aptos/Segoe UI for body and controls; Consolas for seeds, hashes, and service codes
- Cards are limited to true blocking status; page sections remain bands or unframed work surfaces with radii at 5px or below

## Information Hierarchy

1. Brand and local/synthetic runtime status
2. Editable scenario envelope with explicit numeric bounds and deterministic saved-result restore
3. Primary resilience comparison and measured constraint count
4. Real trajectory chart or full daily audit table
5. Selected-day allocation ledger
6. Shock-tape and ONNX policy checksums with synthetic PPO and legacy non-PPO disclosures

## Responsive Contract

- At 1440x900 the editor and comparison share one fixed-height operations desk.
- At 1280x720 the summary wraps its facts and the result region scrolls without overlap.
- Below 1100px the chart and ledger stack.
- At 390x844 the editor and results become one document flow; compact service codes preserve input and ledger widths.

All interactive targets own an actual DOM box of at least 40 by 40 px, including compact service inputs, the custom checkbox target, tabs, and the day range. Focus is visible, dynamic numbers are tabular, chart paths do not animate, and reduced-motion disables the spinner animation. Loading, empty, invalid/dependency error, and recompute states have distinct UI. A dependency failure changes the runtime indicator to a visible `Policy blocked` label even in the compact mobile header.

The constraint proof is never a literal success claim. It sums each planner's returned daily `projection.constraint_violations`, labels candidate and baseline totals independently, and remains visible below 720px. The comparison names the actual SB3 PPO / ONNX candidate and visible OR-Tools GLOP baseline. Recovery days appear beside resilience rather than being implied by the chart. Invalid, persistence, computation, or dependency errors remove the entire prior result region before their blocking state appears.

The saved-results menu is a real restore control backed by canonical local JSON. It is disabled when no result exists, lists deterministic result identities without timestamp claims, and restores the complete authored scenario and evidence together.

## Asset Audit

No raster, generated, external, map, or evidentiary image is needed. The only visual marks are CSS structure, Lucide interface icons, and an SVG chart rendered from the API trajectory. This avoids implying geographic or empirical evidence that the synthetic simulator does not have.
