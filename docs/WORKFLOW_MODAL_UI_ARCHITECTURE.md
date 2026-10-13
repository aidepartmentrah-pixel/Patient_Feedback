# The Case Review Modal — Expandable UI Architecture Reference

**Purpose of this document**: exactly how the "Case Review" modal (the popup in the screenshots — رقم الحالة / نوع الحالة / قسم المعني header, collapsible "تفاصيل الشكوى", and the RCA-accordion + dynamic-action-items form that appears after clicking "تعليق إضافي على الرد") is built — which library provides the expand/collapse behavior, which parts are custom, and how the "simple view → full form" transition works. Every claim is backed by a specific file in the frontend repo (`Front_End_Feedback_Analysis`), read directly and verified against the live component tree, not paraphrased from memory.

**The short answer to "what technology makes it expandable and nice"**: it's almost entirely **MUI Joy UI**'s own `Accordion`/`AccordionGroup`/`AccordionDetails` components, driven by plain React `useState`, styled with Joy's `sx` prop plus one shared JS design-token file (`src/theme/theme.js`). There is no animation library, no headless-UI package, and only one small hand-rolled toggle (a chevron-rotating "show description" button) that isn't a real accordion at all.

---

## 1. Component tree

The modal is not one big file — it's a "region scaffold" of small, single-purpose components, all in `Front_End_Feedback_Analysis/src/components/workflow/`:

```
CaseReviewModal.jsx            ← owns ALL state + handlers, wires everything together
 └─ WorkflowFormShell.jsx      ← <Modal>/<ModalDialog> wrapper, sized by theme.modal.sizes
     └─ ModalLayoutShell.jsx   ← fixed header / scrolling 1-or-2-column body / sticky footer
         ├─ ContextRegion.jsx          ← title bar (رقم الحالة / نوع الحالة / قسم المعني / حالة المعالجة)
         ├─ MainContentRegion.jsx      ← hosts ComplaintDetailsSection + the active form field
         │   └─ ComplaintDetailsSection.jsx   ← "تفاصيل الشكوى" — collapsible Accordion
         ├─ SupportRegion.jsx          ← only rendered once an action is selected
         │   ├─ RcaPairsPicker.jsx           ← "تحليل السبب الجذري (RCA)" — AccordionGroup
         │   └─ ActionItemsEditor.jsx        ← "بنود الإجراءات" — dynamic list + item add/remove
         └─ ActionFooterRegion.jsx
             └─ WorkflowActionButtons.jsx    ← تعليق إضافي على الرد / قبول الرد / تأكيد / إلغاء
```

A second, older implementation (`CaseActionModal.jsx`) duplicates a lot of this logic in one monolithic file but is **dead code** — confirmed not imported anywhere in the app. `CaseReviewModal.jsx` is the one actually rendering what's in the screenshots.

---

## 2. The library stack — nothing exotic

| Concern | Library / mechanism |
|---|---|
| Modal/dialog | `@mui/joy`'s `Modal` + `ModalDialog` + `ModalClose` |
| Accordions (both the "تفاصيل الشكوى" section and the RCA categories) | `@mui/joy`'s `Accordion` / `AccordionSummary` / `AccordionDetails` / `AccordionGroup` — a real, built-in Joy component, not custom |
| Expand/collapse chevron icon | `@mui/icons-material`'s `ExpandMoreIcon`, passed as the `indicator` prop on `AccordionSummary` (Joy rotates it itself when `expanded` changes) |
| Styling | Joy's `sx` prop everywhere, plus one shared JS theme object (`src/theme/theme.js`, imported as `theme.colors.*`, `theme.radius.*`, `theme.spacing.*`, `theme.modal.sizes`, `theme.typography.cardTitle`). **No CSS modules, no Tailwind, no styled-components, no CSS-in-JS library beyond what Joy already ships with.** |
| State | Plain React `useState`/`useCallback`/`useEffect` — **no Redux, no react-hook-form, no Formik, no useFieldArray** anywhere in this modal |
| Layout | CSS flexbox via `sx` (`display: 'flex'`, `flexDirection`, `flex: '0 0 auto'` / `'1 1 auto'`) — no CSS Grid except for the label/value grid inside the complaint-details card |
| RTL | A single `direction: 'rtl'` + `textAlign: 'right'` set once on `ModalDialog` (`WorkflowFormShell.jsx`) — everything inside inherits it; no per-component RTL handling needed |

---

## 3. The two different "expand/collapse" mechanisms (they are not the same thing)

### 3.1 Real accordions — Joy's `Accordion`, controlled and uncontrolled

**"تفاصيل الشكوى" (Complaint Details)** — a single **controlled** accordion. The open/closed boolean is lifted up to the parent (`CaseReviewModal`'s `complaintOpen` state) rather than living inside the accordion itself, because the parent needs to force it open by default when the modal opens:

```jsx
// ComplaintDetailsSection.jsx
<Accordion
  expanded={open}          // controlled — boolean comes from the parent
  onChange={onChange}      // parent flips its own state
  sx={{ border: `1px solid ${theme.colors.border}`, borderRight: `3px solid ${theme.colors.primary}`, borderRadius: theme.radius.lg, mb: 2 }}
>
  <AccordionSummary indicator={<ExpandMoreIcon />}>
    <Typography sx={theme.typography.cardTitle}>{sectionTitle}</Typography>
    {!open && <Typography level="body-xs" sx={{ color: 'neutral.500', mr: 1 }}>(انقر للتوسيع)</Typography>}
  </AccordionSummary>
  <AccordionDetails sx={{ px: 2, py: 1.5 }}>
    {/* the label/value grid + complaint text card */}
  </AccordionDetails>
</Accordion>
```
```jsx
// CaseReviewModal.jsx — the parent side
const [complaintOpen, setComplaintOpen] = useState(false);
useEffect(() => { if (open) setComplaintOpen(true); /* ...reset on every new item */ }, [open, subcaseId]);
// passed down as: complaintOpen={complaintOpen}  onToggleComplaint={() => setComplaintOpen(p => !p)}
```
A small nice touch: when collapsed, it shows a literal "(انقر للتوسيع)" ("click to expand") hint next to the title — cheap, but removes any doubt that the header is clickable.

**"تحليل السبب الجذري (RCA)"** — a group of **uncontrolled** accordions, one per category, wrapped in Joy's `AccordionGroup`. Each `<Accordion>` here manages its own open/closed state internally (no `expanded`/`onChange` props at all) since nothing outside needs to know or react to which category is open:

```jsx
// RcaPairsPicker.jsx
<AccordionGroup>
  {categories.map(cat => (
    <Accordion key={cat.category_id}>
      <AccordionSummary>
        <Typography level="body-sm">{cat.category_name_ar || cat.category_name_en}</Typography>
      </AccordionSummary>
      <AccordionDetails>
        {cat.pairs.map(p => (/* checkbox row per cause */))}
      </AccordionDetails>
    </Accordion>
  ))}
</AccordionGroup>
```
**Important**: the category names (التواصل / التوظيف والتدريب / السياسات والإجراءات / المعدات والتقنية / البيئة والمرافق / العوامل البشرية, visible in the screenshot) are **not hardcoded JSX strings** — they come from the backend via `getRcaPairsForSubcase(subcaseId)` (`src/api/rcaApi.js`), so an admin can add/edit RCA categories and cause/action pairs without a frontend deploy.

### 3.2 The one genuinely custom toggle — not a real accordion

Inside each action-item card, the "وصف" (description) button that reveals a `Textarea` is **not** Joy's `Accordion` — it's a plain boolean-per-item toggle, because you can't reuse one shared `Accordion` component for an unbounded, dynamically-added list of independent rows without re-deriving a key per row anyway, and a full accordion's chrome (border, summary/details separation) wasn't wanted here — just a chevron and a field appearing:

```jsx
// ActionItemsEditor.jsx
const [expandedKeys, setExpandedKeys] = useState(new Set());   // which item-keys are manually expanded
const itemKey = (ai, idx) => ai.action_item_id ?? `new-${idx}`;
const toggleExpanded = (key) => setExpandedKeys(prev => {
  const next = new Set(prev);
  next.has(key) ? next.delete(key) : next.add(key);
  return next;
});
// per item:
const descriptionOpen = expandedKeys.has(key) || !!ai.description?.trim();  // auto-open if it already has text
...
<Button variant="plain" onClick={() => toggleExpanded(key)} endDecorator={
  <ExpandMoreIcon sx={{ transform: descriptionOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.15s' }} />
}>وصف</Button>
{descriptionOpen && <Textarea ... />}   // plain conditional render, no collapse animation
```
Two small design choices worth copying: (1) a `Set<key>` rather than an array or per-item boolean field on the data object — keeps "is this expanded" as pure UI state, never mixed into the saved payload; (2) `descriptionOpen` auto-becomes `true` if the row already has description text, so a pre-filled item (e.g. from editing an existing response) never hides data behind a collapsed toggle the user has to know to click.

---

## 4. How "تعليق إضافي على الرد" transitions the modal into the big form

There's **no second modal, no route change, no animation library** — it's one `activeAction` string state (`null` initially) and everything downstream branches on it:

```jsx
// CaseReviewModal.jsx
const [activeAction, setActiveAction] = useState(null);

const selectAction = (code) => {
  const isToggleOff = activeAction === code;          // clicking the same button again closes the form
  setActiveAction(isToggleOff ? null : code);
  // ...pre-populates actionItems from the existing response when opening SUBMIT_RESPONSE/OVERRIDE
};
```
`WorkflowActionButtons` (the footer buttons) just calls `onSelect('SUBMIT_RESPONSE')` on click — the button's own `variant` prop flips between `'solid'`/`'outlined'` based on whether it's the active one, giving the pressed-button visual feedback for free.

The layout reacts to `activeAction` in two places:
1. **`MainContentRegion`** renders the matching form field (explanation textarea for `SUBMIT_RESPONSE`/`OVERRIDE`, rejection textarea for `REJECT`/`REOPEN`, nothing for instant actions like `APPROVE`).
2. **`hasSupportContent`**, computed in `CaseReviewModal`, decides whether `SupportRegion` (RCA + action items) renders **at all**:
   ```jsx
   const showActionItemsSupport = !isPatientServicesReview && (activeAction === 'SUBMIT_RESPONSE' || activeAction === 'OVERRIDE');
   const showRcaEditableSupport = !isPatientServicesReview && (activeAction === 'SUBMIT_RESPONSE' || (activeAction === 'OVERRIDE' && isRcaOwner));
   const hasSupportContent = showActionItemsSupport || showRcaSupport;
   ```
   `ModalLayoutShell` receives `supportContent={hasSupportContent ? <SupportRegion .../> : null}` — when it's `null`, the layout's own flex logic collapses from a two-column (60/40 split, `theme.modal.primaryColumnBasis`/`supportColumnBasis`) layout back to a single full-width column, and the whole `WorkflowFormShell` even requests a smaller modal size (`size={hasSupportContent ? 'lg' : 'md'}`) — so clicking "تعليق إضافي على الرد" doesn't just reveal content inside a fixed-size box, it **grows the modal itself**, which is what reads as the UI becoming visibly "bigger/nicer" in the second screenshot.

This is the core reusable pattern: **one string (or enum) state variable, everything else is derived booleans computed from it, and every region is a pure function of those booleans.** No component owns its own "is the form open" state independently — that would risk them disagreeing.

---

## 5. The dynamic "بند" (action item) rows

Plain array-of-objects `useState`, no form-array library:

```jsx
// CaseReviewModal.jsx
const [actionItems, setActionItems] = useState([
  { action_item_id: null, title: '', description: '', due_date: '' }
]);

const addActionItem    = () => setActionItems(prev => [...prev, { action_item_id: null, title: '', description: '', due_date: '' }]);
const removeActionItem = (i) => setActionItems(prev => prev.filter((_, idx) => idx !== i));
const updateActionItem = (i, field, val) =>
  setActionItems(prev => prev.map((ai, idx) => idx === i ? { ...ai, [field]: val } : ai));
```
`ActionItemsEditor` is purely presentational — it receives `actionItems` + the three callbacks as props and never holds the data itself (only the description-expanded `Set` from §3.2, which is display-only and never sent to the backend). Each rendered row is keyed by `action_item_id ?? "new-${idx}"` so an item loaded from an existing response (has a real id) and a freshly-added blank one (has none) both get a stable React key.

A nice detail worth reusing: **suggested action items derived from RCA selections, not separately tracked.** When a user checks an RCA cause that has a paired suggested action, `CaseReviewModal` computes:
```jsx
const acceptedActionTitles = new Set(actionItems.map(ai => ai.title.trim()).filter(Boolean));
const suggestedActions = rcaCategories.flatMap(cat => cat.pairs || [])
  .filter(p => selectedIds.has(p.pair_id) && p.action_text_ar?.trim() && !acceptedActionTitles.has(p.action_text_ar.trim()));
```
There's no separate "accepted" flag — a suggestion simply stops appearing once an action item's title matches its text exactly, because "accepting" it (`acceptSuggestedAction`) just calls the same `setActionItems` used for manual entry. One state, two entry points, zero risk of the two lists disagreeing.

---

## 6. The RCA ↔ explanation-text link (a UX detail worth knowing before copying this pattern)

Checking an RCA cause checkbox doesn't just toggle a `Set` — it also **auto-composes a sentence and appends it into the shared "التوضيح / الشرح" textarea** (`composeRcaSentence()` + `toggleRcaId()` in `CaseReviewModal.jsx`), tracked via a `Map<pairId, generatedSentence>` so unchecking later can cleanly remove exactly that generated sentence **only if the user hasn't since edited it**. An earlier version showed the composed RCA text in its own separate preview box next to the checkboxes — removed after live testing showed editing either box confusingly updated the other. **Lesson**: when auto-generating text into a field a human also edits directly, track what you inserted (and only remove that verbatim), don't try to maintain two synced copies of the same content.

---

## 7. What's reused elsewhere vs. one-off

- `ComplaintDetailsSection` and the whole `WorkflowFormShell`/`ModalLayoutShell` scaffold are **shared** across the other workflow modals (Notice, Patient Services decision, etc.) — e.g. other callers override `sectionTitle`/`emptyText` props rather than forking the component.
- The `AccordionGroup`-of-`Accordion` pattern for RCA categories is the same shape used in both the live `RcaPairsPicker` and the dead `CaseActionModal`'s equivalent — i.e. it was already the established pattern before the refactor, not invented new.
- The description chevron-toggle in `ActionItemsEditor` (§3.2) is a one-off, not extracted into a shared component — fine at its current single call site, but worth promoting to a small shared `ExpandableField` if a third place ever needs the same "toggle to reveal a text field" behavior.

---

## 8. Recipe — recreating this pattern in a new project

1. Pull in `@mui/joy` (and `@mui/icons-material` for icons) — that alone gives you `Modal`/`ModalDialog`, `Accordion`/`AccordionGroup`, and every form control used here, with zero extra dependencies.
2. Define a **shared JS design-token object** (colors/spacing/radius/typography/modal-sizes) once, imported everywhere via `sx` — not a full styled-system, just a plain object literal (`theme.js`), which is enough to keep every modal/card/accordion visually consistent without a CSS build step.
3. Split a complex modal into **region components** (header/main/support/footer) that each take plain props and render nothing but JSX + `sx` — keeps one parent component (`CaseReviewModal`) as the single owner of all state, and every child dumb/presentational.
4. Model "which step/action is active" as **one nullable string/enum state**, and derive every other boolean (`showX`, `hasY`) from it — never let two components independently decide whether a section is visible.
5. For a fixed set of named collapsible sections, use the accordion component's own controlled `expanded`/`onChange` only when a parent genuinely needs to force it open/closed; otherwise let each accordion instance manage itself uncontrolled — don't lift state you don't need to.
6. For a dynamically-growable list of rows (add/remove/edit), a plain array of plain objects in `useState` plus three small callbacks (`add`/`remove`/`update`) is sufficient — a form-array library is unnecessary overhead unless you also need field-level validation/dirty-tracking at scale.
7. If auto-inserting generated text into a field the user can also hand-edit, track exactly what you inserted (id → generated string) so you can safely undo only your own contribution later — never assume the field's current content is still exactly what you put there.

---

## 9. File map

| Concern | File |
|---|---|
| Owns all state/handlers, top-level orchestration | `src/components/workflow/CaseReviewModal.jsx` |
| Modal/dialog wrapper, sizing, RTL | `src/components/workflow/WorkflowFormShell.jsx` |
| Header/body/footer region scaffold | `src/components/workflow/ModalLayoutShell.jsx` |
| Title bar (case #, type, department, status) | `src/components/workflow/ContextRegion.jsx` |
| Collapsible complaint details | `src/components/workflow/ComplaintDetailsSection.jsx` |
| RCA accordion picker | `src/components/workflow/RcaPairsPicker.jsx` |
| Dynamic action-items list | `src/components/workflow/ActionItemsEditor.jsx` |
| Action-selector buttons + footer | `src/components/workflow/WorkflowActionButtons.jsx`, `ActionFooterRegion.jsx` |
| Shared design tokens | `src/theme/theme.js` (see `modal`, `spacing`, `radius`, `typography` keys) |
| RCA data source | `src/api/rcaApi.js` (`getRcaPairsForSubcase`, `saveRcaSelections`) |
