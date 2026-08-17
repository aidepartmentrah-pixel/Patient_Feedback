# PEMS Administrator Handbook — Part B: Functionality per Tab

*Patient Experience Management System (PEMS) — for the Complaint Supervisor / Administrator role*

> This document assumes you've read **Part A: Concepts** — each entry below links back to the concept section that explains the *why* behind what you're seeing. This part is a reference: come back to the specific tab you're on, not a cover-to-cover read.
>
> Passages marked **[VERIFY]** haven't been independently confirmed on the live screen and should be checked before this is handed to real users.

---

## Daily Operations

### Dashboard
Your landing page. Shows real-time counts (open / in progress / closed / force-closed), breakdowns by Domain/Category/Severity, Red Flag and Never Event counts, and a scope filter to switch between Hospital-wide, a specific Administration, Department, or Section view.
*Related: §1 Org hierarchy, §3 Classification, §6 Red Flag/Never Event.*

### Notifications (Inbox)
Your central work queue. A single chronological list mixing every message type — Complaints, Notices, Seasonal Report notices, and Decision notifications — rather than separate tabs per type. As Complaint Supervisor you see everything across the whole hospital, not just one Section/Department/Administration.
- Filter chips let you narrow by type (All / Complaints / Notices / Seasonal) without leaving the page.
- Two real tabs: **Notifications** (active) and **Archive**.
- Each item shows action buttons appropriate to its state — accept, reject/return for revision, approve, or acknowledge (for Notices).
*Related: §2 Incident/Case/Notice, §4 Workflow lifecycle.*

### Workflow Page
The core case-monitoring and intervention screen. This is where you act on a Case directly rather than just viewing it in the queue: accept/reject/approve at whatever level you're standing in for, and use **"Fill Data Instead of User"** to enter a response on behalf of a Section/Department/Administration user who hasn't acted (this opens the dedicated Manual Fill screen — see below). Cases that are late or force-closed show a distinct visual marker here.
*Related: §4 Workflow lifecycle, §5 Force Close.*

### Table View
The full, filterable, exportable grid of every Complaint and Notice in the system.
- Filter by Domain, Category, Sub-Category, Severity, Harm, target org unit, and date range.
- Save and reuse custom column views. As Complaint Supervisor you can create/edit/delete saved views; other roles can use them but not modify them.
- Export respects whatever filters are currently active — it does not silently export everything.
*Related: §3 Classification.*

### Insert Record
Where a new Incident is created. You enter the shared Incident data (patient, doctor/worker, feedback details) once, then add one or more Case tabs — one per department the feedback concerns — each carrying its own classification if it's a Complaint, or none if it's a Notice.
- Patient search supports "create inline" if the person isn't found; doctor/worker search pulls from both the hospital's HIS/HR system and the local reserve list (§9 of Part A).
- A new record starts as **Draft**, becomes **Ready to Send** once complete, and only enters anyone's Inbox once you explicitly **Publish** it.
*Related: §2 Incident/Case/Notice, §3 Classification, §4 Workflow lifecycle, §9 Doctor/Employee linkage.*

### Edit Record
Same form as Insert Record, opened against an existing Incident — edits the whole Incident and all its sibling Cases together, not just one Case in isolation.

### Calendar (formerly "Follow Up")
A calendar view of **Action Items** — the preventive/corrective actions logged as part of an RCA (§8 of Part A) — organized by due date. Items are grouped as Overdue, Active, Upcoming, or Completed so you can see at a glance what's falling behind. Clicking an item shows the Case it belongs to.
*Related: §8 RCA.*

### Manual Fill
Reached from the Workflow Page's "Fill Data Instead of User" action. This is where you (or a Worker) actually type the response on behalf of whichever level hasn't answered — the system records that the entry was made on someone else's behalf rather than by the normal owner, so it stays traceable later.
*Related: §4 Workflow lifecycle, §5 Force Close.*

### Drawer Notes
A shared scratchpad, **not part of the official complaint record**. A few things worth knowing before you rely on it:
- It's **global**, not scoped to any department or section — every note is visible to everyone with access.
- It's **shared, not personal** — any Complaint Supervisor or Worker can edit or delete *any* note, not just their own. There's no "my notes" concept.
- Notes can optionally be linked to a patient and tagged with labels, and can be exported.
- Only Complaint Supervisor and Worker accounts can open this tab — Section/Department/Administration Admins cannot.

Treat it as a shared team whiteboard, not a private notebook and not a substitute for logging something in the actual Case.

---

## Analysis & Reporting

### Investigation
Hierarchical tree visualizations — Incident Count, Domain breakdown, Severity, Red Flag, and Never Event trees — scoped to Hospital/Administration/Department/Section, filterable by season. Node size and color encode volume and compliance so you can spot problem areas without reading a table.
*Related: §6 Red Flag/Never Event.*

### Target Analysis (formerly "Trend Monitoring")
Time-series charts of Domain/Category/Stage counts over month, season, or year, with your configured policy thresholds (§7 of Part A) overlaid as reference lines so you can see compliance visually rather than having to compare numbers by hand. Supports multi-quarter and multi-year comparisons.
*Related: §7 Seasonal reporting & policy thresholds.*

### Reporting
Generates the Monthly Report, the Seasonal Report (plus 2/3/4-quarter Seasonal Comparison), and a Section Workflow Activity Report (a per-section rundown of cases, RCA, and action items, meant for management meetings). Exports are scoped — Hospital, one or more Administrations, Departments, or Sections — and come out as zipped Word/PDF/CSV.
*Related: §7 Seasonal reporting.*

### History
Look up everything tied to a specific Doctor, Worker, or Patient — every Complaint and Notice associated with that person, in one place.

### Critical Issues
The dedicated view for escalated Cases — this is where Red Flag and Never Event Cases actually live in the UI, together in one page rather than split across separate screens. Confirmed working and showing real data.
*Related: §6 Red Flag/Never Event.*

---

## Configuration (Settings)

Everything in this section is visible to you as Complaint Supervisor and to no one else except a Worker account's much narrower Doctors/Patients view (§1 of Part A).

### Departments & Sections
- **Rename or reassign** an existing Section to a different Department — this works normally and doesn't affect historical records.
- **Create a brand-new Section** — **[KNOWN ISSUE]** the button for this is currently gated by a hardcoded check for the old SOFTWARE_ADMIN role (`primaryRole === 'SOFTWARE_ADMIN'` in the code), which no account holds anymore. As it stands, **no one can currently create a new Section through the UI** — this needs a code fix (change the check to allow Complaint Supervisor) before it can be documented as usable. Don't advertise this capability to users until it's fixed.
- Department-level create/update/delete works normally for you.

### Users
Create accounts, assign roles and org-unit scope, reset passwords, and update or deactivate existing users.
*Related: §1 Roles & org hierarchy.*

### Force Close Policy
Turn the automatic force-close policy on or off, and set the three deadline values (in days) for Section, Department, and Administration.
*Related: §5 Force Close.*

### Classification Management
Add, rename, or freeze classification values within existing Sub-Categories. Existing classifications are never deleted outright (freezing preserves historical reporting integrity), only retired from future use.
*Related: §3 Classification taxonomy.*

### Data Import
Bulk-load complaints/notices from an Excel file.
- Download the template, upload a `.xlsx` file, preview/confirm the batch before it's committed.
- Doctor/worker matching during import uses fuzzy name matching against the same merged HIS/HR + reserve-list source used elsewhere (§9 of Part A) — review suggested matches rather than assuming they're always exact.
- If one row in an Incident group fails validation, the whole group is rejected together rather than partially imported.
- View past import batches and their outcomes under batch history.

### Report Config
Configure the header/branding content that appears on generated reports.

### RCA Suggestions
Manage the list of cause categories, sub-causes, and preventive-action types available when someone fills out an RCA. **[VERIFY]** — confirm whether this list is now fully editable here or still partly fixed in the code, before describing it as complete self-service configuration.
*Related: §8 RCA.*

### Policy Configuration
**[VERIFY]** — this tab exists and is visible to you, but its exact contents (likely the organizational severity/domain policy thresholds referenced in §7 of Part A) haven't been independently confirmed. Check the live screen and fill this section in before publishing.

### Training
**[VERIFY]** — this tab exists and is visible to you; contents not yet confirmed. Check the live screen before publishing.

---

## Not Covered

A few things exist in the system but are deliberately left out of this handbook:

- **Data Migration** (`/migration`) — tooling for bringing legacy pre-PEMS cases into the current schema. This is a one-time/occasional technical task, not routine admin work; document separately if it's still actively used.
- **Config page** (`/config`) — a separate, password-protected bootstrap/installation screen (database and network setup), not part of day-to-day use.
- Two pages that were part of the original design but are **no longer reachable in the app**: Department Feedback and standalone Seasonal Reports — both were folded into the Reporting page and their routes are commented out in the code. If either name comes up in old materials or from other staff, it now lives under Reporting.

---

*End of Part B.*
