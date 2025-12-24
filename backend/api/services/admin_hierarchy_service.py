from backend.api.db_layer.admin_units import get_admin_unit_children
from backend.api.db_layer.admin_units import get_active_admin_units



def get_next_level_options(parent_id):
    """
    Returns child administration units suitable for UI selection.
    """
    children = get_admin_unit_children(parent_id)
    return [
        {
            "id": c.UniqueID,
            "name": c.Name,
            "is_leaf": False  # can be extended later
        }
        for c in children
        if c.Frozen == 0
    ]


def get_dashboard_hierarchy():
    """
    Build hierarchy for Dashboard cascading selectors:
    Administration -> Department -> Section
    """

    units = get_active_admin_units()


    Administration = []
    Department = {}
    Section = {}

    # -----------------------------
    # Helper: does unit have children
    # -----------------------------
    def has_children(unit_id):
        return any(u.ParentID == unit_id for u in units)

    # -----------------------------
    # Step 1: Administrations (root)
    # Rule: ParentID == UniqueID
    # -----------------------------
    for u in units:
        if u.ParentID == u.UniqueID:
            Administration.append({
                "id": u.UniqueID,
                "nameAr": u.Name,
                "nameEn": u.Name
            })

    # -----------------------------
    # Step 2: Departments
    # -----------------------------
    for admin in Administration:
        admin_id = admin["id"]
        Department[admin_id] = []

        for u in units:
            if u.ParentID == admin_id and u.UniqueID != admin_id:
                Department[admin_id].append({
                    "id": u.UniqueID,
                    "nameAr": u.Name,
                    "nameEn": u.Name
                })

    # -----------------------------
    # Step 3: Sections
    # -----------------------------
    for dept_list in Department.values():
        for dept in dept_list:
            dept_id = dept["id"]
            Section[dept_id] = []

            for u in units:
                if u.ParentID == dept_id:
                    Section[dept_id].append({
                        "id": u.UniqueID,
                        "nameAr": u.Name,
                        "nameEn": u.Name
                    })

    return {
        "Administration": Administration,
        "Department": Department,
        "Section": Section
    }
