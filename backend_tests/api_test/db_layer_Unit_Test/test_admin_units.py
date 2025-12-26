from backend.api.db_layer.admin_units import (
    get_admin_unit_by_id,
    get_admin_unit_children,
    get_admin_unit_parent,
    get_admin_unit_tree,
    get_admin_unit_leaves,
    get_active_admin_units,
)


def test_admin_units():
    print("\n--- TEST: get_admin_unit_by_id ---")
    unit = get_admin_unit_by_id(1)
    print(unit)

    print("\n--- TEST: get_admin_unit_children ---")
    children = get_admin_unit_children(1)
    print(f"Children count: {len(children)}")
    for c in children[:5]:
        print(c)

    print("\n--- TEST: get_admin_unit_parent ---")
    parent = get_admin_unit_parent(14)
    print(parent)

    print("\n--- TEST: get_admin_unit_tree ---")
    all_units = get_admin_unit_tree()
    print(f"Total units: {len(all_units)}")

    print("\n--- TEST: get_admin_unit_leaves ---")
    leaves = get_admin_unit_leaves()
    print(f"Leaf units: {len(leaves)}")
    for l in leaves[:5]:
        print(l)

    print("\n--- TEST: get_active_admin_units ---")
    active_units = get_active_admin_units()
    print(f"Active units: {len(active_units)}")
    for a in active_units[:5]:
        print(a)


if __name__ == "__main__":
    test_admin_units()
