from backend.api.services.admin_hierarchy_service import get_dashboard_hierarchy

data = get_dashboard_hierarchy()

print("Administrations:", len(data["Administration"]))
print("Departments keys:", data["Department"].keys())
print("Sections keys:", data["Section"].keys())
