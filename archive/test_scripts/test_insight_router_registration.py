"""
Test B-I18: Insight Router Registration in main.py

Verifies:
1. Insight router is successfully imported and registered
2. Application starts without errors
3. Router is included in FastAPI app
4. All 4 insight endpoints are discoverable
5. OpenAPI schema includes /api/v2/insight/* paths
6. Router tags are correctly configured
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add backend directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from backend.main import app


@pytest.fixture
def client():
    """Create test client."""
    return TestClient(app)


def test_app_imports_successfully():
    """Test 1: Application imports without errors."""
    # If we got here, import succeeded
    assert app is not None
    assert app.title == "Incident Manager API"


def test_app_has_insight_router():
    """Test 2: Insight router is registered in app."""
    # Check that app has routers
    assert hasattr(app, 'routes')
    
    # Get all route paths
    route_paths = [route.path for route in app.routes]
    
    # Verify insight endpoints are in routes
    insight_paths = [
        '/api/v2/insight/kpi-summary',
        '/api/v2/insight/distribution',
        '/api/v2/insight/trend',
        '/api/v2/insight/stuck'
    ]
    
    for path in insight_paths:
        assert path in route_paths, f"Expected route {path} not found in app routes"


def test_kpi_summary_endpoint_exists(client):
    """Test 3: GET /api/v2/insight/kpi-summary endpoint is discoverable."""
    # Don't authenticate, just check endpoint exists (will get 401)
    response = client.get("/api/v2/insight/kpi-summary")
    
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401, \
        f"Expected 401 (unauthorized), got {response.status_code}. Endpoint may not be registered."


def test_distribution_endpoint_exists(client):
    """Test 4: POST /api/v2/insight/distribution endpoint is discoverable."""
    # Don't authenticate, just check endpoint exists (will get 401)
    response = client.post(
        "/api/v2/insight/distribution",
        json={"dimension": "Doctor"}
    )
    
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401, \
        f"Expected 401 (unauthorized), got {response.status_code}. Endpoint may not be registered."


def test_trend_endpoint_exists(client):
    """Test 5: POST /api/v2/insight/trend endpoint is discoverable."""
    # Don't authenticate, just check endpoint exists (will get 401)
    response = client.post(
        "/api/v2/insight/trend",
        json={"bucket": "day"}
    )
    
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401, \
        f"Expected 401 (unauthorized), got {response.status_code}. Endpoint may not be registered."


def test_stuck_endpoint_exists(client):
    """Test 6: GET /api/v2/insight/stuck endpoint is discoverable."""
    # Don't authenticate, just check endpoint exists (will get 401)
    response = client.get("/api/v2/insight/stuck")
    
    # Should return 401 (unauthorized) not 404 (not found)
    assert response.status_code == 401, \
        f"Expected 401 (unauthorized), got {response.status_code}. Endpoint may not be registered."


def test_openapi_includes_insight_endpoints():
    """Test 7: OpenAPI schema includes insight endpoints."""
    # Get OpenAPI schema
    openapi_schema = app.openapi()
    
    # Check paths section exists
    assert 'paths' in openapi_schema
    
    # Verify all insight endpoints are in OpenAPI schema
    insight_paths = [
        '/api/v2/insight/kpi-summary',
        '/api/v2/insight/distribution',
        '/api/v2/insight/trend',
        '/api/v2/insight/stuck'
    ]
    
    for path in insight_paths:
        assert path in openapi_schema['paths'], \
            f"Endpoint {path} not found in OpenAPI schema"


def test_insight_endpoints_have_correct_tags():
    """Test 8: Insight endpoints have correct tags in OpenAPI schema."""
    openapi_schema = app.openapi()
    
    # Check kpi-summary endpoint tags
    kpi_path = openapi_schema['paths']['/api/v2/insight/kpi-summary']
    assert 'get' in kpi_path
    assert 'tags' in kpi_path['get']
    assert 'api_v2_insight' in kpi_path['get']['tags']
    
    # Check distribution endpoint tags
    dist_path = openapi_schema['paths']['/api/v2/insight/distribution']
    assert 'post' in dist_path
    assert 'tags' in dist_path['post']
    assert 'api_v2_insight' in dist_path['post']['tags']
    
    # Check trend endpoint tags
    trend_path = openapi_schema['paths']['/api/v2/insight/trend']
    assert 'post' in trend_path
    assert 'tags' in trend_path['post']
    assert 'api_v2_insight' in trend_path['post']['tags']
    
    # Check stuck endpoint tags
    stuck_path = openapi_schema['paths']['/api/v2/insight/stuck']
    assert 'get' in stuck_path
    assert 'tags' in stuck_path['get']
    assert 'api_v2_insight' in stuck_path['get']['tags']


def test_insight_endpoints_require_authentication(client):
    """Test 9: All insight endpoints require authentication."""
    # Test runtime enforcement: endpoints should reject unauthenticated requests
    endpoints = [
        ('/api/v2/insight/kpi-summary', 'GET', None),
        ('/api/v2/insight/distribution', 'POST', {"dimension": "Doctor"}),
        ('/api/v2/insight/trend', 'POST', {"bucket": "day"}),
        ('/api/v2/insight/stuck', 'GET', None)
    ]
    
    for path, method, body in endpoints:
        if method == 'GET':
            response = client.get(path)
        else:
            response = client.post(path, json=body)
        
        # Should return 401 (unauthorized), not 200 or 404
        assert response.status_code == 401, \
            f"{path} {method} should require authentication (got {response.status_code})"


def test_router_prefix_is_correct():
    """Test 10: Router has correct prefix /api/v2/insight."""
    # Get all routes
    insight_routes = [
        route for route in app.routes 
        if hasattr(route, 'path') and '/api/v2/insight/' in route.path
    ]
    
    # Verify we have exactly 4 insight routes
    assert len(insight_routes) == 4, \
        f"Expected 4 insight routes, found {len(insight_routes)}"
    
    # Verify all routes start with /api/v2/insight
    for route in insight_routes:
        assert route.path.startswith('/api/v2/insight/'), \
            f"Route {route.path} does not start with /api/v2/insight/"


def test_router_registration_order():
    """Test 11: Insight router is registered after workflow router."""
    # Get all route paths in order
    route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
    
    # Find indices of workflow and insight routes
    workflow_indices = [
        i for i, path in enumerate(route_paths) 
        if '/api/v2/workflow/' in path
    ]
    
    insight_indices = [
        i for i, path in enumerate(route_paths) 
        if '/api/v2/insight/' in path
    ]
    
    # Verify both routers are registered
    assert len(workflow_indices) > 0, "Workflow router not found"
    assert len(insight_indices) > 0, "Insight router not found"
    
    # Verify insight routes come after workflow routes
    # (Registration order affects route ordering in FastAPI)
    first_workflow = min(workflow_indices)
    first_insight = min(insight_indices)
    
    assert first_insight > first_workflow, \
        "Insight router should be registered after workflow router"


def test_no_duplicate_routes():
    """Test 12: No duplicate insight routes registered."""
    # Get all route paths
    route_paths = [route.path for route in app.routes if hasattr(route, 'path')]
    
    # Filter insight routes
    insight_routes = [path for path in route_paths if '/api/v2/insight/' in path]
    
    # Check for duplicates
    unique_routes = set(insight_routes)
    assert len(insight_routes) == len(unique_routes), \
        f"Duplicate insight routes detected: {insight_routes}"


def test_kpi_summary_openapi_spec():
    """Test 13: KPI summary endpoint has correct OpenAPI specification."""
    openapi_schema = app.openapi()
    kpi_spec = openapi_schema['paths']['/api/v2/insight/kpi-summary']['get']
    
    # Check summary and description
    assert 'summary' in kpi_spec
    assert 'KPI Summary' in kpi_spec['summary'] or 'kpi' in kpi_spec['summary'].lower()
    
    # Check responses
    assert 'responses' in kpi_spec
    assert '200' in kpi_spec['responses']


def test_distribution_openapi_spec():
    """Test 14: Distribution endpoint has correct OpenAPI specification."""
    openapi_schema = app.openapi()
    dist_spec = openapi_schema['paths']['/api/v2/insight/distribution']['post']
    
    # Check it expects request body
    assert 'requestBody' in dist_spec
    
    # Check responses
    assert 'responses' in dist_spec
    assert '200' in dist_spec['responses']


def test_trend_openapi_spec():
    """Test 15: Trend endpoint has correct OpenAPI specification."""
    openapi_schema = app.openapi()
    trend_spec = openapi_schema['paths']['/api/v2/insight/trend']['post']
    
    # Check it expects request body
    assert 'requestBody' in trend_spec
    
    # Check responses
    assert 'responses' in trend_spec
    assert '200' in trend_spec['responses']


def test_stuck_openapi_spec():
    """Test 16: Stuck cases endpoint has correct OpenAPI specification."""
    openapi_schema = app.openapi()
    stuck_spec = openapi_schema['paths']['/api/v2/insight/stuck']['get']
    
    # Check summary
    assert 'summary' in stuck_spec
    
    # Check responses
    assert 'responses' in stuck_spec
    assert '200' in stuck_spec['responses']


def test_insight_router_methods():
    """Test 17: Insight endpoints have correct HTTP methods."""
    # Check each route has the correct method
    routes_by_path = {}
    for route in app.routes:
        if hasattr(route, 'path') and '/api/v2/insight/' in route.path:
            if route.path not in routes_by_path:
                routes_by_path[route.path] = []
            if hasattr(route, 'methods'):
                routes_by_path[route.path].extend(route.methods)
    
    # Verify methods
    assert 'GET' in routes_by_path.get('/api/v2/insight/kpi-summary', [])
    assert 'POST' in routes_by_path.get('/api/v2/insight/distribution', [])
    assert 'POST' in routes_by_path.get('/api/v2/insight/trend', [])
    assert 'GET' in routes_by_path.get('/api/v2/insight/stuck', [])


def test_main_py_has_insight_import():
    """Test 18: main.py contains insight router import."""
    main_path = os.path.join(
        os.path.dirname(__file__), 
        'backend', 
        'main.py'
    )
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify import statement exists
    assert 'from api_v2.routers.insight_router import router as insight_router' in content, \
        "main.py missing insight router import"
    
    # Verify registration statement exists
    assert 'app.include_router(insight_router)' in content, \
        "main.py missing insight router registration"


def test_main_py_insight_comment():
    """Test 19: main.py has descriptive comment for insight router."""
    main_path = os.path.join(
        os.path.dirname(__file__), 
        'backend', 
        'main.py'
    )
    
    with open(main_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Verify Phase 4B comment exists
    assert 'Phase 4B' in content or 'Insight Router' in content or 'analytics' in content, \
        "main.py should have descriptive comment for insight router"


def test_app_startup_succeeds():
    """Test 20: Application starts successfully with insight router."""
    # Create a fresh test client (triggers startup)
    try:
        client = TestClient(app)
        # If we get here, startup succeeded
        assert True
        
        # Verify health check still works
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
    except Exception as e:
        pytest.fail(f"Application startup failed: {str(e)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
