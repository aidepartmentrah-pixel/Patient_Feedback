"""Test hardware config service functionality."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.services.hardware_config_service import HardwareConfigService

def test_service():
    # Test get all configs
    print('Testing HardwareConfigService...')
    result = HardwareConfigService.get_all_configurations()
    print(f'Total configs: {result["total_configs"]}')
    print(f'Groups: {list(result["groups"].keys())}')

    # Test deployment summary
    summary = HardwareConfigService.get_deployment_summary()
    print(f'Deployment mode: {summary["deployment_mode"]}')
    print(f'DB Server: {summary["database"]["server"]}')

    # Test database connection
    print('Testing database connection...')
    db_test = HardwareConfigService.test_database_connection()
    print(f'DB test: {db_test["success"]} - {db_test["message"]}')

    print('All tests passed!')

if __name__ == '__main__':
    test_service()
