"""
Seasonal Comparison Service
Handles multi-quarter (2, 3, 4) seasonal comparison data aggregation and analysis.
"""

from typing import Dict, Any, List, Optional
import logging

from backend.api.services.seasonal_report_orchestrator import get_or_generate_seasonal_report
from backend.api.db_layer.seasonal_report import get_previous_season

logger = logging.getLogger("seasonal_comparison")


class SeasonalComparisonService:
    """
    Service for generating multi-quarter seasonal comparisons.
    Supports 2, 3, and 4 quarter comparisons with trend analysis.
    """
    
    def fetch_multiple_seasonal_reports(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int = 1
    ) -> List[Dict[str, Any]]:
        """
        Fetch multiple seasonal reports for comparison.
        
        Args:
            season_ids: List of season IDs to fetch (2, 3, or 4 seasons)
            orgunit_id: Organizational unit ID
            orgunit_type: Organizational unit type (0=Hospital, 1=Admin, 2=Dept, 3=Section)
            user_id: User ID for generation
            
        Returns:
            List of seasonal report dictionaries in order
        """
        reports = []
        
        for season_id in season_ids:
            try:
                logger.info(f"[COMPARISON] Fetching report for season_id={season_id}")
                report = get_or_generate_seasonal_report(
                    season_id=season_id,
                    orgunit_id=orgunit_id,
                    orgunit_type=orgunit_type,
                    user_id=user_id
                )
                reports.append(report)
            except Exception as e:
                logger.error(f"[COMPARISON] Failed to fetch season {season_id}: {str(e)}")
                # Return empty report structure for failed fetch
                reports.append({
                    'header': {
                        'period': f'Season-{season_id}',
                        'total_cases': 0,
                        'clinical_domain_count': 0,
                        'management_domain_count': 0,
                        'relational_domain_count': 0
                    },
                    'classification_stats': []
                })
        
        return reports
    
    def generate_3_quarter_comparison_data(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """
        Generate 3-quarter comparison data with trend analysis.
        
        Args:
            season_ids: List of exactly 3 season IDs (chronological order)
            orgunit_id: Organizational unit ID
            orgunit_type: Organizational unit type
            user_id: User ID for generation
            
        Returns:
            Dictionary with comparison data and trends
            
        Raises:
            ValueError: If season_ids doesn't contain exactly 3 seasons
        """
        if len(season_ids) != 3:
            raise ValueError(f"3-quarter comparison requires exactly 3 season IDs, got {len(season_ids)}")
        
        logger.info(f"[3Q-COMPARISON] Starting comparison for seasons: {season_ids}")
        
        # Fetch all 3 reports
        reports = self.fetch_multiple_seasonal_reports(
            season_ids=season_ids,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            user_id=user_id
        )
        
        # Extract periods
        periods = [r['header'].get('period', f'Q{i+1}') for i, r in enumerate(reports)]
        
        # Aggregate data
        domain_comparison = self._aggregate_domains(reports)
        category_comparison = self._aggregate_categories(reports)
        subcategory_comparison = self._aggregate_subcategories(reports)
        
        # Calculate trends
        trends = self._calculate_trends(reports)
        
        comparison_data = {
            'reports': reports,
            'periods': periods,
            'season_ids': season_ids,
            'domain_comparison': domain_comparison,
            'category_comparison': category_comparison,
            'subcategory_comparison': subcategory_comparison,
            'trends': trends,
            'orgunit_id': orgunit_id,
            'orgunit_type': orgunit_type,
            'orgunit_name': reports[0]['header'].get('orgunit_name', 'N/A')
        }
        
        logger.info(f"[3Q-COMPARISON] Comparison data generated successfully")
        return comparison_data
    
    def _aggregate_domains(self, reports: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Aggregate domain-level data across multiple reports.
        
        Returns:
            Dictionary mapping domain names to list of values (one per quarter)
        """
        # Collect all unique domains
        all_domains = set()
        for report in reports:
            classification_stats = report.get('classification_stats', [])
            for stat in classification_stats:
                all_domains.add(stat.get('domain_name', 'Unknown'))
        
        # Aggregate counts for each domain across all quarters
        domain_data = {}
        for domain in sorted(all_domains):
            domain_data[domain] = []
            for report in reports:
                count = 0
                classification_stats = report.get('classification_stats', [])
                for stat in classification_stats:
                    if stat.get('domain_name') == domain:
                        count += stat.get('total_count', 0)
                domain_data[domain].append(count)
        
        return domain_data
    
    def _aggregate_categories(self, reports: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Aggregate category-level data across multiple reports.
        
        Returns:
            Dictionary mapping category names to list of values (one per quarter)
        """
        # Collect all unique categories
        all_categories = set()
        for report in reports:
            classification_stats = report.get('classification_stats', [])
            for stat in classification_stats:
                domain = stat.get('domain_name', 'Unknown')
                category = stat.get('category_name', 'Unknown')
                full_name = f"{domain} - {category}"
                all_categories.add(full_name)
        
        # Aggregate counts for each category across all quarters
        category_data = {}
        for full_name in sorted(all_categories):
            category_data[full_name] = []
            domain_name, category_name = full_name.split(' - ', 1)
            
            for report in reports:
                count = 0
                classification_stats = report.get('classification_stats', [])
                for stat in classification_stats:
                    if (stat.get('domain_name') == domain_name and 
                        stat.get('category_name') == category_name):
                        count += stat.get('total_count', 0)
                category_data[full_name].append(count)
        
        return category_data
    
    def _aggregate_subcategories(self, reports: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Aggregate subcategory-level data across multiple reports.
        
        Returns:
            Dictionary mapping subcategory names to list of values (one per quarter)
        """
        # Collect all unique subcategories
        all_subcategories = set()
        for report in reports:
            classification_stats = report.get('classification_stats', [])
            for stat in classification_stats:
                category = stat.get('category_name', 'Unknown')
                subcategory = stat.get('subcategory_name', 'Unknown')
                full_name = f"{category} - {subcategory}"
                all_subcategories.add(full_name)
        
        # Aggregate counts for each subcategory across all quarters
        subcategory_data = {}
        for full_name in sorted(all_subcategories):
            subcategory_data[full_name] = []
            category_name, subcategory_name = full_name.split(' - ', 1)
            
            for report in reports:
                count = 0
                classification_stats = report.get('classification_stats', [])
                for stat in classification_stats:
                    if (stat.get('category_name') == category_name and 
                        stat.get('subcategory_name') == subcategory_name):
                        count += stat.get('total_count', 0)
                subcategory_data[full_name].append(count)
        
        return subcategory_data
    
    def _calculate_trends(self, reports: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Calculate trend indicators for key metrics.
        
        Trend logic:
        - ↑↑ (Strong Increase): Last value > First value by >20%
        - ↑ (Increase): Last value > First value by >5%
        - → (Stable): Change within ±5%
        - ↓ (Decrease): Last value < First value by >5%
        - ↓↓ (Strong Decrease): Last value < First value by >20%
        
        Returns:
            Dictionary mapping metric names to trend indicators
        """
        if len(reports) < 2:
            return {}
        
        trends = {}
        
        # Total cases trend
        total_cases = [r['header'].get('total_cases', 0) for r in reports]
        trends['total_cases'] = self._get_trend_indicator(total_cases[0], total_cases[-1])
        
        # Domain trends
        trends['clinical'] = self._get_trend_indicator(
            reports[0]['header'].get('clinical_domain_count', 0),
            reports[-1]['header'].get('clinical_domain_count', 0)
        )
        trends['management'] = self._get_trend_indicator(
            reports[0]['header'].get('management_domain_count', 0),
            reports[-1]['header'].get('management_domain_count', 0)
        )
        trends['relational'] = self._get_trend_indicator(
            reports[0]['header'].get('relational_domain_count', 0),
            reports[-1]['header'].get('relational_domain_count', 0)
        )
        
        # Severity trends
        trends['low_severity'] = self._get_trend_indicator(
            reports[0]['header'].get('low_severity_count', 0),
            reports[-1]['header'].get('low_severity_count', 0)
        )
        trends['medium_severity'] = self._get_trend_indicator(
            reports[0]['header'].get('medium_severity_count', 0),
            reports[-1]['header'].get('medium_severity_count', 0)
        )
        trends['high_severity'] = self._get_trend_indicator(
            reports[0]['header'].get('high_severity_count', 0),
            reports[-1]['header'].get('high_severity_count', 0)
        )
        
        return trends
    
    def _get_trend_indicator(self, first_value: int, last_value: int) -> str:
        """
        Get trend indicator based on first and last values.
        
        Returns:
            Trend string: '↑↑', '↑', '→', '↓', '↓↓'
        """
        if first_value == 0:
            if last_value == 0:
                return '→'
            else:
                return '↑↑'  # Any increase from zero is strong
        
        change_percent = ((last_value - first_value) / first_value) * 100
        
        if change_percent > 20:
            return '↑↑'
        elif change_percent > 5:
            return '↑'
        elif change_percent < -20:
            return '↓↓'
        elif change_percent < -5:
            return '↓'
        else:
            return '→'
    
    def calculate_percentage_changes(self, reports: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate percentage changes between first and last quarters.
        
        Args:
            reports: List of seasonal reports (2, 3, or 4 quarters)
            
        Returns:
            Dictionary mapping metric names to percentage changes
        """
        if len(reports) < 2:
            return {}
        
        first = reports[0]['header']
        last = reports[-1]['header']
        
        changes = {}
        
        # Calculate percentage change for each metric
        metrics = [
            'total_cases',
            'clinical_domain_count',
            'management_domain_count',
            'relational_domain_count',
            'low_severity_count',
            'medium_severity_count',
            'high_severity_count',
            'prevention_action_count',
            'explanation_count'
        ]
        
        for metric in metrics:
            first_value = first.get(metric, 0)
            last_value = last.get(metric, 0)
            
            if first_value == 0:
                if last_value == 0:
                    changes[metric] = 0.0
                else:
                    changes[metric] = 100.0  # Infinite increase represented as 100%
            else:
                changes[metric] = round(((last_value - first_value) / first_value) * 100, 2)
        
        return changes
    
    def aggregate_domain_data(self, reports: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Public wrapper for _aggregate_domains for API compatibility.
        
        Returns:
            Dictionary mapping domain names to list of values (one per quarter)
        """
        return self._aggregate_domains(reports)
    
    def aggregate_category_data(self, reports: List[Dict[str, Any]]) -> Dict[str, List[int]]:
        """
        Public wrapper for _aggregate_categories for API compatibility.
        
        Returns:
            Dictionary mapping category names to list of values (one per quarter)
        """
        return self._aggregate_categories(reports)
    
    def generate_4_quarter_comparison_data(
        self,
        season_ids: List[int],
        orgunit_id: int,
        orgunit_type: int,
        user_id: int = 1
    ) -> Dict[str, Any]:
        """
        Generate 4-quarter comparison data with trend analysis.
        
        Args:
            season_ids: List of exactly 4 season IDs (chronological order)
            orgunit_id: Organizational unit ID
            orgunit_type: Organizational unit type
            user_id: User ID for generation
            
        Returns:
            Dictionary with comparison data and trends
            
        Raises:
            ValueError: If season_ids doesn't contain exactly 4 seasons
        """
        if len(season_ids) != 4:
            raise ValueError(f"4-quarter comparison requires exactly 4 season IDs, got {len(season_ids)}")
        
        logger.info(f"[4Q-COMPARISON] Starting comparison for seasons: {season_ids}")
        
        # Fetch all 4 reports
        reports = self.fetch_multiple_seasonal_reports(
            season_ids=season_ids,
            orgunit_id=orgunit_id,
            orgunit_type=orgunit_type,
            user_id=user_id
        )
        
        # Extract periods
        periods = [r['header'].get('period', f'Q{i+1}') for i, r in enumerate(reports)]
        
        # Aggregate data
        domain_comparison = self._aggregate_domains(reports)
        category_comparison = self._aggregate_categories(reports)
        subcategory_comparison = self._aggregate_subcategories(reports)
        
        # Calculate trends
        trends = self._calculate_trends(reports)
        
        # Calculate yearly totals
        yearly_totals = self._calculate_yearly_totals(reports)
        
        comparison_data = {
            'reports': reports,
            'periods': periods,
            'season_ids': season_ids,
            'domain_comparison': domain_comparison,
            'category_comparison': category_comparison,
            'subcategory_comparison': subcategory_comparison,
            'trends': trends,
            'yearly_totals': yearly_totals,
            'orgunit_id': orgunit_id,
            'orgunit_type': orgunit_type,
            'orgunit_name': reports[0]['header'].get('orgunit_name', 'N/A')
        }
        
        logger.info(f"[4Q-COMPARISON] Comparison data generated successfully")
        return comparison_data
    
    def _calculate_yearly_totals(self, reports: List[Dict[str, Any]]) -> Dict[str, int]:
        """
        Calculate yearly totals across all 4 quarters.
        
        Returns:
            Dictionary with yearly totals for key metrics
        """
        yearly_totals = {
            'total_cases': sum(r['header'].get('total_cases', 0) for r in reports),
            'clinical': sum(r['header'].get('clinical_domain_count', 0) for r in reports),
            'management': sum(r['header'].get('management_domain_count', 0) for r in reports),
            'relational': sum(r['header'].get('relational_domain_count', 0) for r in reports),
            'low_severity': sum(r['header'].get('low_severity_count', 0) for r in reports),
            'medium_severity': sum(r['header'].get('medium_severity_count', 0) for r in reports),
            'high_severity': sum(r['header'].get('high_severity_count', 0) for r in reports),
        }
        
        return yearly_totals


# Singleton instance
seasonal_comparison_service = SeasonalComparisonService()
